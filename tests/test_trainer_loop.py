"""Trainer.run end to end on a tiny model with stubbed data/tokenizer (no downloads).

Covers: step-0 eval on a fresh run, exact resume after a crash without duplicated log rows,
budget-signature refusal, staged freezing of old rows, and identical token/step budgets
across an Exp1-like and an Exp2-like arm.
"""

import json
import os

import numpy as np
import pytest
import torch

import vocabx.training.trainer as T
from vocabx.config import Config, StageSpec
from vocabx.data.packing import PackedCorpus
from vocabx.modeling.surgery import install_split_embedding


class _Tok:
    def __init__(self, n):
        self.n = n

    def get_added_tokens_decoder(self):
        return {}

    def get_vocab(self, with_added_tokens=True):
        return {f"t{i}": i for i in range(self.n)}


class _Bundle:
    def __init__(self, n_new):
        self.n_base, self.n_new = 300, n_new
        self.vocab_size = 300 + n_new
        self.new_tokens = [f"n{i}" for i in range(n_new)]
        self.tokenizer = _Tok(self.vocab_size)
        self.meta = {}


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    from transformers import LlamaConfig, LlamaForCausalLM

    n_new = {"value": 0}

    def build_model(cfg, bundle, device):
        torch.manual_seed(0)
        c = LlamaConfig(vocab_size=300, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                        num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True)
        c._attn_implementation = "eager"
        m = LlamaForCausalLM(c)
        m.tie_weights()
        if bundle.n_new:
            W = torch.cat([m.get_input_embeddings().weight.data, m.get_input_embeddings().weight.data[: bundle.n_new]])
            emb = torch.nn.Embedding(*W.shape)
            emb.weight.data.copy_(W)
            m.set_input_embeddings(emb)
        install_split_embedding(m, 300)
        m.config.use_cache = False
        return m.to(device), {}

    def corpus(vocab):
        rng = np.random.RandomState(0)
        lens = rng.randint(20, 120, 400)
        toks = rng.randint(1, vocab, lens.sum()).astype(np.uint16)
        offs = np.zeros(401, dtype=np.int64)
        np.cumsum(lens, out=offs[1:])
        return PackedCorpus(toks, offs, (lens * 3).astype(np.int64), "el")

    monkeypatch.setattr(T, "build_tokenizer", lambda cfg: _Bundle(n_new["value"]))
    monkeypatch.setattr(T, "prepare_data", lambda cfg: None)
    monkeypatch.setattr(T, "build_model", build_model)
    monkeypatch.setattr(T, "ensure_packed", lambda cfg, b, langs: {"el": corpus(b.vocab_size)})
    monkeypatch.setattr(T, "load_eval_chunks", lambda cfg, s, n=None: ["chunk"])
    monkeypatch.setattr(T, "score_chunks", lambda *a, **k: [])
    monkeypatch.setattr(T, "summarize_records", lambda recs: {"x": {"bpb": 1.0, "language": "el"}})
    monkeypatch.setattr(T, "save_model", lambda *a, **k: None)

    def make(name="run", max_steps=8, new=0, stages=None):
        n_new["value"] = new
        cfg = Config()
        cfg.run_name, cfg.workdir = name, str(tmp_path)
        cfg.eval.dev_sets, cfg.eval.sets = ["x"], []
        tc = cfg.train
        tc.seq_len, tc.micro_batch_size, tc.grad_accum, tc.max_steps = 16, 2, 1, max_steps
        tc.precision, tc.eval_every, tc.log_every, tc.checkpoint_every_min = "fp32", 2, 1, 0.0
        if stages:
            tc.stages = stages
        return T.Trainer(cfg, device="cpu")

    return make


def _steps(path):
    return [json.loads(l)["step"] for l in open(path)]


def test_fresh_run_has_step0_and_resume_has_no_duplicates(stubbed):
    tr = stubbed()
    real_save, real_step = tr.save_checkpoint, tr.train_step
    tr.save_checkpoint = lambda: real_save() if tr.meter.steps == 3 else None  # one checkpoint, at step 3

    def crash(step):
        if step == 5:
            raise RuntimeError("disconnect")
        return real_step(step)

    tr.train_step = crash
    with pytest.raises(RuntimeError):
        tr.run(final_eval=False)
    rd = tr.run_dir
    assert _steps(os.path.join(rd, "dev_evals.jsonl"))[0] == 0  # step-0 eval on the fresh run

    tr2 = stubbed()
    s = tr2.run(final_eval=False)
    assert _steps(os.path.join(rd, "dev_evals.jsonl")) == [0, 2, 4, 6, 8]
    assert _steps(os.path.join(rd, "train_log.jsonl")) == list(range(1, 9))
    assert s["budget"]["steps"] == 8 and s["budget"]["tokens"] == 8 * 2 * 16
    assert not os.path.exists(os.path.join(rd, "checkpoint.pt"))


def test_finished_run_with_different_budget_is_refused(stubbed):
    stubbed(name="b", max_steps=4).run(final_eval=False)
    with pytest.raises(ValueError, match="different budget"):
        T._check_finished(stubbed(name="b", max_steps=6).cfg)
    assert T._check_finished(stubbed(name="b", max_steps=4).cfg)["max_steps"] == 4


def test_staged_freezes_old_rows_and_arms_share_budgets(stubbed):
    stages = [StageSpec("a", 0.5, ["emb_new"]), StageSpec("b", 1.0)]
    tr = stubbed(name="exp2", max_steps=4, new=20, stages=stages)
    old0, new0, body0 = tr.split.old.detach().clone(), tr.split.new.detach().clone(), tr.groups["body"][0].detach().clone()
    tr.train_step(0)
    tr.train_step(1)
    assert torch.equal(tr.split.old, old0)  # frozen: no update, no weight decay
    assert torch.equal(tr.groups["body"][0], body0)
    assert not torch.equal(tr.split.new, new0)
    tr.train_step(2)
    assert not torch.equal(tr.split.old, old0)  # unfrozen in stage b
    e1 = stubbed(name="exp1", max_steps=4).run(final_eval=False)
    e2 = stubbed(name="exp2b", max_steps=4, new=20).run(final_eval=False)
    assert (e1["budget"]["tokens"], e1["budget"]["steps"]) == (e2["budget"]["tokens"], e2["budget"]["steps"])
    assert e2["budget"]["flops"] > e1["budget"]["flops"]  # larger vocab costs more and is reported
