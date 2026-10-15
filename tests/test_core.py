"""Config, text processing, schedule, streams, bootstrap (pure-Python, no download)."""

import os

import numpy as np
import pytest

from vocabx.config import ScheduleConfig, load_config
from vocabx.data.packing import Mixture, PackedCorpus, TokenStream
from vocabx.data.text import chunk_text, has_greek, normalize_text, script_ratio
from vocabx.evaluation.stats import bootstrap_bpb, ci, macro
from vocabx.training.schedule import lr_multiplier

ROOT = os.path.join(os.path.dirname(__file__), "..")


def test_config_inheritance_overrides_and_strictness(tmp_path):
    cfg = load_config(os.path.join(ROOT, "configs/ablations/exp2_staged.yaml"), ["train.optim.lr=1e-3"])
    assert cfg.run_name == "exp2_staged"
    assert cfg.tokenizer.extender == "continued_bpe"  # from exp2_extend.yaml
    assert cfg.train.seq_len == 1024  # from base.yaml
    assert cfg.train.optim.lr == pytest.approx(1e-3)
    assert [s.name for s in cfg.train.stages] == ["new_rows_only", "full"]
    with pytest.raises(KeyError):
        load_config(os.path.join(ROOT, "configs/exp1_finetune.yaml"), ["train.max_step=3"])
    a = load_config(os.path.join(ROOT, "configs/exp1_finetune.yaml"))
    b = load_config(os.path.join(ROOT, "configs/exp2_extend.yaml"))
    assert a.key("data") == b.key("data")  # both arms share one data artifact
    assert a.key("tokenizer").startswith("base-") and not b.key("tokenizer").startswith("base")
    other = load_config(os.path.join(ROOT, "configs/exp1_finetune.yaml"), ["model.revision=deadbeef"])
    assert other.key("tokenizer") != a.key("tokenizer") and other.key("data") != a.key("data")


def test_yaml_numbers_diamond_inheritance_and_plugins(tmp_path):
    from vocabx.modeling.init import INITIALIZERS
    from vocabx.registry import import_plugins

    (tmp_path / "base.yaml").write_text("run_name: b\ntrain: {optim: {lr: 1e-3}}\n")
    (tmp_path / "mix_a.yaml").write_text("inherits: base.yaml\ntrain: {english_ratio: 0.2}\n")
    (tmp_path / "mix_b.yaml").write_text("inherits: base.yaml\ntrain: {group_lr_mult: {emb_new: 3e0}}\n")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "plug.py").write_text(
        "from vocabx.modeling.init import INITIALIZERS\n"
        "@INITIALIZERS.register('test_zero_plugin')\n"
        "def f(ctx):\n    return ctx.E_old.new_zeros(len(ctx.new_tokens), ctx.E_old.shape[1])\n"
    )
    (sub / "child.yaml").write_text("inherits: [../mix_a.yaml, ../mix_b.yaml]\nplugins: [plug.py]\ninit: {method: test_zero_plugin, params: {alpha: 5e-1}}\n")
    cfg = load_config(str(sub / "child.yaml"), ["train.max_steps=1.2e2"])
    assert cfg.train.optim.lr == pytest.approx(1e-3)
    assert cfg.train.english_ratio == pytest.approx(0.2)
    assert cfg.train.group_lr_mult["emb_new"] == pytest.approx(3.0) and isinstance(cfg.train.group_lr_mult["emb_new"], float)
    assert cfg.init.params["alpha"] == pytest.approx(0.5)
    assert cfg.train.max_steps == 120 and isinstance(cfg.train.max_steps, int)
    assert cfg.plugins == [str(sub / "plug.py")]  # resolved relative to the declaring YAML
    import_plugins(cfg.plugins)
    import_plugins(cfg.plugins)  # idempotent: a second run in the same plan must not crash
    assert "test_zero_plugin" in INITIALIZERS


def test_all_shipped_configs_load():
    import glob

    for p in glob.glob(os.path.join(ROOT, "configs", "**", "*.yaml"), recursive=True):
        if os.sep + "plans" + os.sep in p:
            continue
        load_config(p)


def test_normalize_nfc_and_controls():
    decomposed = "άρα"  # alpha + combining acute
    assert normalize_text(decomposed) == "άρα"
    assert normalize_text("a\x04b\r\nc\x1d") == "ab\nc"
    assert normalize_text(";") == ";"  # Greek question mark -> semicolon under NFC
    assert has_greek("abc β") and not has_greek("abc")
    assert script_ratio("Καλημέρα κόσμε", "greek") == 1.0


def test_chunking_is_exact_and_bounded():
    text = "\n".join(f"Πρόταση νούμερο {i}. Και μια ακόμη πρόταση εδώ." * 3 for i in range(40))
    chunks = chunk_text(text, max_chars=200, max_chunks=50, fits=lambda t: len(t) <= 200)
    assert chunks and all(len(c) <= 200 for c in chunks)
    assert all(c in text for c in chunks)  # cut from the original text, so bytes are identical


def test_wsd_schedule_shape():
    sc = ScheduleConfig(name="wsd", warmup_frac=0.1, decay_frac=0.2)
    lrs = [lr_multiplier(s, 100, sc) for s in range(100)]
    assert lrs[0] == pytest.approx(0.1)
    assert lrs[9] == pytest.approx(1.0) and lrs[50] == pytest.approx(1.0)
    assert lrs[80] == pytest.approx(1.0) and lrs[-1] == pytest.approx(1 / 20)  # last step > 0
    assert all(a > b for a, b in zip(lrs[80:], lrs[81:]))
    # a group unfrozen at step 10 re-warms but decays on the same absolute steps
    late = [lr_multiplier(s, 100, sc, start_step=10) for s in range(100)]
    assert late[5] == 0.0 and late[10] == pytest.approx(0.1) and late[-1] == pytest.approx(lrs[-1])


def _corpus(n_docs=50, seed=0):
    rng = np.random.RandomState(seed)
    lens = rng.randint(5, 40, size=n_docs)
    toks = rng.randint(1, 1000, size=lens.sum()).astype(np.uint16)
    off = np.zeros(n_docs + 1, dtype=np.int64)
    off[1:] = np.cumsum(lens)
    return PackedCorpus(toks, off, lens * 2, "el")


def test_stream_deterministic_resumable_and_epochs():
    c = _corpus()
    a, b = TokenStream(c, 16, seed=3), TokenStream(c, 16, seed=3)
    blocks_a = [a.next_block().copy() for _ in range(10)]
    st = a.state()
    nxt = a.next_block().copy()
    for _ in range(10):
        b.next_block()
    b2 = TokenStream(c, 16, seed=3)
    b2.load_state(st)
    assert np.array_equal(b2.next_block(), nxt)
    assert all(len(x) == 17 for x in blocks_a)
    for _ in range(200):  # crosses an epoch boundary
        a.next_block()
    assert a.epoch >= 1
    assert not np.array_equal(TokenStream(c, 16, seed=4).next_block(), TokenStream(c, 16, seed=3).next_block())


def test_mixture_exact_ratio():
    c = _corpus()
    mix = Mixture({"el": TokenStream(c, 8, 0), "en": TokenStream(c, 8, 1)}, 0.2)
    langs = [mix.next()[0] for _ in range(1000)]
    assert langs.count("en") == 200


def test_bootstrap_paired_point_and_ci():
    recs_a, recs_b = [], []
    rng = np.random.RandomState(0)
    for d in range(200):
        nb = int(rng.randint(100, 1000))
        nll_a = nb * 0.7
        recs_a.append({"set": "s", "doc_id": str(d), "nll": nll_a, "n_bytes": nb, "language": "el"})
        recs_b.append({"set": "s", "doc_id": str(d), "nll": nll_a * (0.9 + 0.01 * rng.randn()), "n_bytes": nb, "language": "el"})
    boot = bootstrap_bpb({"a": recs_a, "b": recs_b}, ["s"], n_boot=500)
    pa = sum(r["nll"] for r in recs_a) / sum(r["n_bytes"] for r in recs_a) / np.log(2)
    assert boot["a"]["s"][0] == pytest.approx(pa)
    p, lo, hi = ci(macro(boot["b"], ["s"]) - macro(boot["a"], ["s"]))
    assert lo <= p <= hi and hi < 0  # b is ~10% better and the paired CI excludes 0
