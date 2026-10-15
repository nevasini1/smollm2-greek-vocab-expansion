import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture(scope="session")
def tiny_model_factory():
    """Factory for a tiny random tied-embedding Llama (no download)."""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    def make(vocab: int = 300, seed: int = 0):
        torch.manual_seed(seed)
        cfg = LlamaConfig(
            vocab_size=vocab,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=256,
            tie_word_embeddings=True,
        )
        cfg._attn_implementation = "eager"
        m = LlamaForCausalLM(cfg).float().eval()
        m.tie_weights()
        return m

    return make


@pytest.fixture(scope="session")
def smollm_tokenizer():
    """The real SmolLM2 tokenizer (network on first use; cached by huggingface_hub)."""
    try:
        from vocabx.tokenization.base import load_base_tokenizer

        return load_base_tokenizer("HuggingFaceTB/SmolLM2-135M")
    except Exception as e:  # pragma: no cover
        pytest.skip(f"SmolLM2 tokenizer unavailable: {e}")


GREEK_SAMPLE = [
    "Η Ελλάδα είναι χώρα της νοτιοανατολικής Ευρώπης, στο νοτιότερο άκρο της Βαλκανικής χερσονήσου.",
    "Η πρωτεύουσα και μεγαλύτερη πόλη της χώρας είναι η Αθήνα, ενώ η δεύτερη μεγαλύτερη είναι η Θεσσαλονίκη.",
    "Ο πληθυσμός της χώρας ανέρχεται σε περίπου δέκα εκατομμύρια κατοίκους σύμφωνα με την απογραφή.",
    "Η ελληνική γλώσσα είναι μία από τις αρχαιότερες καταγεγραμμένες γλώσσες του κόσμου.",
    "Το κλίμα της Ελλάδας είναι κυρίως μεσογειακό, με ήπιους και υγρούς χειμώνες και θερμά ξηρά καλοκαίρια.",
    "Η Ακρόπολη των Αθηνών είναι ένας από τους σημαντικότερους αρχαιολογικούς χώρους της Ευρώπης.",
    "Οι Ολυμπιακοί Αγώνες ξεκίνησαν στην αρχαία Ολυμπία και αναβίωσαν στην Αθήνα το 1896.",
    "Τα ελληνικά νησιά προσελκύουν κάθε χρόνο εκατομμύρια επισκέπτες από όλο τον κόσμο.",
] * 30

ENGLISH_SAMPLE = [
    "The quick brown fox jumps over the lazy dog.",
    "Greece is a country in Southeast Europe; its capital is Athens (population ~3.1M).",
    "def f(x):\n    return {'a': [1, 2, 3]}  # 2024-10-14\n",
    "Machine learning models are trained on large corpora of text, code and math.",
    "naïve café résumé coöperate — “quotes” and ‘single quotes’…",
]
