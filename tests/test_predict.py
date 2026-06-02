"""
tests/test_predict.py — Smoke tests for the predict() helper.

Run:
    pytest tests/
"""

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────
# Mock helpers (no Streamlit / real model needed)
# ─────────────────────────────────────────────

def make_mock_model(logits):
    """Return a callable that mimics HuggingFace model output."""
    class FakeOutput:
        def __init__(self):
            self.logits = torch.tensor([logits], dtype=torch.float32)

    class FakeModel:
        def __call__(self, **kwargs):
            return FakeOutput()
        def eval(self):
            return self

    return FakeModel()


def make_mock_tokenizer():
    class FakeTokenizer:
        def __call__(self, text, **kwargs):
            return {
                "input_ids": torch.zeros(1, 5, dtype=torch.long),
                "attention_mask": torch.ones(1, 5, dtype=torch.long),
            }
    return FakeTokenizer()


# ── Standalone predict() (mirrors app.py logic) ──────────────

def predict(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0]
    return probs.argmax().item(), probs


# ─────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────

def test_positive_prediction():
    idx, probs = predict("yaxşı məhsuldur", make_mock_tokenizer(), make_mock_model([0.1, 0.9]))
    assert idx == 1, "Should predict Positive (1)"
    assert probs.shape == (2,)


def test_negative_prediction():
    idx, probs = predict("çox pis idi", make_mock_tokenizer(), make_mock_model([0.9, 0.1]))
    assert idx == 0, "Should predict Negative (0)"


def test_probs_sum_to_one():
    _, probs = predict("normal", make_mock_tokenizer(), make_mock_model([0.4, 0.6]))
    assert abs(probs.sum().item() - 1.0) < 1e-5


def test_low_confidence_detectable():
    """Both classes near 50% → confidence below 0.60 threshold."""
    idx, probs = predict("bilmirəm", make_mock_tokenizer(), make_mock_model([0.48, 0.52]))
    assert probs[idx].item() < 0.60
