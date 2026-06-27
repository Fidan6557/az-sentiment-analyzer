from types import SimpleNamespace

import train


def make_args():
    return SimpleNamespace(
        output_dir="./az-sentiment-model",
        epochs=1,
        batch_size=2,
        lr=2e-5,
        seed=42,
    )


def test_build_training_arguments_uses_eval_strategy(monkeypatch):
    captured = {}

    class FakeTrainingArguments:
        def __init__(self, *, eval_strategy, **kwargs):
            captured.update(kwargs)
            captured["eval_strategy"] = eval_strategy

    monkeypatch.setattr(train, "TrainingArguments", FakeTrainingArguments)
    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: False)

    train.build_training_arguments(make_args())

    assert captured["eval_strategy"] == "epoch"
    assert "evaluation_strategy" not in captured
    assert captured["fp16"] is False


def test_build_training_arguments_falls_back_to_evaluation_strategy(monkeypatch):
    captured = {}

    class FakeTrainingArguments:
        def __init__(self, *, evaluation_strategy, **kwargs):
            captured.update(kwargs)
            captured["evaluation_strategy"] = evaluation_strategy

    monkeypatch.setattr(train, "TrainingArguments", FakeTrainingArguments)
    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: True)

    train.build_training_arguments(make_args())

    assert captured["evaluation_strategy"] == "epoch"
    assert "eval_strategy" not in captured
    assert captured["fp16"] is True
