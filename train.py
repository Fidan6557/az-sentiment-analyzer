"""
train.py — XLM-RoBERTa fine-tuning on Azerbaijani sentiment dataset

Dataset : hajili/azerbaijani_review_sentiment_classification (HuggingFace)
Labels  : 0 = Mənfi (Negative), 1 = Müsbət (Positive)
          Score ≤ 2 → 0 (Negative)
          Score ≥ 4 → 1 (Positive)
          Score = 3 (neutral) → excluded

Usage:
    python train.py
    python train.py --epochs 4 --batch_size 16 --lr 2e-5 --output_dir ./az-sentiment-model
"""

import argparse
import os
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report


# ─────────────────────────────────────────────
# 1. Args
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune XLM-RoBERTa for Azerbaijani sentiment")
    parser.add_argument("--base_model",  type=str, default="xlm-roberta-base")
    parser.add_argument("--dataset",     type=str, default="hajili/azerbaijani_review_sentiment_classification")
    parser.add_argument("--output_dir",  type=str, default="./az-sentiment-model")
    parser.add_argument("--epochs",      type=int, default=4)
    parser.add_argument("--batch_size",  type=int, default=16)
    parser.add_argument("--lr",          type=float, default=2e-5)
    parser.add_argument("--max_length",  type=int, default=128)
    parser.add_argument("--seed",        type=int, default=42)
    return parser.parse_args()


# ─────────────────────────────────────────────
# 2. Dataset helpers
# ─────────────────────────────────────────────

def score_to_label(example):
    """
    Convert raw review score to binary sentiment label.
    Returns None for neutral (score == 3) so we can filter them out.
    """
    score = example.get("label") or example.get("score") or example.get("rating")
    if score is None:
        return {"binary_label": None}
    score = int(score)
    if score <= 2:
        return {"binary_label": 0}   # Negative / Mənfi
    elif score >= 4:
        return {"binary_label": 1}   # Positive / Müsbət
    else:
        return {"binary_label": None}  # Neutral — will be filtered


def load_and_prepare_dataset(dataset_name: str):
    print(f"[INFO] Loading dataset: {dataset_name}")
    raw = load_dataset(dataset_name)

    # Detect the text column name
    sample = raw["train"][0]
    text_col = None
    for candidate in ["text", "review", "comment", "content", "sentence"]:
        if candidate in sample:
            text_col = candidate
            break
    if text_col is None:
        raise ValueError(f"Cannot detect text column. Available columns: {list(sample.keys())}")
    print(f"[INFO] Text column detected: '{text_col}'")

    # Map to binary labels and filter neutrals
    raw = raw.map(score_to_label)
    raw = raw.filter(lambda x: x["binary_label"] is not None)

    print(f"[INFO] After filtering neutrals:")
    for split in raw:
        labels = raw[split]["binary_label"]
        neg = labels.count(0)
        pos = labels.count(1)
        print(f"  {split}: {len(labels)} samples  (Negative={neg}, Positive={pos})")

    return raw, text_col


# ─────────────────────────────────────────────
# 3. Tokenisation
# ─────────────────────────────────────────────

def tokenize_dataset(raw_dataset, tokenizer, text_col: str, max_length: int):
    def tokenize_fn(examples):
        return tokenizer(
            examples[text_col],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized = raw_dataset.map(tokenize_fn, batched=True)
    tokenized = tokenized.rename_column("binary_label", "labels")
    tokenized.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )
    return tokenized


# ─────────────────────────────────────────────
# 4. Metrics
# ─────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}


# ─────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Load tokenizer & model ──────────────────
    print(f"[INFO] Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=2,
        id2label={0: "Mənfi", 1: "Müsbət"},
        label2id={"Mənfi": 0, "Müsbət": 1},
    )

    # ── Dataset ─────────────────────────────────
    raw_dataset, text_col = load_and_prepare_dataset(args.dataset)
    tokenized = tokenize_dataset(raw_dataset, tokenizer, text_col, args.max_length)

    train_ds = tokenized["train"]
    # Use "validation" if present, otherwise split from train
    if "validation" in tokenized:
        eval_ds = tokenized["validation"]
    elif "test" in tokenized:
        eval_ds = tokenized["test"]
    else:
        split = train_ds.train_test_split(test_size=0.1, seed=args.seed)
        train_ds = split["train"]
        eval_ds  = split["test"]

    print(f"[INFO] Train size: {len(train_ds)} | Eval size: {len(eval_ds)}")

    # ── Training arguments ───────────────────────
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        seed=args.seed,
        fp16=torch.cuda.is_available(),   # mixed precision if GPU available
        report_to="none",                 # disable wandb / mlflow
    )

    # ── Trainer ──────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("[INFO] Starting training...")
    trainer.train()

    # ── Evaluate & report ────────────────────────
    print("[INFO] Evaluating on eval set...")
    results = trainer.evaluate()
    print(f"\n{'='*40}")
    print(f"  Accuracy : {results['eval_accuracy']:.4f}")
    print(f"  F1 (weighted): {results['eval_f1']:.4f}")
    print(f"{'='*40}\n")

    # Detailed classification report
    preds_output = trainer.predict(eval_ds)
    preds = np.argmax(preds_output.predictions, axis=-1)
    labels = preds_output.label_ids
    print(classification_report(labels, preds, target_names=["Mənfi (0)", "Müsbət (1)"]))

    # ── Save ─────────────────────────────────────
    print(f"[INFO] Saving model and tokenizer to: {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("[INFO] Done ✅")


if __name__ == "__main__":
    main()
