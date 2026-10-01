"""Fine-tune the inherited binary BERT architecture for RetailRAG."""

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_MODEL = ROOT / "rag_qa" / "core" / "bert_query_classifier"
DEFAULT_OUTPUT = ROOT / "rag_qa" / "models" / "retail-query-classifier"
DATA_DIR = ROOT / "rag_qa" / "classify_data"
LABEL_TO_ID = {"general": 0, "internal": 1}
ID_TO_LABEL = {0: "general", 1: "internal"}
SEED = 20260923


class JsonlDataset(Dataset):
    def __init__(self, path, tokenizer):
        self.path = Path(path)
        self.rows = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        encoded = self.tokenizer(row["query"], truncation=True, max_length=128)
        encoded["labels"] = LABEL_TO_ID[row["label"]]
        return encoded


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics_from_predictions(labels, predictions):
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", pos_label=1, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "internal_precision": float(precision),
        "internal_recall": float(recall),
        "internal_f1": float(f1),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "internal_to_general": int(sum((label == 1 and pred == 0) for label, pred in zip(labels, predictions))),
    }


def compute_metrics(eval_prediction):
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)
    values = metrics_from_predictions(labels, predictions)
    return {key: value for key, value in values.items() if key != "confusion_matrix"}


def evaluate_dataset(trainer, dataset):
    output = trainer.predict(dataset)
    predictions = np.argmax(output.predictions, axis=-1)
    return metrics_from_predictions(output.label_ids.tolist(), predictions.tolist())


def main():
    parser = argparse.ArgumentParser(description="Train RetailRAG general/internal BERT classifier")
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    base_model = args.base_model.resolve()
    output_dir = args.output.resolve()
    if base_model == output_dir:
        raise ValueError("Retail model output must not overwrite the inherited classifier")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    set_seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)

    train_path = DATA_DIR / "retail_query_classifier_train.jsonl"
    dev_path = DATA_DIR / "retail_query_classifier_dev.jsonl"
    test_path = DATA_DIR / "retail_query_classifier_test.jsonl"

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=2,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    train_dataset = JsonlDataset(train_path, tokenizer)
    dev_dataset = JsonlDataset(dev_path, tokenizer)
    test_dataset = JsonlDataset(test_path, tokenizer)

    run_dir = output_dir.parent / "retail-query-classifier-run"
    training_args = TrainingArguments(
        output_dir=str(run_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
        seed=SEED,
        data_seed=SEED,
        use_cpu=not torch.cuda.is_available(),
        disable_tqdm=True,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(dev_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    print(f"Seed: {SEED}")
    print(f"Device: {trainer.args.device}")
    trainer.train()

    dev_metrics = evaluate_dataset(trainer, dev_dataset)
    test_metrics = evaluate_dataset(trainer, test_dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    (output_dir / "label_map.json").write_text(
        json.dumps(LABEL_TO_ID, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    run_summary = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": str(base_model),
        "output_dir": str(output_dir),
        "seed": SEED,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_size": len(train_dataset),
        "validation_size": len(dev_dataset),
        "test_size": len(test_dataset),
        "data_sha256": {
            "train": sha256(train_path),
            "validation": sha256(dev_path),
            "test": sha256(test_path),
        },
        "validation_metrics": dev_metrics,
        "test_metrics": test_metrics,
    }
    (output_dir / "training_metrics.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Validation metrics:", json.dumps(dev_metrics, ensure_ascii=False))
    print("Test metrics:", json.dumps(test_metrics, ensure_ascii=False))
    print(f"Saved Retail classifier to: {output_dir}")


if __name__ == "__main__":
    main()
