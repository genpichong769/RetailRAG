"""Evaluate the trained Retail classifier on the independent 50-query set."""

import csv
import json
import sys
from pathlib import Path

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from base import Config
from rag_qa.core.query_classifier import QueryClassifier


DATA_PATH = ROOT / "rag_qa" / "classify_data" / "retail_manual_test_50.jsonl"
OUTPUT_CSV = ROOT / "RETAILRAG_CLASSIFIER_MANUAL_RESULTS.csv"
OUTPUT_JSON = ROOT / "RETAILRAG_CLASSIFIER_MANUAL_METRICS.json"


def metrics(expected, predicted):
    y_true = [1 if value == "internal" else 0 for value in expected]
    y_pred = [1 if value == "internal" else 0 for value in predicted]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "internal_precision": float(precision),
        "internal_recall": float(recall),
        "internal_f1": float(f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "internal_to_general": int(sum(a == 1 and b == 0 for a, b in zip(y_true, y_pred))),
    }


def main():
    rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines()]
    classifier = QueryClassifier(Config().bert_intent_cls)
    results = []
    for index, row in enumerate(rows, 1):
        details = classifier.predict_details(row["query"])
        results.append(
            {
                "id": index,
                "query": row["query"],
                "expected": row["label"],
                **details,
                "model_correct": details["model_prediction"] == row["label"],
                "routed_correct": details["routed_prediction"] == row["label"],
            }
        )

    model_metrics = metrics(
        [row["expected"] for row in results],
        [row["model_prediction"] for row in results],
    )
    routed_metrics = metrics(
        [row["expected"] for row in results],
        [row["routed_prediction"] for row in results],
    )
    fieldnames = list(results[0].keys())
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    summary = {
        "model_path": Config().bert_intent_cls,
        "sample_count": len(results),
        "model_metrics": model_metrics,
        "routed_metrics": routed_metrics,
        "rule_matches": sum(bool(row["rule_match"]) for row in results),
        "rule_overrides": sum(row["rule_overrode_model"] for row in results),
    }
    OUTPUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for row in results:
        print(
            f"{row['id']:02d} expected={row['expected']} model={row['model_prediction']} "
            f"confidence={row['model_confidence']:.4f} rule={row['rule_match'] or '-'} "
            f"routed={row['routed_prediction']} | {row['query']}"
        )


if __name__ == "__main__":
    main()
