"""Load the configured Retail classifier and run a seeded 10+10 online sample."""

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from base import Config
from rag_qa.core.query_classifier import QueryClassifier


def main():
    config = Config()
    rows = [
        json.loads(line)
        for line in (ROOT / "rag_qa/classify_data/retail_manual_test_50.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    rng = random.Random(20260923)
    selected = []
    for label in ("general", "internal"):
        selected.extend(rng.sample([row for row in rows if row["label"] == label], 10))
    classifier = QueryClassifier(config.bert_intent_cls)
    results = []
    for row in selected:
        details = classifier.predict_details(row["query"])
        results.append({**row, **details, "correct": details["routed_prediction"] == row["label"]})
    expected_path = (ROOT / "rag_qa/models/retail-query-classifier").resolve()
    loaded_path = Path(classifier.model_path).resolve()
    summary = {
        "status": "PASS" if (
            loaded_path == expected_path
            and len(results) == 20
            and sum(row["label"] == "general" for row in results) == 10
            and sum(row["label"] == "internal" for row in results) == 10
        ) else "FAIL",
        "quality_status": "PASS" if all(row["correct"] for row in results) else "FAIL",
        "seed": 20260923,
        "configured_model_path": str(loaded_path),
        "expected_retail_model_path": str(expected_path),
        "path_matches": loaded_path == expected_path,
        "sample_count": len(results),
        "general_count": sum(row["label"] == "general" for row in results),
        "internal_count": sum(row["label"] == "internal" for row in results),
        "correct": sum(row["correct"] for row in results),
        "internal_to_general": sum(
            row["label"] == "internal" and row["routed_prediction"] == "general" for row in results
        ),
    }
    output = {"summary": summary, "results": results}
    (ROOT / "RETAILRAG_CLASSIFIER_ONLINE_RESULTS.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    if summary["status"] != "PASS":
        raise RuntimeError("Configured online classifier verification failed")


if __name__ == "__main__":
    main()
