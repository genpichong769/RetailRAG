"""Read-only checks for RetailRAG's frozen benchmark and final deliverables."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FINAL = (
    "RETAILRAG_QWEN37_FINAL_EVALUATION_REPORT.md",
    "RETAILRAG_QWEN37_FINAL_METRICS.json",
    "RETAILRAG_QWEN37_FINAL_350_RESULTS.json",
    "RETAILRAG_QWEN37_FINAL_GENERATION_80.json",
    "RETAILRAG_QWEN37_FINAL_RETRIEVAL_235.json",
    "RETAILRAG_QWEN37_FINAL_FAQ_480.json",
    "RETAILRAG_QWEN37_FINAL_FAILURES.json",
    "RETAILRAG_QWEN37_FINAL_FAILURES.md",
    "RETAILRAG_QWEN37_FINAL_CATEGORY_METRICS.json",
    "RETAILRAG_QWEN37_FINAL_LATENCY.json",
    "RETAILRAG_QWEN37_FINAL_USAGE.json",
    "RETAILRAG_QWEN37_FINAL_VALIDATION.json",
)


def load(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def main():
    audit = load("RETAILRAG_SCALE_DATA_AUDIT.json")
    mismatches = []
    for relative, expected in audit["sha256"].items():
        path = ROOT / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if actual != expected:
            mismatches.append(relative)

    with (ROOT / "mysql_qa/data/retail_faq.csv").open(encoding="utf-8-sig", newline="") as handle:
        faq_count = len(list(csv.DictReader(handle)))
    doc_count = sum(1 for source in ("membership", "promotion", "after_sales",
                                     "store_operation", "product", "equipment",
                                     "policy", "notice")
                    for _ in (ROOT / "rag_qa/data" / f"{source}_data").glob("*.md"))
    faq_eval = load("RETAILRAG_FAQ_EVALUATION_SET.json")["samples"]
    retrieval_eval = load("RETAILRAG_SCALE_RETRIEVAL_SET.json")["samples"]
    rag_eval = load("RETAILRAG_RAG_EVALUATION_SET.json")["samples"]
    generation_eval = load("RETAILRAG_SCALE_GENERATION_SELECTION.json")["samples"]
    multiturn_count = Counter(row.get("question_type") for row in rag_eval)["multiturn"]
    counts = {
        "faq": faq_count,
        "documents": doc_count,
        "faq_evaluation": len(faq_eval),
        "retrieval_evaluation": len(retrieval_eval),
        "rag_evaluation": len(rag_eval),
        "multiturn_evaluation": multiturn_count,
        "generation_evaluation": len(generation_eval),
    }
    expected_counts = {
        "faq": 200, "documents": 80, "faq_evaluation": 480,
        "retrieval_evaluation": 235, "rag_evaluation": 350,
        "multiturn_evaluation": 30, "generation_evaluation": 80,
    }
    metrics = load("RETAILRAG_QWEN37_FINAL_METRICS.json")
    final_metrics_ok = (
        metrics["final_model"] == "qwen3.7-plus"
        and metrics["faq"]["f1"] == 0.863636
        and metrics["retrieval"]["recall_at_3"] == 0.901064
        and metrics["full_rag_350"]["strict_correct_rate"] == 0.791429
        and metrics["full_rag_350"]["execution_success"] == 350
    )
    missing_final = [name for name in REQUIRED_FINAL if not (ROOT / name).is_file()]
    checks = {
        "counts": counts == expected_counts,
        "benchmark_hash_84": len(audit["sha256"]) == 84 and not mismatches,
        "final_files": not missing_final,
        "final_metrics": final_metrics_ok,
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "counts": counts,
        "hash_mismatches": mismatches,
        "missing_final_files": missing_final,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
