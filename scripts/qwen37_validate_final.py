"""Validate the final qwen3.7-plus deliverables without issuing model calls."""

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from base import Config


REQUIRED = [
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
]


def load(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def main():
    checks = {}
    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    checks["required_files_present"] = not missing

    faq = load("RETAILRAG_QWEN37_FINAL_FAQ_480.json")
    retrieval = load("RETAILRAG_QWEN37_FINAL_RETRIEVAL_235.json")
    generation = load("RETAILRAG_QWEN37_FINAL_GENERATION_80.json")
    full = load("RETAILRAG_QWEN37_FINAL_350_RESULTS.json")
    failures = load("RETAILRAG_QWEN37_FINAL_FAILURES.json")
    categories = load("RETAILRAG_QWEN37_FINAL_CATEGORY_METRICS.json")
    latency = load("RETAILRAG_QWEN37_FINAL_LATENCY.json")
    usage = load("RETAILRAG_QWEN37_FINAL_USAGE.json")
    metrics = load("RETAILRAG_QWEN37_FINAL_METRICS.json")
    engineering = load("RETAILRAG_QWEN37_FINAL_ENGINEERING.json")
    freeze = load("RETAILRAG_QWEN37_BENCHMARK_FREEZE.json")
    audit = load("RETAILRAG_SCALE_DATA_AUDIT.json")

    checks["config_model"] = Config().LLM_MODEL == "qwen3.7-plus"
    checks["faq_480"] = faq.get("status") == "COMPLETED" and len(faq.get("results", [])) == 480
    checks["retrieval_235"] = retrieval.get("status") == "COMPLETED" and len(retrieval["modes"]["hybrid_reranker"]["results"]) == 235
    checks["generation_80"] = generation.get("status") == "COMPLETED" and len(generation.get("results", [])) == 80 and generation.get("execution_errors") == 0
    checks["full_rag_350"] = full.get("status") == "COMPLETED" and len(full.get("results", [])) == 350 and full.get("execution_errors") == 0 and full.get("correctness_judged") == 350

    final_models = {
        generation.get("generator_model"),
        generation.get("judge_model"),
        full.get("final_llm_model"),
        full.get("correctness_judge_model"),
        failures.get("model"),
        categories.get("model"),
        latency.get("model"),
        usage.get("model"),
        metrics.get("final_model"),
        engineering.get("model"),
    }
    checks["uniform_final_model"] = final_models == {"qwen3.7-plus"}

    allowed = {"CORRECT", "PARTIALLY_CORRECT", "INCORRECT", "CORRECT_REFUSAL", "WRONG_REFUSAL", "EXECUTION_ERROR"}
    classifications = [row.get("classification") for row in full["results"]]
    checks["all_rows_classified"] = len(classifications) == 350 and set(classifications) <= allowed and None not in classifications
    m = full["metrics"]
    checks["classification_totals"] = sum(m[key] for key in ("correct", "partial", "incorrect", "correct_refusal", "wrong_refusal", "execution_errors")) == 350
    expected_failures = m["partial"] + m["incorrect"] + m["wrong_refusal"] + m["execution_errors"]
    checks["failure_audit_complete"] = failures.get("failure_count") == expected_failures == len(failures.get("failures", [])) and sum(failures.get("failure_stage_counts", {}).values()) == expected_failures
    checks["failure_stages_valid"] = all(row.get("failure_stage") in set(failures["allowed_stages"]) for row in failures["failures"])
    checks["category_metrics_present"] = bool(categories.get("generation_80")) and bool(categories.get("full_rag_350"))
    checks["latency_complete"] = latency.get("status") == "COMPLETED" and bool(latency.get("full_rag_350"))
    checks["usage_complete"] = usage.get("status") == "COMPLETED" and usage.get("full_rag_350_production_calls") is not None
    checks["engineering_pass"] = engineering.get("status") == "PASS" and all(engineering.get(key) == "PASS" for key in ("mysql", "redis", "milvus", "minio", "etcd", "fastapi", "websocket", "browser_e2e"))
    checks["freeze_record_pass"] = freeze.get("status") == "PASS" and not freeze.get("mismatches") and freeze.get("verified_file_count") == 84

    current_mismatches = []
    for relative, expected in audit["sha256"].items():
        path = ROOT / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if actual != expected:
            current_mismatches.append({"path": relative, "expected": expected, "actual": actual})
    checks["current_benchmark_hashes_match"] = not current_mismatches

    report = (ROOT / "RETAILRAG_QWEN37_FINAL_EVALUATION_REPORT.md").read_text(encoding="utf-8")
    checks["report_has_required_sections"] = all(text in report for text in ("FINAL RESULT", "HISTORICAL_MODEL_EXPERIMENTS", "Final assessment questions", "ENGINEERING"))
    checks["no_mixed_model_metrics"] = metrics.get("historical_model_experiments", {}).get("mixed_model_results") is False

    output = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "validated_at": datetime.now().astimezone().isoformat(),
        "checks": checks,
        "missing_files": missing,
        "benchmark_hash_mismatches": current_mismatches,
        "final_model_values": sorted(value for value in final_models if value),
        "counts": {
            "faq": len(faq["results"]),
            "retrieval": len(retrieval["modes"]["hybrid_reranker"]["results"]),
            "generation": len(generation["results"]),
            "full_rag": len(full["results"]),
            "failures": len(failures["failures"]),
        },
    }
    (ROOT / "RETAILRAG_QWEN37_FINAL_VALIDATION.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if output["status"] != "PASS":
        raise RuntimeError("Final deliverable validation failed")


if __name__ == "__main__":
    main()
