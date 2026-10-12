"""Lightweight, offline validation for the public RetailRAG release.

This checks retained assets and frozen summary values. It intentionally does
not recompute paid-model experiments or validate the private raw-result archive.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = ROOT / "docs" / "evaluation"
METRICS_PATH = EVALUATION_DIR / "RETAILRAG_QWEN37_FINAL_METRICS.json"
REPORT_PATH = EVALUATION_DIR / "RETAILRAG_QWEN37_FINAL_EVALUATION_REPORT.md"

EXPECTED_SHA256 = {
    METRICS_PATH: "f07a0ec4e5ede322d408a78f74a98e8e7554ba44b040b132d2af09919c8e4639",
    REPORT_PATH: "4455117da95d5d3cdd2c57e313e9584bbc6eabcacd8f1fb3e350d69fbd4a0c5c",
}

REQUIRED_FILES = (
    ROOT / "README.md",
    ROOT / "LICENSE",
    ROOT / ".gitignore",
    ROOT / ".gitattributes",
    ROOT / ".env.example",
    ROOT / "config.example.ini",
    ROOT / "requirements.txt",
    ROOT / "requirements-windows.txt",
    ROOT / "requirements-mac.txt",
    ROOT / "docker-compose.yml",
    ROOT / "app.py",
    ROOT / "new_main.py",
    ROOT / "static" / "index.html",
    ROOT / "RETAILRAG_PHASE6_EVIDENCE_GATE_CONFIG.json",
    METRICS_PATH,
    REPORT_PATH,
)

CLASSIFIER_FILES = (
    "config.json",
    "label_map.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "training_metrics.json",
    "vocab.txt",
)

KNOWLEDGE_SOURCES = (
    "membership",
    "promotion",
    "after_sales",
    "store_operation",
    "product",
    "equipment",
    "policy",
    "notice",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def main() -> None:
    missing_required = [
        str(path.relative_to(ROOT)) for path in REQUIRED_FILES if not path.is_file()
    ]

    faq_path = ROOT / "mysql_qa" / "data" / "retail_faq.csv"
    with faq_path.open(encoding="utf-8-sig", newline="") as handle:
        faq_count = len(list(csv.DictReader(handle)))

    document_count = sum(
        1
        for source in KNOWLEDGE_SOURCES
        for _ in (ROOT / "rag_qa" / "data" / f"{source}_data").glob("*.md")
    )

    classifier_dir = ROOT / "rag_qa" / "models" / "retail-query-classifier"
    missing_classifier = [
        name for name in CLASSIFIER_FILES if not (classifier_dir / name).is_file()
    ]
    classifier_weight = classifier_dir / "model.safetensors"
    classifier_weight_materialized = (
        classifier_weight.is_file() and classifier_weight.stat().st_size > 100_000_000
    )

    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    lfs_rule_present = (
        "rag_qa/models/retail-query-classifier/model.safetensors "
        "filter=lfs diff=lfs merge=lfs -text"
    ) in attributes

    hashes = {
        str(path.relative_to(ROOT)): sha256(path) == expected
        for path, expected in EXPECTED_SHA256.items()
        if path.is_file()
    }
    metrics = load_metrics()
    expected_metrics = (
        metrics["final_model"] == "qwen3.7-plus"
        and metrics["data_scale"]
        == {
            "faq": 200,
            "documents": 80,
            "faq_evaluation": 480,
            "retrieval_evaluation": 235,
            "rag_evaluation": 350,
            "generation_evaluation": 80,
            "multiturn_evaluation": 30,
        }
        and metrics["faq"]["f1"] == 0.863636
        and metrics["retrieval"]["recall_at_3"] == 0.901064
        and metrics["retrieval"]["mrr"] == 0.877447
        and metrics["generation_80"]["execution_count"] == 80
        and metrics["generation_80"]["metrics"]["faithfulness"] == 0.98125
        and metrics["full_rag_350"]["execution_success"] == 350
        and metrics["full_rag_350"]["strict_correct_rate"] == 0.791429
    )

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    repository_links_current = (
        "github.com/genpichong769/RetailRAG" in readme
        and "EduRAG-Intelligent-QA-System" not in readme
    )

    checks = {
        "required_public_files": not missing_required,
        "faq_count_200": faq_count == 200,
        "knowledge_document_count_80": document_count == 80,
        "classifier_assets": not missing_classifier,
        "classifier_lfs_weight_materialized": classifier_weight_materialized,
        "classifier_lfs_rule": lfs_rule_present,
        "frozen_summary_hashes": len(hashes) == len(EXPECTED_SHA256)
        and all(hashes.values()),
        "frozen_summary_metrics": expected_metrics,
        "repository_links": repository_links_current,
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "PUBLIC_RELEASE_LIGHTWEIGHT",
        "checks": checks,
        "counts": {"faq": faq_count, "knowledge_documents": document_count},
        "missing_required_files": missing_required,
        "missing_classifier_files": missing_classifier,
        "summary_hash_checks": hashes,
        "limitations": [
            "Does not call Qwen or rerun paid-model evaluations.",
            "Does not recompute the 480/235/80/350 per-case benchmark outputs.",
            "Does not validate the local-only 84-file frozen archive baseline.",
        ],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
