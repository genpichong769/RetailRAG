"""Validate generic metadata version filtering with the conflicting gold-points docs."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag_qa.core.vector_store import VectorStore

OUT = ROOT / "RETAILRAG_VERSION_CONFLICT_RESULTS.json"


def describe(doc):
    return {
        "document_name": doc.metadata.get("document_name"),
        "version": doc.metadata.get("version"),
        "effective_date": doc.metadata.get("effective_date"),
        "expire_date": doc.metadata.get("expire_date"),
        "status": doc.metadata.get("status"),
        "version_group": doc.metadata.get("version_group"),
        "content_contains_current_rate": "1.5" in doc.page_content,
        "content_contains_old_rate": "1 元累计 1 积分" in doc.page_content,
    }


def main():
    store = VectorStore()
    query = "现在金卡积分怎么算？"
    active = store.search_documents(
        query, mode="hybrid", rerank=False, candidate_k=10, limit=10, active_only=True
    )
    all_versions = store.search_documents(
        query, mode="hybrid", rerank=False, candidate_k=10, limit=10, active_only=False
    )
    active_rows = [describe(doc) for doc in active]
    all_rows = [describe(doc) for doc in all_versions]
    relevant_active = [row for row in active_rows if row["version_group"] == "member_gold_points_rate"]
    relevant_all = [row for row in all_rows if row["version_group"] == "member_gold_points_rate"]
    passed = (
        bool(relevant_active)
        and all(row["status"] == "active" and row["version"] == "2.0" for row in relevant_active)
        and any(row["version"] == "1.0" and row["status"] == "superseded" for row in relevant_all)
        and any(row["version"] == "2.0" and row["status"] == "active" for row in relevant_all)
    )
    output = {
        "status": "PASS" if passed else "FAIL",
        "query": query,
        "active_only_results": active_rows,
        "unfiltered_results": all_rows,
        "assertions": {
            "active_path_returns_v2": bool(relevant_active) and all(row["version"] == "2.0" for row in relevant_active),
            "active_path_excludes_superseded": all(row["status"] != "superseded" for row in active_rows),
            "audit_path_can_see_v1_and_v2": {row["version"] for row in relevant_all} >= {"1.0", "2.0"},
        },
    }
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if not passed:
        raise RuntimeError("version conflict validation failed")


if __name__ == "__main__":
    main()
