"""Run real BGE-M3 hybrid retrieval, parent restore and reranking against Milvus."""

import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from base import Config
from rag_qa.core.vector_store import VectorStore


def main():
    config = Config()
    cases = json.loads(
        (ROOT / "RETAILRAG_RETRIEVAL_TEST_QUESTIONS.json").read_text(encoding="utf-8")
    )["questions"]
    store = VectorStore()
    rows = []
    started_all = time.perf_counter()
    for case in cases:
        started = time.perf_counter()
        parents = store.hybrid_search_with_rerank(case["query"], k=config.RETRIEVAL_K)
        trace = store.last_search_trace or {}
        returned_names = [doc.metadata.get("document_name") for doc in parents]
        relevant = any(name in case["expected_documents"] for name in returned_names)
        child_hits = trace.get("child_hits", [])
        parent_results = trace.get("parent_results", [])
        metadata_ok = bool(child_hits) and bool(parent_results) and all(
            item.get("source") and item.get("category") and item.get("document_name")
            for item in child_hits + parent_results
        )
        rows.append(
            {
                "id": case["id"],
                "category": case["category"],
                "query": case["query"],
                "strategy": "direct retriever audit (no Qwen selector)",
                "retrieval_query": case["query"],
                "expected_documents": case["expected_documents"],
                "returned_documents": returned_names,
                "expected_document_in_top_m": relevant,
                "metadata_ok": metadata_ok,
                "top_k_child_chunks": child_hits,
                "parent_results": parent_results,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )
        print(
            f"{case['id']:02d}/{len(cases)} category={case['category']} "
            f"relevant={relevant} docs={returned_names}"
        )

    visible = store.client.query(
        collection_name=store.collection_name,
        filter="",
        output_fields=["id"],
        limit=1000,
    )
    category_counts = Counter(row["category"] for row in rows)
    spot_checks = rows[:10]
    summary = {
        "status": "PASS" if (
            len(rows) >= 28
            and len(category_counts) == 8
            and all(category_counts[name] >= 3 for name in config.VALID_SOURCES)
            and all(row["metadata_ok"] for row in rows)
            and all(len(row["top_k_child_chunks"]) <= config.RETRIEVAL_K for row in rows)
            and all(len(row["parent_results"]) <= config.CANDIDATE_M for row in rows)
            and len(visible) == 78
        ) else "FAIL",
        "quality_status": "PASS" if all(row["expected_document_in_top_m"] for row in spot_checks) else "FAIL",
        "scope": "real BGE-M3 dense+sparse, Milvus hybrid, WeightedRanker, parent restore and BGE reranker",
        "collection": store.collection_name,
        "visible_entities": len(visible),
        "unique_entity_ids": len({row["id"] for row in visible}),
        "question_count": len(rows),
        "per_category": dict(category_counts),
        "metadata_complete": sum(row["metadata_ok"] for row in rows),
        "expected_document_top_m": sum(row["expected_document_in_top_m"] for row in rows),
        "spot_check_first_10_relevant": sum(row["expected_document_in_top_m"] for row in spot_checks),
        "parameters": {
            "parent_chunk_size": config.PARENT_CHUNK_SIZE,
            "child_chunk_size": config.CHILD_CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "top_k": config.RETRIEVAL_K,
            "top_m": config.CANDIDATE_M,
            "weighted_ranker": [1.0, 0.7]
        },
        "elapsed_seconds": round(time.perf_counter() - started_all, 3)
    }
    (ROOT / "RETAILRAG_RETRIEVAL_ONLINE_RESULTS.json").write_text(
        json.dumps({"summary": summary, "results": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (ROOT / "RETAILRAG_RETRIEVAL_ONLINE_RESULTS.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        fields = [
            "id", "category", "query", "strategy", "retrieval_query", "expected_documents",
            "returned_documents", "expected_document_in_top_m", "metadata_ok",
            "top_k_child_chunks", "parent_results", "elapsed_seconds"
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **row,
                "expected_documents": json.dumps(row["expected_documents"], ensure_ascii=False),
                "returned_documents": json.dumps(row["returned_documents"], ensure_ascii=False),
                "top_k_child_chunks": json.dumps(row["top_k_child_chunks"], ensure_ascii=False),
                "parent_results": json.dumps(row["parent_results"], ensure_ascii=False),
            })
    (ROOT / "RETAILRAG_RETRIEVAL_ONLINE_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["status"] != "PASS":
        raise RuntimeError("Online retrieval chain verification failed")


if __name__ == "__main__":
    main()
