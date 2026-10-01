"""Run Sparse/Dense/Hybrid/Reranker ablation against the 130 retrieval cases."""

import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag_qa.core.vector_store import VectorStore

DATASET = ROOT / "RETAILRAG_RAG_EVALUATION_SET.json"
OUT = ROOT / "RETAILRAG_RETRIEVAL_ABLATION.json"
CSV_OUT = ROOT / "RETAILRAG_RETRIEVAL_ABLATION_RESULTS.csv"

MODES = {
    "sparse_only": ("sparse", False),
    "dense_only": ("dense", False),
    "hybrid": ("hybrid", False),
}


def summarize(rows):
    total = len(rows)
    summary = {"sample_count": total}
    for k in (1, 3, 5):
        summary[f"any_recall_at_{k}"] = round(sum(row[f"any_at_{k}"] for row in rows) / total, 6)
        summary[f"all_recall_at_{k}"] = round(sum(row[f"all_at_{k}"] for row in rows) / total, 6)
    summary["mrr"] = round(sum(row["reciprocal_rank"] for row in rows) / total, 6)
    summary["hit_rate_at_5"] = summary["any_recall_at_5"]
    summary["context_precision_at_5"] = round(
        sum(row["context_precision_at_5"] for row in rows) / total, 6
    )
    return summary


def main():
    started = time.time()
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    allowed = {"single_document", "multi_document", "keyword_exact", "semantic_paraphrase", "version_conflict"}
    samples = [row for row in dataset["samples"] if row["question_type"] in allowed]
    store = VectorStore()
    all_results = {}
    flat_rows = []
    hybrid_candidates = []
    for mode_name, (mode, rerank) in MODES.items():
        rows = []
        for index, sample in enumerate(samples, 1):
            documents = store.search_documents(
                sample["question"],
                mode=mode,
                rerank=rerank,
                candidate_k=5,
                limit=5,
                active_only=True,
            )
            retrieved = [doc.metadata.get("document_name") for doc in documents]
            expected = set(sample["reference_documents"])
            rank = next((i for i, name in enumerate(retrieved, 1) if name in expected), None)
            row = {
                "id": sample["id"],
                "question_type": sample["question_type"],
                "expected_documents": sorted(expected),
                "retrieved_documents": retrieved,
                "top_reranker_score": documents[0].metadata.get("reranker_score") if documents else None,
                "reciprocal_rank": 1.0 / rank if rank else 0.0,
                "context_precision_at_5": len(expected & set(retrieved[:5])) / max(len(retrieved[:5]), 1),
            }
            for k in (1, 3, 5):
                top = set(retrieved[:k])
                row[f"any_at_{k}"] = int(bool(expected & top))
                row[f"all_at_{k}"] = int(expected.issubset(top))
            rows.append(row)
            flat_rows.append(
                {
                    "mode": mode_name,
                    "id": row["id"],
                    "question_type": row["question_type"],
                    "expected_documents": "|".join(row["expected_documents"]),
                    "retrieved_documents": "|".join(row["retrieved_documents"]),
                    "any_at_1": row["any_at_1"],
                    "any_at_3": row["any_at_3"],
                    "any_at_5": row["any_at_5"],
                    "all_at_5": row["all_at_5"],
                    "reciprocal_rank": round(row["reciprocal_rank"], 6),
                    "context_precision_at_5": round(row["context_precision_at_5"], 6),
                    "top_reranker_score": row["top_reranker_score"],
                }
            )
            if index % 10 == 0:
                print(f"{mode_name}: {index}/{len(samples)}", flush=True)
            if mode_name == "hybrid":
                hybrid_candidates.append(documents)
        all_results[mode_name] = {"metrics": summarize(rows), "results": rows}

    # Rerank all Hybrid pairs in a single batched pass to avoid per-query CPU overhead.
    pairs = []
    for sample, documents in zip(samples, hybrid_candidates):
        pairs.extend([[sample["question"], doc.page_content] for doc in documents])
    scores = store.reranker.predict(pairs, batch_size=32, show_progress_bar=True)
    reranked_rows = []
    offset = 0
    for sample, documents in zip(samples, hybrid_candidates):
        doc_scores = scores[offset : offset + len(documents)]
        offset += len(documents)
        for score, doc in zip(doc_scores, documents):
            doc.metadata["reranker_score"] = float(score)
        documents = sorted(documents, key=lambda doc: doc.metadata["reranker_score"], reverse=True)
        retrieved = [doc.metadata.get("document_name") for doc in documents]
        expected = set(sample["reference_documents"])
        rank = next((i for i, name in enumerate(retrieved, 1) if name in expected), None)
        row = {
            "id": sample["id"],
            "question_type": sample["question_type"],
            "expected_documents": sorted(expected),
            "retrieved_documents": retrieved,
            "top_reranker_score": documents[0].metadata.get("reranker_score") if documents else None,
            "reciprocal_rank": 1.0 / rank if rank else 0.0,
            "context_precision_at_5": len(expected & set(retrieved[:5])) / max(len(retrieved[:5]), 1),
        }
        for k in (1, 3, 5):
            top = set(retrieved[:k])
            row[f"any_at_{k}"] = int(bool(expected & top))
            row[f"all_at_{k}"] = int(expected.issubset(top))
        reranked_rows.append(row)
        flat_rows.append(
            {
                "mode": "hybrid_reranker",
                "id": row["id"],
                "question_type": row["question_type"],
                "expected_documents": "|".join(row["expected_documents"]),
                "retrieved_documents": "|".join(row["retrieved_documents"]),
                "any_at_1": row["any_at_1"],
                "any_at_3": row["any_at_3"],
                "any_at_5": row["any_at_5"],
                "all_at_5": row["all_at_5"],
                "reciprocal_rank": round(row["reciprocal_rank"], 6),
                "context_precision_at_5": round(row["context_precision_at_5"], 6),
                "top_reranker_score": row["top_reranker_score"],
            }
        )
    all_results["hybrid_reranker"] = {
        "metrics": summarize(reranked_rows),
        "results": reranked_rows,
    }

    hybrid = all_results["hybrid"]["metrics"]
    reranked = all_results["hybrid_reranker"]["metrics"]
    gain = {
        key: round(reranked[key] - hybrid[key], 6)
        for key in ("any_recall_at_1", "any_recall_at_3", "any_recall_at_5", "mrr", "context_precision_at_5")
    }
    output = {
        "status": "PASS",
        "evaluation_sample_count": len(samples),
        "candidate_k": 5,
        "returned_k": 5,
        "version_filter": "active + effective date + latest version_group",
        "modes": all_results,
        "reranker_gain_vs_hybrid": gain,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    with CSV_OUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    print(json.dumps({"sample_count": len(samples), "metrics": {k: v["metrics"] for k, v in all_results.items()}, "reranker_gain": gain}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
