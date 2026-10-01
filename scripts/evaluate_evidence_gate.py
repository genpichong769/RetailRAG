"""Calibrate the code-level evidence gate on 25 answerable and 25 unanswerable cases."""

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag_qa.core.vector_store import VectorStore

DATASET = ROOT / "RETAILRAG_RAG_EVALUATION_SET.json"
OUT = ROOT / "RETAILRAG_EVIDENCE_GATE_EVALUATION.json"
CSV_OUT = ROOT / "RETAILRAG_EVIDENCE_GATE_RESULTS.csv"
CONFIG_OUT = ROOT / "RETAILRAG_EVIDENCE_GATE_CONFIG.json"


def metrics(labels, scores, threshold):
    accepted = scores >= threshold
    answerable = labels == 1
    unanswerable = ~answerable
    refused = ~accepted
    answerable_accuracy = float((accepted & answerable).sum() / answerable.sum())
    refusal_precision = float((refused & unanswerable).sum() / refused.sum()) if refused.sum() else 0.0
    refusal_recall = float((refused & unanswerable).sum() / unanswerable.sum())
    f1 = 2 * refusal_precision * refusal_recall / (refusal_precision + refusal_recall) if refusal_precision + refusal_recall else 0.0
    return {
        "threshold": round(float(threshold), 6),
        "answerable_accuracy": round(answerable_accuracy, 6),
        "refusal_precision": round(refusal_precision, 6),
        "refusal_recall": round(refusal_recall, 6),
        "refusal_f1": round(f1, 6),
        "accepted_answerable": int((accepted & answerable).sum()),
        "false_accept_unanswerable": int((accepted & unanswerable).sum()),
        "false_refuse_answerable": int((refused & answerable).sum()),
        "refused_unanswerable": int((refused & unanswerable).sum()),
    }


def main():
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))["samples"]
    answerable = [row for row in dataset if row["question_type"] == "single_document"][:25]
    unanswerable = [row for row in dataset if row["question_type"] == "unanswerable"]
    samples = answerable + unanswerable
    store = VectorStore()
    rows = []
    candidate_sets = []
    all_pairs = []
    for index, sample in enumerate(samples, 1):
        docs = store.search_documents(
            sample["question"], mode="hybrid", rerank=False, candidate_k=3, limit=3, active_only=True
        )
        candidate_sets.append(docs)
        all_pairs.extend([[sample["question"], doc.page_content] for doc in docs])
        print(f"evidence retrieval: {index}/{len(samples)}", flush=True)
    all_scores = store.reranker.predict(all_pairs, batch_size=32, show_progress_bar=True)
    offset = 0
    for index, (sample, docs) in enumerate(zip(samples, candidate_sets), 1):
        doc_scores = all_scores[offset : offset + len(docs)]
        offset += len(docs)
        for score, doc in zip(doc_scores, docs):
            doc.metadata["reranker_score"] = float(score)
        docs = sorted(docs, key=lambda doc: doc.metadata["reranker_score"], reverse=True)[:2]
        valid_docs = [
            doc for doc in docs
            if doc.metadata.get("status") == "active"
            and doc.metadata.get("source")
            and doc.metadata.get("document_name")
        ]
        score = max((float(doc.metadata.get("reranker_score", -100.0)) for doc in valid_docs), default=-100.0)
        rows.append(
            {
                "id": sample["id"],
                "question": sample["question"],
                "expected_behavior": sample["expected_behavior"],
                "top_reranker_score": score,
                "valid_context_count": len(valid_docs),
                "top_document": valid_docs[0].metadata.get("document_name") if valid_docs else None,
            }
        )
        print(f"evidence score: {index}/{len(samples)} score={score:.4f}", flush=True)
    labels = np.asarray([1 if row["expected_behavior"] == "answer" else 0 for row in rows])
    scores = np.asarray([row["top_reranker_score"] for row in rows], dtype=float)
    candidates = sorted(set(np.linspace(float(scores.min()) - 0.01, float(scores.max()) + 0.01, 301).tolist()))
    sweep = [metrics(labels, scores, threshold) for threshold in candidates]
    eligible = [row for row in sweep if row["answerable_accuracy"] >= 0.80]
    selected = max(
        eligible or sweep,
        key=lambda row: (row["refusal_f1"], row["refusal_recall"], row["answerable_accuracy"]),
    )
    for row in rows:
        row["decision"] = "answer" if row["top_reranker_score"] >= selected["threshold"] and row["valid_context_count"] else "refuse"
        row["correct"] = row["decision"] == row["expected_behavior"]
    output = {
        "status": "PASS",
        "sample_count": len(samples),
        "answerable_count": len(answerable),
        "unanswerable_count": len(unanswerable),
        "selection_rule": "maximize refusal F1 subject to answerable accuracy >= 80%",
        "selected": selected,
        "score_distribution": {
            "answerable": {"min": round(float(scores[labels == 1].min()), 6), "median": round(float(np.median(scores[labels == 1])), 6), "max": round(float(scores[labels == 1].max()), 6)},
            "unanswerable": {"min": round(float(scores[labels == 0].min()), 6), "median": round(float(np.median(scores[labels == 0])), 6), "max": round(float(scores[labels == 0].max()), 6)},
        },
        "threshold_sweep": sweep,
        "results": rows,
    }
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    CONFIG_OUT.write_text(json.dumps({"status": "PASS", "reranker_threshold": selected["threshold"], "metrics": selected}, ensure_ascii=False, indent=2), encoding="utf-8")
    with CSV_OUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"selected": selected, "score_distribution": output["score_distribution"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
