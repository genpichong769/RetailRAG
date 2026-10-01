"""Evaluate the deterministic FAQ completeness gate on the frozen benchmarks."""

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from base import Config
from base.query_analysis import analyze_query
from rag_qa.core.vector_store import VectorStore


def load(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def save(name, value):
    path = ROOT / name
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def metrics(rows, prediction_key):
    tp = fp = fn = tn = 0
    rejected_positive = 0
    accepted_negative = 0
    exact = []
    paraphrase = []
    hard_negative = []
    for row in rows:
        prediction = row[prediction_key]
        wants_answer = row["expected_behavior"] == "answer"
        correct_answer = wants_answer and prediction == row["expected_faq_id"]
        accepted = prediction is not None
        if correct_answer:
            tp += 1
        if accepted and not correct_answer:
            fp += 1
        if wants_answer and not correct_answer:
            fn += 1
        if not wants_answer and not accepted:
            tn += 1
        rejected_positive += int(wants_answer and not accepted)
        accepted_negative += int(not wants_answer and accepted)
        if row["question_type"] == "exact":
            exact.append(int(correct_answer))
        if row["question_type"] == "paraphrase":
            paraphrase.append(int(correct_answer))
        if row["question_type"] == "hard_negative":
            hard_negative.append(int(prediction is None))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "sample_count": len(rows),
        "exact_accuracy": round(sum(exact) / max(len(exact), 1), 6),
        "paraphrase_recall": round(sum(paraphrase) / max(len(paraphrase), 1), 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(2 * precision * recall / max(precision + recall, 1e-12), 6),
        "false_accept_rate": round(
            accepted_negative / max(sum(not row["expected_behavior"] == "answer" for row in rows), 1),
            6,
        ),
        "false_reject_rate": round(rejected_positive / max(sum(row["expected_behavior"] == "answer" for row in rows), 1), 6),
        "hard_negative_accuracy": round(
            sum(hard_negative) / max(len(hard_negative), 1), 6
        ),
        "coverage": round(sum(row[prediction_key] is not None for row in rows) / len(rows), 6),
        "faq_to_rag_fallback_ratio": round(
            sum(row[prediction_key] is None for row in rows) / len(rows), 6
        ),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
    }


def main():
    cfg = Config()
    baseline = load("RETAILRAG_PHASE6_BASELINE_REFERENCE_RETAILRAG_SCALE_FAQ_RESULTS.json")
    rows = []
    latency = []
    for row in baseline["results"]:
        started = time.perf_counter()
        analysis = analyze_query(row["query"], row.get("top_question"))
        latency.append((time.perf_counter() - started) * 1000)
        before = row["predicted_faq_id"]
        after = None if before and analysis["faq_force_rag"] else before
        rows.append(
            {
                **row,
                "before_prediction": before,
                "after_prediction": after,
                "gate_blocked": bool(before and analysis["faq_force_rag"]),
                "gate_reasons": analysis["faq_gate_reasons"],
                "query_analysis": analysis,
                "before_correct": (
                    before == row["expected_faq_id"]
                    if row["expected_behavior"] == "answer"
                    else before is None
                ),
                "after_correct": (
                    after == row["expected_faq_id"]
                    if row["expected_behavior"] == "answer"
                    else after is None
                ),
            }
        )
    before_metrics = metrics(rows, "before_prediction")
    after_metrics = metrics(rows, "after_prediction")
    for key, prediction_key in (("before", "before_prediction"), ("after", "after_prediction")):
        accepted_wrong_complex = [
            row
            for row in rows
            if row[prediction_key] is not None
            and not row[f"{key}_correct"]
            and (
                row["query_analysis"]["context_complex"]
                or bool(row["query_analysis"]["faq_gate_reasons"])
            )
        ]
        (before_metrics if key == "before" else after_metrics)[
            "complex_query_wrong_intercept_count"
        ] = len(accepted_wrong_complex)

    with (ROOT / "mysql_qa/data/retail_faq.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        faq = list(csv.DictReader(handle))
    rag_samples = load("RETAILRAG_RAG_EVALUATION_SET.json")["samples"]
    store = VectorStore()
    faq_vectors = np.asarray(
        store.embedding_function([row["问题"] for row in faq])["dense"], dtype=np.float32
    )
    rag_rows = []
    for start in range(0, len(rag_samples), 32):
        block = rag_samples[start : start + 32]
        query_vectors = np.asarray(
            store.embedding_function([row["question"] for row in block])["dense"],
            dtype=np.float32,
        )
        similarity = query_vectors @ faq_vectors.T
        for sample, scores in zip(block, similarity):
            index = int(scores.argmax())
            score = float(scores[index])
            candidate = faq[index]["问题"]
            analysis = analyze_query(sample["question"], candidate)
            before_hit = score >= cfg.FAQ_MATCH_THRESHOLD
            after_hit = before_hit and not analysis["faq_force_rag"]
            rag_rows.append(
                {
                    "id": sample["id"],
                    "question": sample["question"],
                    "question_type": sample["question_type"],
                    "expected_behavior": sample["expected_behavior"],
                    "faq_candidate": candidate,
                    "score": score,
                    "before_faq_hit": before_hit,
                    "after_faq_hit": after_hit,
                    "complex": analysis["context_complex"],
                    "gate_reasons": analysis["faq_gate_reasons"],
                }
            )
        print(f"rag FAQ routing {min(start + 32, len(rag_samples))}/{len(rag_samples)}", flush=True)

    rag_summary = {
        "sample_count": len(rag_rows),
        "before_faq_hit_count": sum(row["before_faq_hit"] for row in rag_rows),
        "after_faq_hit_count": sum(row["after_faq_hit"] for row in rag_rows),
        "before_complex_faq_route_count": sum(
            row["before_faq_hit"] and row["complex"] for row in rag_rows
        ),
        "after_complex_faq_route_count": sum(
            row["after_faq_hit"] and row["complex"] for row in rag_rows
        ),
        "fallback_added_count": sum(
            row["before_faq_hit"] and not row["after_faq_hit"] for row in rag_rows
        ),
        "interpretation": "RAG benchmark rows do not carry expected_faq_id. These counts describe routing exposure only and are not labelled as wrong intercepts; wrong-intercept accuracy is measured on FAQ-480.",
    }
    output = {
        "status": "COMPLETED",
        "threshold": cfg.FAQ_MATCH_THRESHOLD,
        "faq_480": {
            "before": before_metrics,
            "after": after_metrics,
            "average_gate_latency_ms": round(sum(latency) / len(latency), 6),
            "p95_gate_latency_ms": round(sorted(latency)[int(len(latency) * 0.95) - 1], 6),
            "results": rows,
        },
        "rag_350_routing": {"metrics": rag_summary, "results": rag_rows},
    }
    save("RETAILRAG_PHASE6_FAQ_GATE_EVALUATION.json", output)
    print(json.dumps({"faq_480": output["faq_480"], "rag_350": rag_summary}, ensure_ascii=False, indent=2, default=str)[:12000])


if __name__ == "__main__":
    main()
