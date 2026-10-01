"""Evaluate BM25 and BGE-M3 FAQ retrieval candidates on the PHASE 5 set."""

import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from milvus_model.hybrid import BGEM3EmbeddingFunction
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from base import Config
from mysql_qa.utils.preprocess import preprocess_text

EVAL_PATH = ROOT / "RETAILRAG_FAQ_EVALUATION_SET.json"
FAQ_PATH = ROOT / "mysql_qa" / "data" / "retail_faq.csv"
BASELINE_PATH = ROOT / "RETAILRAG_FAQ_BASELINE_EVALUATION.json"
COMPARISON_PATH = ROOT / "RETAILRAG_FAQ_METHOD_COMPARISON.json"
SWEEP_PATH = ROOT / "RETAILRAG_FAQ_THRESHOLD_SWEEP.csv"
RESULTS_PATH = ROOT / "RETAILRAG_FAQ_OPTIMIZED_RESULTS.csv"
CONFIG_PATH = ROOT / "RETAILRAG_FAQ_FINAL_CONFIG.json"


def _softmax(values):
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _metrics(samples, candidate_indices, decision_scores, faq_ids, threshold):
    accepted = decision_scores >= threshold
    predictions = [faq_ids[index] if accepted[row] else None for row, index in enumerate(candidate_indices)]
    positive = np.array([sample["expected_behavior"] == "answer" for sample in samples])
    exact_mask = np.array([sample["question_type"] == "exact" for sample in samples])
    para_mask = np.array([sample["question_type"] == "paraphrase" for sample in samples])
    negative = ~positive
    correct = np.array(
        [prediction == sample.get("expected_faq_id") for prediction, sample in zip(predictions, samples)]
    ) & positive
    wrong_accept = accepted & ~correct
    tp = int(correct.sum())
    fp = int(wrong_accept.sum())
    fn = int((positive & ~correct).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / positive.sum() if positive.sum() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": round(float(threshold), 6),
        "exact_accuracy": round(float(correct[exact_mask].mean()), 6),
        "paraphrase_recall": round(float(correct[para_mask].mean()), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "false_accept_rate": round(float(accepted[negative].mean()), 6),
        "false_reject_rate": round(float((~accepted & positive).sum() / positive.sum()), 6),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "accepted": int(accepted.sum()),
        "predictions": predictions,
        "accepted_mask": accepted.tolist(),
    }


def _select_best(rows):
    eligible = [
        row
        for row in rows
        if row["false_accept_rate"] <= 0.02 and row["exact_accuracy"] >= 0.98
    ]
    candidates = eligible or rows
    return max(
        candidates,
        key=lambda row: (
            row["f1"],
            row["paraphrase_recall"],
            -row["false_accept_rate"],
            row["precision"],
        ),
    )


def _public_metrics(row):
    return {key: value for key, value in row.items() if key not in {"predictions", "accepted_mask"}}


def main():
    started = time.time()
    config = Config()
    evaluation = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    samples = evaluation["samples"]
    with FAQ_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        faq_rows = list(csv.DictReader(handle))
    faq_ids = [f"faq_{index:03d}" for index in range(1, len(faq_rows) + 1)]
    questions = [row["问题"] for row in faq_rows]
    queries = [sample["query"] for sample in samples]

    bm25 = BM25Okapi([preprocess_text(question) for question in questions])
    raw_bm25 = np.asarray(
        [bm25.get_scores(preprocess_text(query)) for query in queries], dtype=np.float32
    )
    bm25_softmax = _softmax(raw_bm25)
    bm25_scaled = 1.0 - np.exp(-np.maximum(raw_bm25, 0.0) / 2.0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding = BGEM3EmbeddingFunction(
        model_name_or_path=config.bge_m3,
        use_fp16=device == "cuda",
        device=device,
    )
    faq_dense = np.asarray(embedding(questions)["dense"], dtype=np.float32)
    query_dense_parts = []
    for start in range(0, len(queries), 32):
        query_dense_parts.append(
            np.asarray(embedding(queries[start : start + 32])["dense"], dtype=np.float32)
        )
    query_dense = np.vstack(query_dense_parts)
    dense_scores = np.matmul(query_dense, faq_dense.T)
    dense_scaled = np.clip(dense_scores, 0.0, 1.0)

    fixed_thresholds = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    search_thresholds = sorted(set(fixed_thresholds + [round(value, 2) for value in np.arange(0.35, 0.96, 0.01)]))
    methods = {}

    bm25_indices = np.argmax(raw_bm25, axis=1)
    methods["bm25_only"] = (bm25_indices, bm25_softmax[np.arange(len(samples)), bm25_indices], None)
    dense_indices = np.argmax(dense_scores, axis=1)
    methods["dense_only"] = (dense_indices, dense_scores[np.arange(len(samples)), dense_indices], None)
    methods["bm25_dense_validation"] = (
        bm25_indices,
        dense_scores[np.arange(len(samples)), bm25_indices],
        None,
    )
    for lexical_weight in (0.20, 0.35, 0.50, 0.65, 0.80):
        hybrid_scores = lexical_weight * bm25_scaled + (1.0 - lexical_weight) * dense_scaled
        indices = np.argmax(hybrid_scores, axis=1)
        name = f"hybrid_w{lexical_weight:.2f}"
        methods[name] = (
            indices,
            hybrid_scores[np.arange(len(samples)), indices],
            lexical_weight,
        )

    sweep_rows = []
    best_by_method = {}
    internal_rows = {}
    for method, (indices, scores, weight) in methods.items():
        rows = []
        for threshold in search_thresholds:
            row = _metrics(samples, indices, scores, faq_ids, threshold)
            row["method"] = method
            row["lexical_weight"] = weight
            rows.append(row)
            sweep_rows.append(_public_metrics(row))
        internal_rows[method] = rows
        best_by_method[method] = _select_best(rows)

    baseline = _metrics(
        samples,
        bm25_indices,
        bm25_softmax[np.arange(len(samples)), bm25_indices],
        faq_ids,
        0.85,
    )
    baseline_failures = []
    for index, (sample, prediction, accepted) in enumerate(
        zip(samples, baseline["predictions"], baseline["accepted_mask"])
    ):
        expected = sample.get("expected_faq_id")
        if (sample["expected_behavior"] == "answer" and prediction != expected) or (
            sample["expected_behavior"] == "reject" and accepted
        ):
            top_index = int(bm25_indices[index])
            baseline_failures.append(
                {
                    "id": sample["id"],
                    "query": sample["query"],
                    "question_type": sample["question_type"],
                    "expected_faq_id": expected,
                    "predicted_faq_id": prediction,
                    "top1_question": questions[top_index],
                    "raw_bm25_score": round(float(raw_bm25[index, top_index]), 6),
                    "decision_score": round(float(bm25_softmax[index, top_index]), 6),
                }
            )
    baseline_output = {
        "status": "PASS",
        "method": "Jieba + BM25 + Softmax",
        "sample_count": len(samples),
        "metrics": _public_metrics(baseline),
        "failure_count": len(baseline_failures),
        "failures": baseline_failures,
    }
    BASELINE_PATH.write_text(
        json.dumps(baseline_output, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    candidate_best = []
    for method, row in best_by_method.items():
        public = _public_metrics(row)
        public["method"] = method
        public["lexical_weight"] = methods[method][2]
        candidate_best.append(public)
    final = max(
        best_by_method.values(),
        key=lambda row: (
            row["false_accept_rate"] <= 0.02,
            row["f1"],
            row["paraphrase_recall"],
            -row["false_accept_rate"],
        ),
    )
    final_method = final["method"]
    final_indices, final_scores, final_weight = methods[final_method]
    comparison = {
        "status": "PASS",
        "device": device,
        "sample_count": len(samples),
        "baseline": _public_metrics(baseline),
        "candidate_best": sorted(candidate_best, key=lambda row: row["f1"], reverse=True),
        "selected": {
            **_public_metrics(final),
            "method": final_method,
            "lexical_weight": final_weight,
            "selection_rule": "maximize F1 with FAR<=2% and exact accuracy>=98% when feasible",
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    COMPARISON_PATH.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with SWEEP_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = list(sweep_rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sweep_rows)

    with RESULTS_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = [
            "id",
            "query",
            "question_type",
            "expected_behavior",
            "expected_faq_id",
            "predicted_faq_id",
            "top1_question",
            "decision_score",
            "accepted",
            "correct",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, sample in enumerate(samples):
            accepted = bool(final_scores[index] >= final["threshold"])
            predicted = faq_ids[int(final_indices[index])] if accepted else None
            correct = (
                predicted == sample.get("expected_faq_id")
                if sample["expected_behavior"] == "answer"
                else not accepted
            )
            writer.writerow(
                {
                    "id": sample["id"],
                    "query": sample["query"],
                    "question_type": sample["question_type"],
                    "expected_behavior": sample["expected_behavior"],
                    "expected_faq_id": sample.get("expected_faq_id"),
                    "predicted_faq_id": predicted,
                    "top1_question": questions[int(final_indices[index])],
                    "decision_score": round(float(final_scores[index]), 6),
                    "accepted": accepted,
                    "correct": correct,
                }
            )

    final_config = {
        "status": "PASS",
        "method": final_method,
        "threshold": final["threshold"],
        "lexical_weight": final_weight,
        "dense_weight": None if final_weight is None else round(1.0 - final_weight, 6),
        "metrics": _public_metrics(final),
        "faq_count": len(faq_rows),
        "evaluation_sample_count": len(samples),
    }
    CONFIG_PATH.write_text(
        json.dumps(final_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"baseline": _public_metrics(baseline), "selected": final_config}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
