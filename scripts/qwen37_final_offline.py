"""Run the frozen, non-generative portions of FINAL FULL EVALUATION V1.

The script writes qwen3.7-specific result files without replacing historical
PHASE 5/6 artifacts.  It does not tune thresholds or modify production code.
"""

import argparse
import copy
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from base import Config
from base.query_analysis import analyze_query
from mysql_qa.utils.preprocess import preprocess_text
from rag_qa.core.new_rag_system import RAGSystem
from rag_qa.core.vector_store import VectorStore
import evaluate_scale as scale
import phase6_dynamic_m_evaluation as dynamic_eval
from phase6_faq_gate_evaluation import metrics as faq_metrics


FAQ_BASE = "RETAILRAG_QWEN37_FINAL_FAQ_480_BASE.json"
FAQ_OUT = "RETAILRAG_QWEN37_FINAL_FAQ_480.json"
RETRIEVAL_OUT = "RETAILRAG_QWEN37_FINAL_RETRIEVAL_235.json"
VERSION_OUT = "RETAILRAG_QWEN37_FINAL_VERSION.json"
EVIDENCE_OUT = "RETAILRAG_QWEN37_FINAL_EVIDENCE_GATE.json"
DYNAMIC_OUT = "RETAILRAG_QWEN37_FINAL_DYNAMIC_M.json"


def load(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def save(name, value):
    path = ROOT / name
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def average(values):
    values = [float(value) for value in values if value is not None]
    return round(sum(values) / len(values), 6) if values else None


def percentile(values, p):
    values = sorted(float(value) for value in values if value is not None)
    if not values:
        return None
    position = (len(values) - 1) * p / 100
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return round(values[lower] * (1 - weight) + values[upper] * weight, 6)


def distribution(values):
    return {
        "count": len([value for value in values if value is not None]),
        "mean": average(values),
        "median": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
    }


def rank_of_expected(row):
    expected = set(row.get("expected_documents") or [])
    return next(
        (
            index
            for index, name in enumerate(row.get("retrieved_documents") or [], 1)
            if name in expected
        ),
        None,
    )


def run_faq(store, cfg):
    original_save = scale.save

    def redirect(name, value):
        if name == "RETAILRAG_SCALE_FAQ_RESULTS.json":
            save(FAQ_BASE, value)
        else:
            original_save(name, value)

    scale.save = redirect
    try:
        scale.run_faq(store, cfg)
    finally:
        scale.save = original_save

    base = load(FAQ_BASE)
    rows = []
    gate_latencies = []
    for source in base["results"]:
        import time

        started = time.perf_counter()
        analysis = analyze_query(source["query"], source.get("top_question"))
        gate_latencies.append((time.perf_counter() - started) * 1000)
        before = source.get("predicted_faq_id")
        after = None if before and analysis["faq_force_rag"] else before
        wants_answer = source["expected_behavior"] == "answer"
        rows.append(
            {
                **source,
                "before_prediction": before,
                "after_prediction": after,
                "predicted_faq": after,
                "route": "FAQ_DIRECT" if after else "RAG_FALLBACK",
                "dense_score": source.get("score"),
                "gate_blocked": bool(before and analysis["faq_force_rag"]),
                "gate_reasons": analysis["faq_gate_reasons"],
                "query_analysis": analysis,
                "before_correct": before == source.get("expected_faq_id")
                if wants_answer
                else before is None,
                "after_correct": after == source.get("expected_faq_id")
                if wants_answer
                else after is None,
            }
        )

    before_metrics = faq_metrics(rows, "before_prediction")
    final_metrics = faq_metrics(rows, "after_prediction")
    for key, prediction_key, target in (
        ("before", "before_prediction", before_metrics),
        ("after", "after_prediction", final_metrics),
    ):
        target["complex_query_wrong_intercept_count"] = sum(
            row[prediction_key] is not None
            and not row[f"{key}_correct"]
            and (
                row["query_analysis"]["context_complex"]
                or bool(row["query_analysis"]["faq_gate_reasons"])
            )
            for row in rows
        )

    with (ROOT / "mysql_qa/data/retail_faq.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        faq_rows = list(csv.DictReader(handle))
    question_field = list(faq_rows[0])[1]
    questions = [row[question_field] for row in faq_rows]
    bm25 = BM25Okapi([preprocess_text(question) for question in questions])
    failures = []
    for row in rows:
        if row["after_correct"]:
            continue
        raw_scores = np.asarray(bm25.get_scores(preprocess_text(row["query"])), dtype=float)
        probabilities = np.exp(raw_scores - raw_scores.max())
        probabilities /= probabilities.sum()
        best = int(probabilities.argmax())
        reason = (
            "Expected FAQ was rejected or a different FAQ was selected."
            if row["expected_behavior"] == "answer"
            else "A non-FAQ or hard-negative query was incorrectly accepted."
        )
        failures.append(
            {
                "id": row["id"],
                "query": row["query"],
                "question_type": row["question_type"],
                "expected_behavior": row["expected_behavior"],
                "expected_faq": row.get("expected_faq_id"),
                "predicted_faq": row.get("after_prediction"),
                "dense_score": row.get("dense_score"),
                "bm25_top_faq": f"faq_{best + 1:03d}",
                "bm25_score": round(float(raw_scores[best]), 9),
                "softmax_score": round(float(probabilities[best]), 9),
                "route": row["route"],
                "gate_reasons": row["gate_reasons"],
                "failure_reason": reason,
            }
        )

    type_metrics = {}
    for kind in ("exact", "paraphrase", "hard_negative", "non_faq"):
        selected = [row for row in rows if row["question_type"] == kind]
        type_metrics[kind] = faq_metrics(selected, "after_prediction")
    complex_rows = [row for row in rows if row["query_analysis"]["context_complex"]]
    type_metrics["complex"] = faq_metrics(complex_rows, "after_prediction")

    output = {
        "status": "COMPLETED",
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "final_model_context": "qwen3.7-plus (FAQ routing itself is deterministic)",
        "sample_count": len(rows),
        "threshold": cfg.FAQ_MATCH_THRESHOLD,
        "retrieval_mode": cfg.FAQ_RETRIEVAL_MODE,
        "before_gate_metrics": before_metrics,
        "metrics": final_metrics,
        "category_metrics": type_metrics,
        "gate_latency_ms": distribution(gate_latencies),
        "failure_count": len(failures),
        "failures": failures,
        "results": rows,
    }
    save(FAQ_OUT, output)
    print("FAQ_FINAL", json.dumps(final_metrics, ensure_ascii=False), flush=True)


def unified_metrics(rows):
    valid = [row for row in rows if row.get("expected_documents")]
    multi = [row for row in valid if len(row["expected_documents"]) > 1]
    result = {"count": len(rows), "reference_bearing_count": len(valid)}
    for k in (1, 3, 5):
        result[f"recall_at_{k}"] = average(
            [row.get(f"document_recall_at_{k}") for row in valid]
        )
        result[f"hit_at_{k}"] = average([row.get(f"any_at_{k}") for row in valid])
        result[f"any_recall_at_{k}"] = result[f"hit_at_{k}"]
        result[f"all_recall_at_{k}"] = average(
            [row.get(f"all_at_{k}") for row in multi]
        )
    result["mrr"] = average([row.get("reciprocal_rank") for row in valid])
    return result


def enrich_retrieval():
    output = load(RETRIEVAL_OUT)
    for mode_name, mode in output["modes"].items():
        rows = mode["results"]
        mode["unified_metrics"] = unified_metrics(rows)
        categories = sorted({row["question_type"] for row in rows})
        mode["category_metrics"] = {
            category: unified_metrics(
                [row for row in rows if row["question_type"] == category]
            )
            for category in categories
        }

    before_rows = {
        row["id"]: row for row in output["modes"]["hybrid"]["results"]
    }
    after_rows = output["modes"]["hybrid_reranker"]["results"]
    comparisons = []
    improved = unchanged = degraded = 0
    for after in after_rows:
        before = before_rows[after["id"]]
        before_rank = rank_of_expected(before)
        after_rank = rank_of_expected(after)
        before_value = before_rank if before_rank is not None else math.inf
        after_value = after_rank if after_rank is not None else math.inf
        change = (
            "IMPROVED"
            if after_value < before_value
            else "DEGRADED"
            if after_value > before_value
            else "UNCHANGED"
        )
        improved += int(change == "IMPROVED")
        unchanged += int(change == "UNCHANGED")
        degraded += int(change == "DEGRADED")
        comparisons.append(
            {
                "id": after["id"],
                "question_type": after["question_type"],
                "expected_documents": after["expected_documents"],
                "correct_document_rank_before": before_rank,
                "correct_document_rank_after": after_rank,
                "change": change,
            }
        )
    output["reranker_ablation"] = {
        "hybrid": output["modes"]["hybrid"]["unified_metrics"],
        "hybrid_reranker": output["modes"]["hybrid_reranker"]["unified_metrics"],
        "reranker_improved_count": improved,
        "reranker_unchanged_count": unchanged,
        "reranker_degraded_count": degraded,
        "comparisons": comparisons,
    }
    output["evaluated_at"] = datetime.now().isoformat(timespec="seconds")
    output["final_model_context"] = "qwen3.7-plus (retrieval is local and deterministic)"
    save(RETRIEVAL_OUT, output)


def run_retrieval(store, cfg):
    original_save = scale.save
    mapping = {
        "RETAILRAG_SCALE_RETRIEVAL_RESULTS.json": RETRIEVAL_OUT,
        "RETAILRAG_SCALE_VERSION_RESULTS.json": VERSION_OUT,
    }

    def redirect(name, value):
        save(mapping.get(name, name), value)

    scale.save = redirect
    try:
        scale.run_retrieval(store, cfg)
    finally:
        scale.save = original_save
    enrich_retrieval()


def run_dynamic(store, cfg):
    retrieval = load("RETAILRAG_SCALE_RETRIEVAL_SET.json")["samples"]
    rag = load("RETAILRAG_RAG_EVALUATION_SET.json")["samples"]
    output = {
        "status": "COMPLETED",
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "design": {
            "candidate_k": 3,
            "fixed_m": 2,
            "dynamic_m": "2 for simple queries, up to 3 for complex queries",
            "thresholds_changed": False,
        },
        "retrieval_235": dynamic_eval.evaluate_cohort(
            "retrieval_235", retrieval, store, cfg
        ),
        "rag_350": dynamic_eval.evaluate_cohort("rag_350", rag, store, cfg),
    }
    for cohort in ("retrieval_235", "rag_350"):
        for mode in ("fixed_m2", "dynamic_m"):
            rows = output[cohort]["modes"][mode]["results"]
            output[cohort]["modes"][mode]["category_metrics"] = {
                kind: dynamic_eval.metrics(
                    [row for row in rows if row["question_type"] == kind]
                )
                for kind in ("single_document", "multi_document")
            }
    save(DYNAMIC_OUT, output)
    retrieval_output = load(RETRIEVAL_OUT)
    retrieval_output["dynamic_m_ablation"] = output["retrieval_235"]
    save(RETRIEVAL_OUT, retrieval_output)
    print(
        "DYNAMIC_M_FINAL",
        json.dumps(
            {
                key: value["metrics"]
                for key, value in output["retrieval_235"]["modes"].items()
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def run_evidence(store, cfg):
    import time

    all_samples = load("RETAILRAG_RAG_EVALUATION_SET.json")["samples"]
    samples = [
        sample
        for sample in all_samples
        if sample["question_type"] in {"single_document", "unanswerable"}
        or sample["expected_behavior"] == "partial_answer"
    ]
    rag = RAGSystem(store, lambda _prompt: iter(()))
    rows = []
    gate_latencies = []
    for index, sample in enumerate(samples, 1):
        ranked = store.search_documents(
            sample["question"], mode="hybrid", rerank=True, candidate_k=3, limit=3
        )
        selected, selection = dynamic_eval.dynamic_select(
            sample["question"], ranked, cfg
        )
        started = time.perf_counter()
        decision = rag._evidence_decision(sample["question"], selected)
        gate_latencies.append((time.perf_counter() - started) * 1000)
        rows.append(
            {
                **sample,
                "dynamic_m": selection,
                "decision": decision,
                "accepted": decision["accepted"],
                "mode": decision["mode"],
                "retrieved_documents": [
                    {
                        "document_name": doc.metadata.get("document_name"),
                        "category": doc.metadata.get("category"),
                        "reranker_score": doc.metadata.get("reranker_score"),
                    }
                    for doc in selected
                ],
            }
        )
        if index % 20 == 0:
            print(f"evidence {index}/{len(samples)}", flush=True)

    answer = [row for row in rows if row["expected_behavior"] == "answer"]
    reject = [row for row in rows if row["expected_behavior"] == "reject"]
    partial = [
        row for row in rows if row["expected_behavior"] == "partial_answer"
    ]
    true_refusal = sum(row["mode"] == "refuse" for row in reject)
    false_refusal = sum(row["mode"] == "refuse" for row in answer)
    false_answer = len(reject) - true_refusal
    precision = true_refusal / max(true_refusal + false_refusal, 1)
    recall = true_refusal / max(len(reject), 1)
    metrics = {
        "sample_count": len(rows),
        "answerable_count": len(answer),
        "unanswerable_count": len(reject),
        "partial_count": len(partial),
        "answerable_accuracy": round(
            sum(row["accepted"] for row in answer) / max(len(answer), 1), 6
        ),
        "unanswerable_refusal_accuracy": round(recall, 6),
        "partial_mode_accuracy": round(
            sum(row["mode"] == "partial" for row in partial)
            / max(len(partial), 1),
            6,
        ),
        "refusal_precision": round(precision, 6),
        "refusal_recall": round(recall, 6),
        "refusal_f1": round(
            2 * precision * recall / max(precision + recall, 1e-12), 6
        ),
        "false_refusal_rate": round(false_refusal / max(len(answer), 1), 6),
        "false_answer_rate": round(false_answer / max(len(reject), 1), 6),
        "true_refusal": true_refusal,
        "false_refusal": false_refusal,
        "false_accept_unanswerable": false_answer,
    }
    output = {
        "status": "COMPLETED",
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "threshold_tuned_in_this_run": False,
        "uses_current_query_aware_gate": True,
        "uses_dynamic_m_contexts": True,
        "metrics": metrics,
        "gate_latency_ms": distribution(gate_latencies),
        "evaluation_conditions": {
            "fully_unanswerable": "question_type=unanswerable",
            "partially_answerable": "expected_behavior=partial_answer",
            "internal_but_absent": "covered by unanswerable cases",
            "answerable_with_evidence": "question_type=single_document",
        },
        "results": rows,
    }
    save(EVIDENCE_OUT, output)
    print("EVIDENCE_FINAL", json.dumps(metrics, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stages", default="faq,retrieval,dynamic,evidence"
    )
    args = parser.parse_args()
    cfg = Config()
    if cfg.LLM_MODEL != "qwen3.7-plus":
        raise RuntimeError(f"Unexpected final model configuration: {cfg.LLM_MODEL}")
    frozen = load("RETAILRAG_QWEN37_BENCHMARK_FREEZE.json")
    if frozen.get("status") != "PASS":
        raise RuntimeError("Frozen benchmark gate has not passed")
    store = VectorStore()
    store.reranker = scale.CachedReranker(store.reranker, cfg)
    for stage in [item.strip() for item in args.stages.split(",") if item.strip()]:
        print(f"START {stage}", flush=True)
        if stage == "faq":
            run_faq(store, cfg)
        elif stage == "retrieval":
            run_retrieval(store, cfg)
        elif stage == "dynamic":
            run_dynamic(store, cfg)
        elif stage == "evidence":
            run_evidence(store, cfg)
        else:
            raise ValueError(stage)
        print(f"DONE {stage}", flush=True)


if __name__ == "__main__":
    main()
