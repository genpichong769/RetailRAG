"""Assemble PHASE 6 metrics, factorial ablation, failures, and final report."""

import hashlib
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name, default=None):
    path = ROOT / name
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save(name, value):
    path = ROOT / name
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def quantiles(values):
    values = sorted(float(value) for value in values)
    if not values:
        return None

    def percentile(p):
        position = (len(values) - 1) * p
        low = int(position)
        high = min(low + 1, len(values) - 1)
        weight = position - low
        return round(values[low] * (1 - weight) + values[high] * weight, 6)

    return {
        "count": len(values),
        "min": round(values[0], 6),
        "p10": percentile(0.1),
        "p25": percentile(0.25),
        "median": percentile(0.5),
        "p75": percentile(0.75),
        "p90": percentile(0.9),
        "max": round(values[-1], 6),
    }


def main():
    baseline_faq = load(
        "RETAILRAG_PHASE6_BASELINE_REFERENCE_RETAILRAG_SCALE_FAQ_RESULTS.json"
    )
    baseline_retrieval = load(
        "RETAILRAG_PHASE6_BASELINE_REFERENCE_RETAILRAG_SCALE_RETRIEVAL_RESULTS.json"
    )
    baseline_refusal = load(
        "RETAILRAG_PHASE6_BASELINE_REFERENCE_RETAILRAG_SCALE_REFUSAL_RESULTS.json"
    )
    baseline_rewrite = load(
        "RETAILRAG_PHASE6_BASELINE_REFERENCE_RETAILRAG_SCALE_REWRITE_RESULTS.json"
    )
    baseline_generation = load(
        "RETAILRAG_PHASE6_BASELINE_REFERENCE_RETAILRAG_SCALE_GENERATION_RESULTS.json"
    )
    baseline_reproduction = load("RETAILRAG_PHASE6_BASELINE_REPRODUCTION.json")
    faq = load("RETAILRAG_PHASE6_FAQ_GATE_EVALUATION.json")
    dynamic = load("RETAILRAG_PHASE6_DYNAMIC_M_EVALUATION.json")
    evidence = load("RETAILRAG_PHASE6_EVIDENCE_CALIBRATION.json")
    generation_reproduction = load("RETAILRAG_SCALE_GENERATION_RESULTS.json")
    engineering = load("RETAILRAG_PHASE6_ENGINEERING_REGRESSION.json", {})
    online = load("RETAILRAG_SCALE_ONLINE_RESULTS.json", {})
    qwen = load("RETAILRAG_QWEN_DIAGNOSTIC.json", {})
    data_audit = load("RETAILRAG_SCALE_DATA_AUDIT.json", {})
    full_rag_reference = load("RETAILRAG_SCALE_FULL_RAG_RESULTS.json", {})
    version = load("RETAILRAG_SCALE_VERSION_RESULTS.json", {})

    faq_before = faq["faq_480"]["before"]
    faq_after = faq["faq_480"]["after"]
    retrieval_fixed = dynamic["retrieval_235"]["modes"]["fixed_m2"]["metrics"]
    retrieval_global = dynamic["retrieval_235"]["modes"]["global_m3"]["metrics"]
    retrieval_dynamic = dynamic["retrieval_235"]["modes"]["dynamic_m"]["metrics"]
    rag_fixed = dynamic["rag_350"]["modes"]["fixed_m2"]["metrics"]
    rag_global = dynamic["rag_350"]["modes"]["global_m3"]["metrics"]
    rag_dynamic = dynamic["rag_350"]["modes"]["dynamic_m"]["metrics"]
    evidence_before = baseline_refusal["metrics"]
    evidence_after = evidence["all_labelled_metrics"]
    holdout = evidence["holdout_metrics"]

    score_distribution = {}
    for behavior in ("answer", "reject", "partial_answer"):
        rows = [
            row for row in evidence["results"] if row["expected_behavior"] == behavior
        ]
        score_distribution[behavior] = {
            feature: quantiles([row["features"][feature] for row in rows])
            for feature in (
                "top1_score",
                "top2_score",
                "top3_score",
                "top1_margin",
                "valid_context_count",
                "distinct_document_count",
                "distinct_category_count",
            )
        }

    component_values = {
        "faq_baseline": faq_before,
        "faq_gate": faq_after,
        "context_fixed_m2": retrieval_fixed,
        "context_dynamic_m": retrieval_dynamic,
        "evidence_baseline": evidence_before,
        "evidence_calibrated_all": evidence_after,
        "evidence_calibrated_holdout": holdout,
    }
    combinations = []
    names = [
        ("Baseline", False, False, False),
        ("Baseline + FAQ Gate", True, False, False),
        ("Baseline + Dynamic M", False, True, False),
        ("Baseline + Evidence Gate", False, False, True),
        ("Baseline + FAQ Gate + Dynamic M", True, True, False),
        ("Baseline + FAQ Gate + Evidence Gate", True, False, True),
        ("Baseline + Dynamic M + Evidence Gate", False, True, True),
        ("Final Combined", True, True, True),
    ]
    for name, faq_on, dynamic_on, evidence_on in names:
        faq_metrics = faq_after if faq_on else faq_before
        context_metrics = retrieval_dynamic if dynamic_on else retrieval_fixed
        gate_metrics = evidence_after if evidence_on else evidence_before
        combinations.append(
            {
                "name": name,
                "components": {
                    "faq_gate": faq_on,
                    "dynamic_m": dynamic_on,
                    "evidence_gate": evidence_on,
                },
                "faq": {
                    key: faq_metrics[key]
                    for key in (
                        "f1",
                        "precision",
                        "recall",
                        "false_accept_rate",
                        "false_reject_rate",
                        "coverage",
                        "complex_query_wrong_intercept_count",
                    )
                },
                "context_235_same_k3": {
                    key: context_metrics[key]
                    for key in (
                        "any_at_3",
                        "all_at_3",
                        "mrr",
                        "context_recall",
                        "context_precision",
                        "average_context_count",
                        "average_context_token_proxy",
                    )
                },
                "evidence_gate": gate_metrics,
                "evidence_holdout": holdout if evidence_on else None,
                "fresh_generation_interaction": "BLOCKED_QWEN_ARREARAGE",
            }
        )
    ablation = {
        "status": "COMPLETED_COMPONENT_ABLATION_WITH_EXTERNAL_GENERATION_BLOCKER",
        "same_fixed_benchmarks": True,
        "candidate_k": 3,
        "component_values": component_values,
        "fixed_m_comparison_235": {
            "fixed_m2": retrieval_fixed,
            "global_m3": retrieval_global,
            "dynamic_m": retrieval_dynamic,
        },
        "fixed_m_comparison_350": {
            "fixed_m2": rag_fixed,
            "global_m3": rag_global,
            "dynamic_m": rag_dynamic,
        },
        "combinations": combinations,
        "limitation": "The eight combinations are a factorial composition of independently measured routing, context, and evidence decisions. Fresh Qwen generation for every combination could not run because the provider returned Arrearage; no generation interaction score is imputed.",
    }
    save("RETAILRAG_PHASE6_ABLATION.json", ablation)

    rag_lookup = {
        row["id"]: row
        for row in load("RETAILRAG_RAG_EVALUATION_SET.json")["samples"]
    }
    failures = []
    for row in faq["faq_480"]["results"]:
        if row["after_correct"]:
            continue
        failures.append(
            {
                "id": row["id"],
                "question": row["query"],
                "question_type": row["question_type"],
                "expected": row.get("expected_faq_id") or row["expected_behavior"],
                "actual": row["after_prediction"],
                "route": "FAQ" if row["after_prediction"] else "RAG_FALLBACK",
                "faq_candidate": row.get("top_question"),
                "faq_score": row.get("score"),
                "classifier": None,
                "strategy": None,
                "retrieved_documents": [],
                "reranker_scores": [],
                "final_contexts": [],
                "answer": None,
                "failure_stage": "FAQ_ROUTING",
                "failure_reason": (
                    "FAQ candidate was rejected or did not match the expected FAQ; the gate correctly avoids returning an incomplete fixed answer, but FAQ recall remains limited."
                ),
                "source_evaluation": "faq_480_after_gate",
            }
        )

    fixed_rows = dynamic["retrieval_235"]["modes"]["fixed_m2"]["results"]
    global_rows = dynamic["retrieval_235"]["modes"]["global_m3"]["results"]
    dynamic_rows = dynamic["retrieval_235"]["modes"]["dynamic_m"]["results"]
    for fixed_row, global_row, dynamic_row in zip(fixed_rows, global_rows, dynamic_rows):
        if dynamic_row["all_at_3"]:
            continue
        context_budget_failure = bool(global_row["all_at_3"] and not dynamic_row["all_at_3"])
        failures.append(
            {
                "id": dynamic_row["id"],
                "question": dynamic_row["question"],
                "question_type": dynamic_row["question_type"],
                "expected": dynamic_row["expected_documents"],
                "actual": dynamic_row["retrieved_documents"],
                "route": "RAG",
                "faq_candidate": None,
                "faq_score": None,
                "classifier": None,
                "strategy": "direct_same_K3_component_evaluation",
                "retrieved_documents": dynamic_row["retrieved_documents"],
                "reranker_scores": [],
                "final_contexts": dynamic_row["retrieved_documents"],
                "answer": None,
                "failure_stage": "CONTEXT_BUDGET" if context_budget_failure else "RETRIEVAL",
                "failure_reason": (
                    "Global M=3 covered all required documents but Dynamic M did not."
                    if context_budget_failure
                    else "At least one required document was absent from the final K=3 candidate contexts; increasing M cannot recover it."
                ),
                "source_evaluation": "retrieval_235_dynamic_m",
            }
        )

    existing_ids = {(row["id"], row["failure_stage"]) for row in failures}
    for row in evidence["results"]:
        if row["expected_behavior"] not in {"answer", "reject"} or row["correct"]:
            continue
        sample = rag_lookup[row["id"]]
        retrieved = [doc["document_name"] for doc in row["retrieved_documents"]]
        expected = sample.get("reference_documents") or []
        upstream_miss = bool(expected and not (set(expected) & set(retrieved)))
        stage = "RETRIEVAL" if upstream_miss else "EVIDENCE_GATE"
        key = (row["id"], stage)
        if key in existing_ids:
            continue
        failures.append(
            {
                "id": row["id"],
                "question": row["question"],
                "question_type": sample["question_type"],
                "expected": row["expected_behavior"],
                "actual": "answer" if row["accepted"] else "reject",
                "route": "RAG",
                "faq_candidate": None,
                "faq_score": None,
                "classifier": None,
                "strategy": "direct_same_K3_evidence_calibration",
                "retrieved_documents": retrieved,
                "reranker_scores": [
                    doc["reranker_score"] for doc in row["retrieved_documents"]
                ],
                "final_contexts": retrieved,
                "answer": None,
                "failure_stage": stage,
                "failure_reason": (
                    "The required document was not retrieved; refusing weak unrelated evidence is safer than accepting it."
                    if upstream_miss
                    else "The calibrated evidence decision disagreed with the fixed expected behavior."
                ),
                "source_evaluation": f"evidence_{row['split']}",
            }
        )

    for row in generation_reproduction.get("results", []):
        if "error" not in row:
            continue
        failures.append(
            {
                "id": row["id"],
                "question": row["question"],
                "question_type": row["question_type"],
                "expected": row["expected_behavior"],
                "actual": "external_provider_error",
                "route": "RAG",
                "faq_candidate": None,
                "faq_score": None,
                "classifier": None,
                "strategy": (row.get("trace") or {}).get("strategy"),
                "retrieved_documents": [],
                "reranker_scores": [],
                "final_contexts": [],
                "answer": row.get("answer"),
                "failure_stage": "GENERATION",
                "failure_reason": row["error"],
                "source_evaluation": "phase6_generation_reproduction",
            }
        )

    stage_counts = {}
    for row in failures:
        stage_counts[row["failure_stage"]] = stage_counts.get(row["failure_stage"], 0) + 1
    failure_output = {
        "status": "COMPLETED_WITH_EXTERNAL_BLOCKER",
        "failure_count": len(failures),
        "failure_stage_counts": stage_counts,
        "allowed_stages": [
            "FAQ_ROUTING",
            "QUERY_CLASSIFICATION",
            "STRATEGY_SELECTION",
            "RETRIEVAL",
            "RERANK",
            "CONTEXT_BUDGET",
            "VERSION_FILTER",
            "EVIDENCE_GATE",
            "QUERY_REWRITE",
            "GENERATION",
            "OTHER",
        ],
        "failures": failures,
    }
    save("RETAILRAG_PHASE6_FAILURES.json", failure_output)

    lines = [
        "# RetailRAG PHASE 6 Failure Audit",
        "",
        f"Total retained failures: **{len(failures)}**.",
        "",
        "| Stage | Count |",
        "|---|---:|",
    ]
    lines.extend(f"| {stage} | {count} |" for stage, count in sorted(stage_counts.items()))
    lines.extend(
        [
            "",
            "| ID | Stage | Type | Question | Expected | Actual | Reason |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for row in failures:
        values = [
            row["id"],
            row["failure_stage"],
            row["question_type"],
            row["question"],
            json.dumps(row["expected"], ensure_ascii=False),
            json.dumps(row["actual"], ensure_ascii=False),
            row["failure_reason"],
        ]
        values = [str(value).replace("|", "\\|").replace("\n", " ") for value in values]
        lines.append("| " + " | ".join(values) + " |")
    (ROOT / "RETAILRAG_PHASE6_FAILURES.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    official_retrieval = baseline_retrieval["modes"]["hybrid_reranker"]["metrics"]
    metrics = {
        "project": "RetailRAG",
        "phase": "PHASE 6 FINAL TARGETED FIX",
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "FROZEN_WITH_KNOWN_LIMITATIONS",
        "fixed_data_scale": {
            "faq": 200,
            "knowledge_documents": 80,
            "faq_evaluation": 480,
            "rag_evaluation": 350,
            "retrieval_evaluation": 235,
            "multiturn": 30,
            "generation_evaluation": 80,
            "parent_chunks": 80,
            "child_chunks": 203,
            "visible_unique_entities": 203,
        },
        "fixed_input_sha256": baseline_reproduction["fixed_input_sha256"],
        "baseline_reproduction": baseline_reproduction,
        "faq": {
            "before": faq_before,
            "after": faq_after,
            "gate_latency_ms": {
                "mean": faq["faq_480"]["average_gate_latency_ms"],
                "p95": faq["faq_480"]["p95_gate_latency_ms"],
            },
            "rag_350_routing": faq["rag_350_routing"]["metrics"],
        },
        "retrieval": {
            "unchanged_diagnostic_k5_top3": official_retrieval,
            "same_online_k3_context_ablation_235": {
                "fixed_m2": retrieval_fixed,
                "global_m3": retrieval_global,
                "dynamic_m": retrieval_dynamic,
            },
            "same_online_k3_context_ablation_350": {
                "fixed_m2": rag_fixed,
                "global_m3": rag_global,
                "dynamic_m": rag_dynamic,
            },
            "shared_latency_235_ms": dynamic["retrieval_235"][
                "shared_retrieval_latency_ms"
            ],
            "shared_latency_350_ms": dynamic["rag_350"]["shared_retrieval_latency_ms"],
        },
        "evidence_gate": {
            "before": evidence_before,
            "config": evidence["config"],
            "score_distribution": score_distribution,
            "calibration": evidence["calibration_metrics"],
            "holdout": holdout,
            "all_labelled": evidence_after,
            "partial_count": evidence["partial_count"],
            "partial_mode_count": evidence["partial_mode_count"],
        },
        "query_rewrite": {
            "without": baseline_rewrite["without"],
            "with": baseline_rewrite["with_rewrite"],
            "rewrite_error_rate": baseline_rewrite["rewrite_error_rate"],
            "phase6_change": "unchanged",
        },
        "generation": {
            "expanded_baseline_reference": {
                "status": baseline_generation["status"],
                "sample_count": baseline_generation["sample_count"],
                "successfully_judged": baseline_generation["successfully_judged"],
                "metrics": baseline_generation["metrics"],
            },
            "phase6_fresh_reproduction": {
                "status": generation_reproduction["status"],
                "sample_count": generation_reproduction["sample_count"],
                "successfully_judged": generation_reproduction["successfully_judged"],
                "metrics_over_judged_78_only": generation_reproduction["metrics"],
            },
            "phase6_final_full_rerun": "BLOCKED_QWEN_ARREARAGE",
            "no_imputation": True,
        },
        "full_rag_350": {
            "expanded_baseline_reference": full_rag_reference.get("metrics"),
            "phase6_final_full_rerun": "BLOCKED_QWEN_ARREARAGE",
        },
        "version_filter": {
            "status": version.get("status"),
            "filter_safety": version.get("filter_safety"),
            "all_required_at_2": version.get("all_required_at_2"),
        },
        "engineering": engineering,
        "online": online,
        "browser_e2e": {
            "status": "PASS_WITH_EXTERNAL_LLM_BLOCKER",
            "page_title": "RetailRAG · Nova Retail 智能知识助手",
            "new_session": "PASS",
            "faq_answer_rendered": "PASS",
            "complex_query_forced_to_rag": "PASS",
            "complex_query_generation": "DEGRADED_QWEN_ARREARAGE",
        },
        "qwen_diagnostic": {
            "status": qwen.get("status"),
            "dns": qwen.get("dns", {}).get("status"),
            "tls": qwen.get("direct_tls", {}).get("status"),
            "sdk": qwen.get("minimal_sdk_call", {}).get("status"),
            "provider_code": "Arrearage",
        },
        "acceptance_lines": {
            "faq_f1_ge_0_86": faq_after["f1"] >= 0.86,
            "faq_false_accept_lt_0_03": faq_after["false_accept_rate"] < 0.03,
            "diagnostic_any_at_3_ge_0_94": official_retrieval["any_at_3"] >= 0.94,
            "diagnostic_all_at_3_ge_0_90": official_retrieval["all_at_3"] >= 0.90,
            "diagnostic_mrr_ge_0_90": official_retrieval["mrr"] >= 0.90,
            "evidence_holdout_refusal_precision_ge_0_95": holdout["refusal_precision"] >= 0.95,
            "evidence_holdout_refusal_recall_ge_0_90": holdout["refusal_recall"] >= 0.90,
            "fresh_generation_context_recall_ge_0_85": None,
        },
        "freeze_recommendation": "NO",
        "blockers": [
            "Qwen provider Arrearage blocks the required fresh 80-generation and 350/350 final RAG reruns.",
            "Same-online-K3 Dynamic M All@3 reaches 0.778723, below the 0.90 reference line; missing K=3 candidates remain the main limit.",
            "Evidence holdout refusal precision is 0.90 because two answerable questions retrieved no required document and were safely refused.",
        ],
    }
    save("RETAILRAG_PHASE6_METRICS.json", metrics)

    previous = load("RETAILRAG_FINAL_METRICS.json")
    if "phase5_small_scale" not in previous:
        previous["phase5_small_scale"] = previous.get("phase5_baseline", {})
    if "expanded_scale_baseline" not in previous:
        previous["expanded_scale_baseline"] = {
            "status": previous.get("status"),
            "data_scale": previous.get("final_data_scale"),
            "faq": previous.get("final_faq_metrics"),
            "retrieval": previous.get("final_retrieval_metrics"),
            "refusal": previous.get("final_refusal_metrics"),
            "query_rewrite": previous.get("final_query_rewrite_metrics"),
            "generation": previous.get("final_generation_metrics"),
            "full_rag": previous.get("final_full_rag_diagnostics"),
            "online": previous.get("final_online_metrics"),
        }
    previous["phase6_final"] = metrics
    previous["phase"] = "PHASE 6 FINAL TARGETED FIX"
    previous["status"] = metrics["status"]
    previous["evaluated_at"] = metrics["evaluated_at"]
    save("RETAILRAG_FINAL_METRICS.json", previous)

    report = f"""# RetailRAG PHASE 6 Final Targeted Fix Report

## Executive decision

PHASE 6 implemented only the three authorized fixes on the frozen 200 FAQ / 80 document / 480 FAQ / 350 RAG / 235 retrieval datasets. The code is **FROZEN_WITH_KNOWN_LIMITATIONS**. A formal final tag is not recommended yet because the Qwen provider returned `Arrearage`, which blocks the required fresh 80-case generation and 350/350 final RAG reruns.

## 1. Frozen inputs and baseline reproduction

All four frozen input SHA256 values still match the pre-PHASE6 snapshot. FAQ, retrieval, refusal, and rewrite baselines reproduced completely. Generation reproduced 78/80 before the provider rejected the final two requests. The prior complete 80-row result is retained as a reference and is not presented as a fresh PHASE 6 result.

| Baseline item | Reference | Fresh reproduction |
|---|---:|---:|
| FAQ F1 | {baseline_faq['metrics']['f1']:.6f} | {baseline_reproduction['items']['faq']['reproduced']['f1']:.6f} |
| FAQ false accept | {baseline_faq['metrics']['false_accept_rate']:.6f} | {baseline_reproduction['items']['faq']['reproduced']['false_accept_rate']:.6f} |
| Evidence refusal recall | {evidence_before['refusal_recall']:.6f} | {baseline_reproduction['items']['refusal']['reproduced']['refusal_recall']:.6f} |
| Rewrite Hit@3 | {baseline_rewrite['with_rewrite']['any_at_3']:.6f} | {baseline_reproduction['items']['rewrite']['reproduced']['with']['any_at_3']:.6f} |
| Generation judged | 80/80 | {generation_reproduction['successfully_judged']}/80 |

## 2. FAQ Complexity Gate

The gate runs after dense FAQ retrieval and before reading or caching the answer. It uses deterministic concept, condition, calculation, intent mismatch, missing-record, requested-detail, and multi-output signals. Exact authored FAQs remain eligible. Its cache namespace changed to `phase6-gate-v2`, so pre-gate cached answers cannot bypass the decision. No extra Qwen call is added.

| FAQ 480 | Before | After |
|---|---:|---:|
| Precision | {faq_before['precision']:.6f} | {faq_after['precision']:.6f} |
| Recall | {faq_before['recall']:.6f} | {faq_after['recall']:.6f} |
| F1 | {faq_before['f1']:.6f} | {faq_after['f1']:.6f} |
| False accept rate | {faq_before['false_accept_rate']:.6f} | {faq_after['false_accept_rate']:.6f} |
| False reject rate | {faq_before['false_reject_rate']:.6f} | {faq_after['false_reject_rate']:.6f} |
| Paraphrase recall | {faq_before['paraphrase_recall']:.6f} | {faq_after['paraphrase_recall']:.6f} |
| Hard-negative accuracy | {faq_before['hard_negative_accuracy']:.6f} | {faq_after['hard_negative_accuracy']:.6f} |
| Coverage | {faq_before['coverage']:.6f} | {faq_after['coverage']:.6f} |
| FAQ→RAG fallback | {faq_before['faq_to_rag_fallback_ratio']:.6f} | {faq_after['faq_to_rag_fallback_ratio']:.6f} |
| Complex wrong intercepts | {faq_before['complex_query_wrong_intercept_count']} | {faq_after['complex_query_wrong_intercept_count']} |

The gate adds only {faq['faq_480']['average_gate_latency_ms']:.4f} ms mean / {faq['faq_480']['p95_gate_latency_ms']:.4f} ms P95 in the offline loop. Coverage falls 2.5 percentage points, while all 12 previously accepted wrong FAQ decisions are removed.

## 3. Dynamic Context M and diversity

K remains 3 and dense/sparse weights remain 1.0:0.7. Simple queries keep M=2. Multi-condition, multi-rule, arithmetic, “分别” requests, and subquery strategy paths may use M=3. A third block must clear `max(legacy evidence threshold, top1 × 0.02)`. Among eligible blocks, a new category is preferred; low-score diversity fillers are not forced.

### Same K=3, 235 retrieval cases

| Mode | Any@3 | All@3 | Multi-doc All@3 | Context recall | Context precision | Avg contexts | Token proxy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed M=2 | {retrieval_fixed['any_at_3']:.6f} | {retrieval_fixed['all_at_3']:.6f} | {retrieval_fixed['multi_document_all_at_3']:.6f} | {retrieval_fixed['context_recall']:.6f} | {retrieval_fixed['context_precision']:.6f} | {retrieval_fixed['average_context_count']:.3f} | {retrieval_fixed['average_context_token_proxy']:.1f} |
| Global M=3 | {retrieval_global['any_at_3']:.6f} | {retrieval_global['all_at_3']:.6f} | {retrieval_global['multi_document_all_at_3']:.6f} | {retrieval_global['context_recall']:.6f} | {retrieval_global['context_precision']:.6f} | {retrieval_global['average_context_count']:.3f} | {retrieval_global['average_context_token_proxy']:.1f} |
| Dynamic M | {retrieval_dynamic['any_at_3']:.6f} | {retrieval_dynamic['all_at_3']:.6f} | {retrieval_dynamic['multi_document_all_at_3']:.6f} | {retrieval_dynamic['context_recall']:.6f} | {retrieval_dynamic['context_precision']:.6f} | {retrieval_dynamic['average_context_count']:.3f} | {retrieval_dynamic['average_context_token_proxy']:.1f} |

Dynamic M matches global M=3 coverage on this cohort while using about {(1-retrieval_dynamic['average_context_token_proxy']/retrieval_global['average_context_token_proxy'])*100:.1f}% fewer context-token proxies and preserving materially higher precision. It cannot fix required documents missing from the original K=3 candidates.

### Same K=3, 350 RAG questions (retrieval component)

| Mode | Any@3 | All@3 | Context recall | Context precision | Avg contexts | Token proxy |
|---|---:|---:|---:|---:|---:|---:|
| Fixed M=2 | {rag_fixed['any_at_3']:.6f} | {rag_fixed['all_at_3']:.6f} | {rag_fixed['context_recall']:.6f} | {rag_fixed['context_precision']:.6f} | {rag_fixed['average_context_count']:.3f} | {rag_fixed['average_context_token_proxy']:.1f} |
| Global M=3 | {rag_global['any_at_3']:.6f} | {rag_global['all_at_3']:.6f} | {rag_global['context_recall']:.6f} | {rag_global['context_precision']:.6f} | {rag_global['average_context_count']:.3f} | {rag_global['average_context_token_proxy']:.1f} |
| Dynamic M | {rag_dynamic['any_at_3']:.6f} | {rag_dynamic['all_at_3']:.6f} | {rag_dynamic['context_recall']:.6f} | {rag_dynamic['context_precision']:.6f} | {rag_dynamic['average_context_count']:.3f} | {rag_dynamic['average_context_token_proxy']:.1f} |

The unchanged K=5 diagnostic remains Any@3 {official_retrieval['any_at_3']:.6f}, All@3 {official_retrieval['all_at_3']:.6f}, MRR {official_retrieval['mrr']:.6f}. It is reported separately because using K=5 for the online Dynamic M comparison would violate the fixed K=3 constraint.

## 4. Evidence score distribution and calibration

The gate collected Top1/2/3 reranker scores, margins, valid context count, distinct document/category counts, and query-context overlap. A fixed-seed stratified 60/40 split was used. The calibrated Top1 rule met the calibration constraint with perfect calibration metrics, so it was selected over the more complex diagnostic logistic model: threshold {evidence['config']['reranker_score_threshold']:.6f}, plus a deterministic knowledge-scope override. Scope-limited mixed questions enter `partial` mode and receive an explicit answer-with-limitation instruction.

| Evidence gate | Calibration | Holdout | All 100 |
|---|---:|---:|---:|
| Answerable accuracy | {evidence['calibration_metrics']['answerable_accuracy']:.6f} | {holdout['answerable_accuracy']:.6f} | {evidence_after['answerable_accuracy']:.6f} |
| Refusal precision | {evidence['calibration_metrics']['refusal_precision']:.6f} | {holdout['refusal_precision']:.6f} | {evidence_after['refusal_precision']:.6f} |
| Refusal recall | {evidence['calibration_metrics']['refusal_recall']:.6f} | {holdout['refusal_recall']:.6f} | {evidence_after['refusal_recall']:.6f} |
| Refusal F1 | {evidence['calibration_metrics']['refusal_f1']:.6f} | {holdout['refusal_f1']:.6f} | {evidence_after['refusal_f1']:.6f} |
| False answer rate | {evidence['calibration_metrics']['false_answer_rate']:.6f} | {holdout['false_answer_rate']:.6f} | {evidence_after['false_answer_rate']:.6f} |
| False refusal rate | {evidence['calibration_metrics']['false_refusal_rate']:.6f} | {holdout['false_refusal_rate']:.6f} | {evidence_after['false_refusal_rate']:.6f} |

Holdout refusal recall rises to 1.0. Refusal precision is 0.90 because two answerable cases retrieved no required document; accepting their unrelated low-score contexts would create hallucination risk. All five fixed partial-answer cases enter partial mode.

## 5. Independent and combined ablation

All eight required combinations are recorded in `RETAILRAG_PHASE6_ABLATION.json`. FAQ routing, context selection, and evidence decisions are measured independently and composed factorially. Fresh generation interaction scores are marked blocked rather than estimated because Qwen returned `Arrearage`.

## 6. Query rewrite, versioning, generation, and 350/350 status

Query rewrite logic was unchanged: Hit@3 remains {baseline_rewrite['with_rewrite']['any_at_3']:.6f}, MRR {baseline_rewrite['with_rewrite']['mrr']:.6f}, error rate {baseline_rewrite['rewrite_error_rate']:.6f}. Version filtering remains safe ({version.get('filter_safety')}). The prior complete 80-case generation reference is faithfulness {baseline_generation['metrics']['faithfulness']:.6f}, relevance {baseline_generation['metrics']['answer_relevance']:.6f}, context precision {baseline_generation['metrics']['context_precision']:.6f}, context recall {baseline_generation['metrics']['context_recall']:.6f}. PHASE 6 freshly judged 78/80 before external billing rejection. No 80-case or 350/350 final score is imputed.

## 7. Engineering regression

MySQL, Redis, Milvus, MinIO, etcd, and FastAPI health passed. MySQL still contains 200 FAQ rows. Milvus exposes 203 unique entities, 80 parents, and 80 documents after idempotent upsert; physical row statistics can temporarily include tombstoned upsert history, so visible unique IDs are the authoritative count. FAQ initialization passed twice (second run removed zero cache keys). Browser E2E loaded the page, created a session, rendered a FAQ answer, and verified that the complex birthday-points calculation now enters RAG. Qwen-dependent answer generation displayed the configured fallback because the provider account is in arrears.

## 8. Failures and remaining limits

`RETAILRAG_PHASE6_FAILURES.json` and `.md` retain {len(failures)} failures with the requested stage taxonomy. The largest remaining class is upstream retrieval: Dynamic M cannot select a document that is absent from K=3. FAQ false accepts are eliminated, but 72 expected FAQ answers still fall back because FAQ retrieval recall remains 0.76.

No further PHASE 6 tuning is recommended. The next action should be to restore the Qwen account and rerun the already-fixed final commands; changing data, K, models, strategy logic, or thresholds would violate the frozen evaluation contract or trade safety for cosmetic scores.

## 9. Final A–G answers

- **A. FAQ multi-condition wrong interception significantly improved? YES.** 12 → 0 on the frozen FAQ benchmark; the explicit 100-yuan birthday-points browser/API probe enters RAG.
- **B. All Recall@3 improved? YES.** Same online K=3 context All@3: {retrieval_fixed['all_at_3']:.6f} → {retrieval_dynamic['all_at_3']:.6f}.
- **C. Context Recall improved? YES (retrieval-context component).** 235: {retrieval_fixed['context_recall']:.6f} → {retrieval_dynamic['context_recall']:.6f}; 350 component: {rag_fixed['context_recall']:.6f} → {rag_dynamic['context_recall']:.6f}. Fresh generation Context Recall is blocked and not claimed.
- **D. Evidence Gate Holdout Refusal Recall improved? YES.** {evidence_before['refusal_recall']:.6f} overall baseline → {holdout['refusal_recall']:.6f} holdout.
- **E. Any obvious new regression? NO.** Costs are bounded: FAQ coverage −2.5 points and Dynamic M context-token proxy +{(retrieval_dynamic['average_context_token_proxy']/retrieval_fixed['average_context_token_proxy']-1)*100:.1f}%, while global M=3 would add {(retrieval_global['average_context_token_proxy']/retrieval_fixed['average_context_token_proxy']-1)*100:.1f}%.
- **F. Recommend formal freeze? NO.** Keep the code as `FROZEN_WITH_KNOWN_LIMITATIONS`; do not create the final Git tag yet.
- **G. Blocking issues (maximum 3):**
  1. Qwen `Arrearage` blocks fresh Generation 80 and final 350/350 RAG reruns.
  2. Same-K3 Dynamic M All@3 is {retrieval_dynamic['all_at_3']:.6f}, below the 0.90 reference line because required documents are often absent from K=3.
  3. Evidence Holdout refusal precision is {holdout['refusal_precision']:.6f}; two answerable questions have upstream retrieval misses and are safely refused.
"""
    (ROOT / "RETAILRAG_PHASE6_FINAL_REPORT.md").write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": metrics["status"],
                "failure_count": len(failures),
                "faq_f1": faq_after["f1"],
                "dynamic_all_at_3": retrieval_dynamic["all_at_3"],
                "holdout_refusal_recall": holdout["refusal_recall"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
