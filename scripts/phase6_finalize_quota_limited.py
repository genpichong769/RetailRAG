"""Finalize PHASE 6 artifacts after the user cancelled quota-limited LLM reruns."""

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(name, default=None):
    path = ROOT / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def save(name, value):
    path = ROOT / name
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def main():
    metrics = load("RETAILRAG_PHASE6_METRICS.json")
    final_metrics = load("RETAILRAG_FINAL_METRICS.json")
    plus_generation = load("RETAILRAG_PHASE6_GENERATION_RESULTS_QWEN36PLUS.json")
    plus_full = load("RETAILRAG_PHASE6_FULL_RAG_RESULTS_QWEN36PLUS_PARTIAL.json")
    flash_generation = load(
        "RETAILRAG_PHASE6_GENERATION_RESULTS_QWEN36FLASH_INCOMPLETE.json"
    )
    plus_minimal = load("RETAILRAG_LLM_MIGRATION_PLUS_SMOKE.json")
    flash_minimal = load("RETAILRAG_LLM_MIGRATION_FLASH_SMOKE.json")
    plus_smoke = load("RETAILRAG_PHASE6_LLM_SMOKE_QWEN36PLUS.json")
    flash_smoke = load("RETAILRAG_PHASE6_LLM_SMOKE_QWEN36FLASH.json")
    audit = load("RETAILRAG_PHASE6_LLM_AUDIT.json", {})
    ablation = load("RETAILRAG_PHASE6_ABLATION.json")
    failures = load("RETAILRAG_PHASE6_FAILURES.json", {})

    evaluated_at = datetime.now().isoformat(timespec="seconds")
    previous_generation = metrics.get("generation", {})
    previous_full = metrics.get("full_rag_350", {})
    final_status = {
        "PROJECT_STATE": "FROZEN_WITH_KNOWN_LIMITATIONS",
        "QWEN3_MAX": "BLOCKED_ARREARAGE",
        "QWEN3_6_PLUS_GENERATION": "80/80 COMPLETE",
        "QWEN3_6_PLUS_FULL_RAG": "133/350 INCOMPLETE_QUOTA",
        "QWEN3_6_FLASH": "SMOKE_PASS_BUT_FULL_EVAL_CANCELLED",
        "FINAL_LLM_REGRESSION": "BLOCKED_BY_API_QUOTA",
        "MIXED_MODEL_RESULTS": "NO",
        "CODE_CHANGE_REQUIRED": "NO",
    }

    flash_generation.update(
        {
            "experiment_name": "QWEN36_FLASH_GENERATION_INCOMPLETE_EXPERIMENT",
            "status": "INCOMPLETE_EXPERIMENT",
            "included_in_final_metrics": False,
            "eligible_for_final_same_model_regression": False,
            "cancellation_reason": (
                "Cancelled by project decision after 20/80 rows to avoid another "
                "quota-limited partial run. The rows are retained for audit only."
            ),
        }
    )
    save(
        "RETAILRAG_PHASE6_GENERATION_RESULTS_QWEN36FLASH_INCOMPLETE.json",
        flash_generation,
    )

    plus_full.update(
        {
            "status": "INCOMPLETE_QUOTA",
            "official_progress_at_quota_exhaustion": "133/350",
            "included_in_final_metrics": False,
            "eligible_for_final_same_model_regression": False,
        }
    )
    save("RETAILRAG_PHASE6_FULL_RAG_RESULTS_QWEN36PLUS_PARTIAL.json", plus_full)
    metrics.update(
        {
            "evaluated_at": evaluated_at,
            "status": "FROZEN_WITH_KNOWN_LIMITATIONS",
            "project_state": "FROZEN_WITH_KNOWN_LIMITATIONS",
            "configured_default_llm_model": "qwen3-max",
            "final_llm_model": None,
            "llm_migration_reason": (
                "qwen3-max returned Arrearage. qwen3.6-plus passed migration and "
                "completed Generation 80, but its 1M free quota was exhausted "
                "during Full RAG. qwen3.6-flash passed smoke testing but the "
                "large run was cancelled to avoid another partial experiment."
            ),
            "generation_model_changed": True,
            "generation_eval_80_completed": True,
            "generation_eval_80_model": "qwen3.6-plus",
            "full_rag_350_completed": False,
            "llm_final_regression": "BLOCKED_BY_API_QUOTA",
            "mixed_model_results": False,
            "code_change_required": False,
            "fastapi_regression": "PASS_WITH_LLM_BLOCKER",
            "websocket_regression": "PASS_TRANSPORT_WITH_LLM_BLOCKER",
            "browser_e2e_regression": "PASS_WITH_LLM_BLOCKER",
            "final_status": final_status,
        }
    )
    metrics["generation"] = {
        "qwen3_max_history": {
            "expanded_baseline_reference": previous_generation.get(
                "expanded_baseline_reference"
            ),
            "phase6_reproduction_before_migration": previous_generation.get(
                "phase6_fresh_reproduction"
            ),
            "status": "BLOCKED_ARREARAGE",
        },
        "qwen36_plus_generation_experiment": {
            "status": "COMPLETED",
            "execution_count": plus_generation["execution_count"],
            "successfully_judged": plus_generation["successfully_judged"],
            "execution_errors": plus_generation["execution_errors"],
            "generator_model": plus_generation["generator_model"],
            "judge_model": plus_generation["judge_model"],
            "same_family_optimism_warning": True,
            "metrics": plus_generation["metrics"],
            "unanswerable_refusal_accuracy": plus_generation.get(
                "unanswerable_refusal_accuracy"
            ),
            "partial_answer_accuracy": plus_generation.get(
                "partial_answer_accuracy"
            ),
            "strictly_comparable_to_qwen3_max": False,
            "eligible_as_complete_final_regression": False,
        },
        "qwen36_flash_incomplete_experiment": {
            "status": "INCOMPLETE_EXPERIMENT",
            "execution_count": flash_generation.get("execution_count"),
            "successfully_judged": flash_generation.get("successfully_judged"),
            "execution_errors": flash_generation.get("execution_errors"),
            "model": flash_generation.get("generator_model"),
            "included_in_final_metrics": False,
        },
        "final_same_model_regression": "BLOCKED_BY_API_QUOTA",
        "mixed_model_results": False,
    }
    metrics["full_rag_350"] = {
        "qwen3_max_history": previous_full.get("expanded_baseline_reference"),
        "qwen36_plus_partial": {
            "status": "INCOMPLETE_QUOTA",
            "official_progress_at_quota_exhaustion": "133/350",
            "raw_checkpoint_execution_count": plus_full.get("execution_count"),
            "raw_checkpoint_successes": plus_full.get("execution_successes"),
            "raw_checkpoint_errors": plus_full.get("execution_errors"),
            "model": plus_full.get("final_llm_model"),
            "provider_error": plus_full.get("quota_exhaustion"),
            "included_as_complete_final_regression": False,
            "raw_checkpoint_note": plus_full.get("raw_checkpoint_note"),
        },
        "qwen36_flash": "FULL_EVAL_CANCELLED",
        "phase6_final_full_rerun": "BLOCKED_BY_API_QUOTA",
        "mixed_model_results": False,
    }
    metrics["llm_migration"] = {
        "old_model": "qwen3-max",
        "old_model_status": "ARREARAGE",
        "plus_minimal_smoke": plus_minimal,
        "plus_module_smoke_status": plus_smoke.get("status"),
        "plus_quota_failure": plus_full.get("quota_exhaustion"),
        "flash_minimal_smoke": flash_minimal,
        "flash_module_smoke_status": flash_smoke.get("status"),
        "flash_large_eval_status": "CANCELLED",
        "selected_final_model": None,
        "configured_default_model": "qwen3-max",
    }
    metrics["acceptance_lines"]["fresh_generation_context_recall_ge_0_85"] = False
    metrics["freeze_recommendation"] = "NO_FINAL_GIT_TAG_UNTIL_UNIFIED_LLM_REGRESSION"
    metrics["blockers"] = [
        "qwen3-max is blocked by account Arrearage.",
        "qwen3.6-plus exhausted its free quota after the complete 80-case generation experiment and at the 133/350 Full RAG quota boundary.",
        "A single model with confirmed sufficient quota, or a quota-free local model, is required before restarting the complete LLM regression.",
    ]
    save("RETAILRAG_PHASE6_METRICS.json", metrics)

    final_metrics["phase6_final"] = metrics
    final_metrics["phase"] = "PHASE 6 FINAL TARGETED FIX"
    final_metrics["status"] = metrics["status"]
    final_metrics["evaluated_at"] = evaluated_at
    save("RETAILRAG_FINAL_METRICS.json", final_metrics)

    for row in ablation.get("combinations", []):
        row["fresh_generation_interaction"] = "NOT_RERUN_AFTER_LLM_MIGRATION"
    ablation["status"] = "COMPLETED_NON_LLM_COMPONENT_ABLATION"
    ablation["limitation"] = (
        "The eight combinations retain the completed frozen non-LLM component "
        "measurements. No cross-model generation values are imputed: qwen3-max "
        "is in Arrearage, qwen3.6-plus exhausted its free quota during Full RAG, "
        "and the qwen3.6-flash large run was cancelled."
    )
    save("RETAILRAG_PHASE6_ABLATION.json", ablation)

    audit.update(
        {
            "status": "COMPLETED_WITH_QUOTA_LIMITATION",
            "old_model": "qwen3-max",
            "old_model_status": "ARREARAGE",
            "configured_default_model": "qwen3-max",
            "selected_final_model": None,
            "final_llm_regression": "BLOCKED_BY_API_QUOTA",
            "migration_attempts": {
                "qwen3.6-plus": {
                    "minimal_smoke": "PASS",
                    "module_smoke": "PASS",
                    "generation_80": "80/80 COMPLETE",
                    "full_rag_350": "133/350 INCOMPLETE_QUOTA",
                    "provider_error": "403 AllocationQuota.FreeTierOnly",
                },
                "qwen3.6-flash": {
                    "minimal_smoke": "PASS",
                    "module_smoke": "PASS",
                    "large_evaluation": "CANCELLED",
                    "partial_generation_rows": flash_generation.get("execution_count"),
                    "status": "INCOMPLETE_EXPERIMENT",
                },
            },
            "mixed_model_results": False,
        }
    )
    audit.pop("selected_model", None)
    save("RETAILRAG_PHASE6_LLM_AUDIT.json", audit)

    failures.update(
        {
            "status": "COMPLETED_WITH_LLM_QUOTA_BLOCKER",
            "project_state": "FROZEN_WITH_KNOWN_LIMITATIONS",
            "llm_final_regression": "BLOCKED_BY_API_QUOTA",
            "quota_blocker": {
                "qwen3-max": "ARREARAGE",
                "qwen3.6-plus": "403 AllocationQuota.FreeTierOnly at the official 133/350 boundary",
                "qwen3.6-flash": "FULL_EVAL_CANCELLED",
            },
            "quota_blocker_is_test_failure": False,
        }
    )
    save("RETAILRAG_PHASE6_FAILURES.json", failures)

    faq = metrics["faq"]
    retrieval = metrics["retrieval"]["same_online_k3_context_ablation_235"]
    evidence = metrics["evidence_gate"]
    gen = plus_generation["metrics"]
    report = f"""# RetailRAG PHASE 6 Final Targeted Fix Report

## Executive decision

RetailRAG remains **FROZEN_WITH_KNOWN_LIMITATIONS**. The frozen FAQ, knowledge documents and benchmark data were not changed. FAQ Complexity Gate, Dynamic M, Evidence Gate, Query Rewrite and non-LLM engineering regression remain complete. The final single-model LLM regression is **BLOCKED_BY_API_QUOTA**, not failed.

## Frozen non-LLM PHASE 6 results

| Item | Before | PHASE 6 |
|---|---:|---:|
| FAQ F1 | {faq['before']['f1']:.6f} | {faq['after']['f1']:.6f} |
| FAQ false accept rate | {faq['before']['false_accept_rate']:.6f} | {faq['after']['false_accept_rate']:.6f} |
| Complex FAQ wrong intercepts | {faq['before']['complex_query_wrong_intercept_count']} | {faq['after']['complex_query_wrong_intercept_count']} |
| Same-K3 All@3 | {retrieval['fixed_m2']['all_at_3']:.6f} | {retrieval['dynamic_m']['all_at_3']:.6f} |
| Context recall | {retrieval['fixed_m2']['context_recall']:.6f} | {retrieval['dynamic_m']['context_recall']:.6f} |
| Evidence holdout refusal recall | — | {evidence['holdout']['refusal_recall']:.6f} |

All four frozen benchmark/FAQ hashes continue to match the pre-PHASE6 snapshot. The 80 knowledge documents also remain byte-identical to the pre-PHASE6 backup.

## LLM FINAL REGRESSION LIMITATION

- The configured historical model `qwen3-max` returns `Arrearage` for real SDK calls.
- `qwen3.6-plus` passed the minimal call and all module smoke tests.
- `qwen3.6-plus` completed the fixed Generation Evaluation: **80/80**, with **0 execution errors**.
- During the complete Full RAG regression, `qwen3.6-plus` reached the **133/350 quota boundary** and then returned HTTP 403 `AllocationQuota.FreeTierOnly`.
- The raw cancellation checkpoint contains 135 successful rows because two already in-flight or FAQ-direct tasks completed while cancellation propagated. The experiment remains incomplete and is not reported as 350/350.
- `qwen3.6-flash` passed minimal and module smoke tests. Its large run was actively cancelled after 20 Generation rows to avoid exhausting another limited free quota.
- Samples from different models are not combined. No incomplete Full RAG result is presented as a final complete regression.

## qwen3.6-plus Generation experiment

This experiment is retained as `QWEN36_PLUS_GENERATION_EXPERIMENT`. Generator and Judge both used `qwen3.6-plus`, so same-family scoring may be optimistic. It is not a controlled comparison with historical `qwen3-max` metrics.

| Metric | Result |
|---|---:|
| Execution | 80/80 |
| Execution errors | 0 |
| Faithfulness | {gen['faithfulness']:.6f} |
| Answer relevance | {gen['answer_relevance']:.6f} |
| Context precision | {gen['context_precision']:.6f} |
| Context recall | {gen['context_recall']:.6f} |
| Unanswerable refusal accuracy | {plus_generation.get('unanswerable_refusal_accuracy'):.6f} |
| Partial-answer accuracy | {plus_generation.get('partial_answer_accuracy'):.6f} |

## Engineering and online status

MySQL, Redis, Milvus, MinIO, etcd and FastAPI health checks passed in the completed PHASE 6 engineering regression. FAQ Direct, complex-query-to-RAG routing, evidence refusal, WebSocket transport and browser rendering were verified. The post-migration full online quality regression was not restarted after quota cancellation; its status remains `PASS_WITH_LLM_BLOCKER` rather than a complete model acceptance.

## Historical comparability

Historical PHASE 5 and Expanded Scale generation metrics used `qwen3-max`. The retained Generation 80 experiment used `qwen3.6-plus`. Retrieval-side PHASE 6 metrics are unchanged, but generation scores across these model versions are not treated as strictly controlled comparisons.

## Restart condition

Do not restart Generation 80 or Full RAG 350 until one condition is met:

1. `qwen3-max` account access is restored;
2. one model has confirmed quota sufficient for the entire unified run; or
3. a local model is deployed without API-token quota limits.

No production-readiness claim or final Git tag is recommended before that unified run completes.

## Final status

```text
PROJECT_STATE: FROZEN_WITH_KNOWN_LIMITATIONS
QWEN3_MAX: BLOCKED_ARREARAGE
QWEN3_6_PLUS_GENERATION: 80/80 COMPLETE
QWEN3_6_PLUS_FULL_RAG: 133/350 INCOMPLETE_QUOTA
QWEN3_6_FLASH: SMOKE_PASS_BUT_FULL_EVAL_CANCELLED
FINAL_LLM_REGRESSION: BLOCKED_BY_API_QUOTA
MIXED_MODEL_RESULTS: NO
CODE_CHANGE_REQUIRED: NO
```
"""
    (ROOT / "RETAILRAG_PHASE6_FINAL_REPORT.md").write_text(report, encoding="utf-8")

    audit_lines = [
        "# RetailRAG PHASE 6 LLM Runtime Audit",
        "",
        "- Configured default: `qwen3-max`",
        "- Default status: `ARREARAGE`",
        "- Final unified LLM regression: `BLOCKED_BY_API_QUOTA`",
        "- Mixed-model metrics: `NO`",
        "- Credential value logged: `NO`",
        "",
        "The runtime call sites remain centralized through `Config.LLM_MODEL`; see `RETAILRAG_PHASE6_LLM_AUDIT.json` for the file/function inventory.",
        "",
        "| Model | Minimal smoke | Module smoke | Large evaluation |",
        "|---|---|---|---|",
        "| qwen3-max | SDK blocked | Not rerun | Arrearage |",
        "| qwen3.6-plus | PASS | PASS | Generation 80/80; Full RAG 133/350 quota boundary |",
        f"| qwen3.6-flash | PASS | PASS | Cancelled after {flash_generation.get('execution_count')} Generation rows |",
        "",
        "Historical `qwen3-max`, plus, and flash artifacts are kept separately and are not merged.",
    ]
    (ROOT / "RETAILRAG_PHASE6_LLM_AUDIT.md").write_text(
        "\n".join(audit_lines) + "\n", encoding="utf-8"
    )

    failure_md_path = ROOT / "RETAILRAG_PHASE6_FAILURES.md"
    if failure_md_path.exists():
        failure_md = failure_md_path.read_text(encoding="utf-8")
        marker = "## Final LLM regression status"
        if marker in failure_md:
            failure_md = failure_md[: failure_md.index(marker)].rstrip() + "\n"
        failure_md += (
            "\n## Final LLM regression status\n\n"
            "`BLOCKED_BY_API_QUOTA` is an execution limitation, not a failed "
            "component result. qwen3.6-plus stopped at the official 133/350 "
            "quota boundary, and qwen3.6-flash was cancelled before a second "
            "partial run could consume its limited quota. No mixed-model rows "
            "are included in final metrics.\n"
        )
        failure_md_path.write_text(failure_md, encoding="utf-8")

    print(json.dumps(final_status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
