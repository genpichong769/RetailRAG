"""Judge the completed qwen3.7-plus Full-RAG-350 run and build failure files."""

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import httpx
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from base import Config


RESULTS_PATH = ROOT / "RETAILRAG_QWEN37_FINAL_350_RESULTS.json"
FAILURES_PATH = ROOT / "RETAILRAG_QWEN37_FINAL_FAILURES.json"
FAILURES_MD_PATH = ROOT / "RETAILRAG_QWEN37_FINAL_FAILURES.md"
CATEGORY_PATH = ROOT / "RETAILRAG_QWEN37_FINAL_CATEGORY_METRICS.json"

LABELS = {
    "CORRECT",
    "PARTIALLY_CORRECT",
    "INCORRECT",
    "CORRECT_REFUSAL",
    "WRONG_REFUSAL",
    "EXECUTION_ERROR",
}
FAILURE_STAGES = (
    "FAQ_ROUTING",
    "QUERY_REWRITE",
    "QUERY_CLASSIFIER",
    "STRATEGY_SELECTION",
    "RETRIEVAL",
    "RERANK",
    "CONTEXT_BUDGET",
    "VERSION_FILTER",
    "EVIDENCE_GATE",
    "GENERATION",
    "MULTIPLE_STAGES",
    "SYSTEM_ERROR",
    "OTHER",
)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def mean(values):
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


def dist(values):
    return {
        "count": len([value for value in values if value is not None]),
        "mean": mean(values),
        "median": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
    }


def judge_prompt(row):
    trace = row.get("trace") or {}
    return f'''你是固定标准的 RetailRAG 最终裁判。只依据给定 ground truth、期望行为和实际检索证据判断，不用外部知识补全。
分类只能取：CORRECT、PARTIALLY_CORRECT、INCORRECT、CORRECT_REFUSAL、WRONG_REFUSAL。
规则：
1. CORRECT 必须完整、准确覆盖可回答部分，且没有无证据的具体事实。
2. PARTIALLY_CORRECT 表示有实质正确内容但遗漏重要条件或包含局部错误，不能计作严格正确。
3. 期望 reject 且回答明确、适当地拒绝无证据内容才是 CORRECT_REFUSAL。
4. 有足够证据却拒答，或 partial_answer 把全部问题拒绝，属于 WRONG_REFUSAL。
5. faithfulness 与 context_recall 各为 0 到 1；不适用时返回 null。
仅返回 JSON：{{"classification":"CORRECT","faithfulness":0.0,"context_recall":0.0,"rationale":"具体理由"}}

问题类型：{row.get('question_type')}
期望行为：{row.get('expected_behavior')}
问题：{row.get('question')}
ground truth：{row.get('ground_truth')}
reference documents：{json.dumps(row.get('reference_documents', []), ensure_ascii=False)}
实际 route：{row.get('route')}
实际 retrieved sources：{json.dumps(trace.get('retrieved_sources', []), ensure_ascii=False)}
实际 final context：{trace.get('final_context') or '无'}
实际回答：{row.get('answer')}'''


def is_fatal_account_error(text):
    lowered = str(text).lower()
    return any(
        marker in lowered
        for marker in (
            "insufficient balance",
            "arrearage",
            "payment required",
            "account suspended",
        )
    )


def judge_one(client, cfg, row):
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=cfg.LLM_MODEL,
        messages=[{"role": "user", "content": judge_prompt(row)}],
        temperature=0.0,
        timeout=90,
    )
    text = response.choices[0].message.content or ""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("Correctness judge did not return JSON")
    parsed = json.loads(match.group(0))
    label = parsed.get("classification")
    if label not in LABELS - {"EXECUTION_ERROR"}:
        raise ValueError(f"Invalid classification: {label}")
    for key in ("faithfulness", "context_recall"):
        if parsed.get(key) is not None:
            parsed[key] = max(0.0, min(1.0, float(parsed[key])))
    return {
        **parsed,
        "judge_model": cfg.LLM_MODEL,
        "judge_latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "judge_usage": response.usage.model_dump()
        if response.usage
        else "NOT_AVAILABLE",
    }


def failure_stage(row):
    if row.get("classification") == "EXECUTION_ERROR":
        return "SYSTEM_ERROR"
    trace = row.get("trace") or {}
    classification = trace.get("classification") or {}
    if row.get("route") == "FAQ_DIRECT":
        return "FAQ_ROUTING"
    if classification.get("routed_prediction") == "general":
        return "QUERY_CLASSIFIER"
    if row.get("classification") == "WRONG_REFUSAL":
        return "EVIDENCE_GATE"
    retrieved = {
        item.get("document_name")
        for item in trace.get("retrieved_sources", [])
        if item.get("document_name")
    }
    expected = set(row.get("reference_documents") or [])
    if expected and not (expected & retrieved):
        if row.get("question_type") == "multiturn":
            return "QUERY_REWRITE"
        if row.get("question_type") == "version_conflict":
            return "VERSION_FILTER"
        return "RETRIEVAL"
    if len(expected) > 1 and not expected.issubset(retrieved):
        return "CONTEXT_BUDGET"
    return "GENERATION"


def summarize(rows):
    counts = {label: sum(row.get("classification") == label for row in rows) for label in LABELS}
    total = len(rows)
    strict = counts["CORRECT"] + counts["CORRECT_REFUSAL"]
    answerable = [row for row in rows if row.get("expected_behavior") != "reject"]
    unanswerable = [row for row in rows if row.get("expected_behavior") == "reject"]
    return {
        "execution_attempted": total,
        "execution_success": total - counts["EXECUTION_ERROR"],
        "execution_errors": counts["EXECUTION_ERROR"],
        "correct": counts["CORRECT"],
        "partial": counts["PARTIALLY_CORRECT"],
        "incorrect": counts["INCORRECT"],
        "correct_refusal": counts["CORRECT_REFUSAL"],
        "wrong_refusal": counts["WRONG_REFUSAL"],
        "strict_correct_rate": round(strict / max(total, 1), 6),
        "partial_rate": round(counts["PARTIALLY_CORRECT"] / max(total, 1), 6),
        "incorrect_rate": round(counts["INCORRECT"] / max(total, 1), 6),
        "wrong_refusal_rate": round(counts["WRONG_REFUSAL"] / max(total, 1), 6),
        "execution_error_rate": round(counts["EXECUTION_ERROR"] / max(total, 1), 6),
        "answerable_question_accuracy": round(
            sum(row.get("classification") == "CORRECT" for row in answerable)
            / max(len(answerable), 1),
            6,
        ),
        "unanswerable_refusal_accuracy": round(
            sum(row.get("classification") == "CORRECT_REFUSAL" for row in unanswerable)
            / max(len(unanswerable), 1),
            6,
        ),
    }


def category_summary(rows):
    result = {}
    for category in sorted({row.get("question_type") for row in rows}):
        selected = [row for row in rows if row.get("question_type") == category]
        counts = {
            label: sum(row.get("classification") == label for row in selected)
            for label in LABELS
        }
        strict = counts["CORRECT"] + counts["CORRECT_REFUSAL"]
        result[category] = {
            "count": len(selected),
            "correct": counts["CORRECT"],
            "partial": counts["PARTIALLY_CORRECT"],
            "incorrect": counts["INCORRECT"],
            "correct_refusal": counts["CORRECT_REFUSAL"],
            "wrong_refusal": counts["WRONG_REFUSAL"],
            "execution_error": counts["EXECUTION_ERROR"],
            "strict_accuracy": round(strict / max(len(selected), 1), 6),
            "partial_rate": round(
                counts["PARTIALLY_CORRECT"] / max(len(selected), 1), 6
            ),
            "faithfulness": mean(
                [(row.get("correctness_judge") or {}).get("faithfulness") for row in selected]
            ),
            "context_recall": mean(
                [(row.get("correctness_judge") or {}).get("context_recall") for row in selected]
            ),
        }
    return result


def build_failures(rows):
    failures = []
    for row in rows:
        if row.get("classification") in {"CORRECT", "CORRECT_REFUSAL"}:
            continue
        trace = row.get("trace") or {}
        retrieval = trace.get("retrieval_trace") or {}
        documents = trace.get("retrieved_sources") or retrieval.get("documents") or []
        stage = failure_stage(row)
        failures.append(
            {
                "id": row.get("id"),
                "question": row.get("question"),
                "question_type": row.get("question_type"),
                "expected": {
                    "behavior": row.get("expected_behavior"),
                    "ground_truth": row.get("ground_truth"),
                    "reference_documents": row.get("reference_documents"),
                },
                "actual_answer": row.get("answer"),
                "final_classification": row.get("classification"),
                "route": row.get("route"),
                "rewritten_query": trace.get("rewritten_query"),
                "classifier_result": trace.get("classification"),
                "strategy": trace.get("strategy"),
                "retrieved_docs": [item.get("document_name") for item in documents],
                "retrieval_scores": [item.get("milvus_score") for item in documents],
                "reranker_scores": [item.get("reranker_score") for item in documents],
                "final_contexts": trace.get("final_context"),
                "evidence_gate_result": trace.get("evidence_decision"),
                "failure_stage": stage,
                "failure_reason": (
                    row.get("error")
                    or (row.get("correctness_judge") or {}).get("rationale")
                    or "No judge rationale available"
                ),
            }
        )
    return failures


def persist(data):
    rows = data["results"]
    judged = sum(bool(row.get("classification")) for row in rows)
    data["correctness_judged"] = judged
    data["correctness_judge_model"] = "qwen3.7-plus"
    if judged == len(rows):
        data["metrics"] = summarize(rows)
        data["category_metrics"] = category_summary(rows)
        data["status"] = "COMPLETED"
    else:
        data["status"] = "JUDGING_OR_PARTIAL"
    save(RESULTS_PATH, data)


def main():
    cfg = Config()
    if cfg.LLM_MODEL != "qwen3.7-plus":
        raise RuntimeError(f"Unexpected final model: {cfg.LLM_MODEL}")
    data = load(RESULTS_PATH)
    rows = data.get("results", [])
    if data.get("final_llm_model") != cfg.LLM_MODEL or len(rows) != 350:
        raise RuntimeError("Full-RAG execution must contain 350 qwen3.7-plus rows")

    client = OpenAI(
        api_key=cfg.DASHSCOPE_API_KEY,
        base_url=cfg.DASHSCOPE_BASE_URL,
        http_client=httpx.Client(trust_env=False, timeout=90),
        max_retries=1,
    )
    for row in rows:
        if row.get("error") and not row.get("classification"):
            row["classification"] = "EXECUTION_ERROR"
            row["correctness_judge"] = {
                "classification": "EXECUTION_ERROR",
                "rationale": row["error"],
                "judge_model": cfg.LLM_MODEL,
                "judge_usage": "NOT_APPLICABLE",
            }
    persist(data)

    pending = [row for row in rows if not row.get("classification")]
    completed = len(rows) - len(pending)
    fatal_error = None
    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="qwen37-correctness") as executor:
        for start in range(0, len(pending), 3):
            batch = pending[start : start + 3]
            futures = [(row, executor.submit(judge_one, client, cfg, row)) for row in batch]
            for row, future in futures:
                try:
                    judged = future.result()
                    row["correctness_judge"] = judged
                    row["classification"] = judged["classification"]
                    row.pop("judge_error", None)
                except Exception as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    row["judge_error"] = message
                    if is_fatal_account_error(message):
                        fatal_error = message
                    else:
                        row["classification"] = "EXECUTION_ERROR"
                        row["correctness_judge"] = {
                            "classification": "EXECUTION_ERROR",
                            "rationale": message,
                            "judge_model": cfg.LLM_MODEL,
                            "judge_usage": "NOT_AVAILABLE",
                        }
                completed += 1
                persist(data)
                print(
                    f"Correctness {completed}/350 {row.get('id')}: "
                    f"{row.get('classification') or row.get('judge_error')}",
                    flush=True,
                )
            if fatal_error:
                break
    if fatal_error:
        data["status"] = "CHECKPOINTED_ACCOUNT_BLOCKER"
        data["account_blocker"] = fatal_error
        save(RESULTS_PATH, data)
        raise RuntimeError(fatal_error)

    failures = build_failures(rows)
    stage_counts = {stage: sum(row["failure_stage"] == stage for row in failures) for stage in FAILURE_STAGES}
    failure_output = {
        "status": "COMPLETED",
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "model": cfg.LLM_MODEL,
        "failure_count": len(failures),
        "failure_stage_counts": stage_counts,
        "failure_stage_percentages": {
            stage: round(count / max(len(failures), 1), 6)
            for stage, count in stage_counts.items()
        },
        "allowed_stages": list(FAILURE_STAGES),
        "failures": failures,
    }
    save(FAILURES_PATH, failure_output)
    save(
        CATEGORY_PATH,
        {
            "status": "COMPLETED",
            "model": cfg.LLM_MODEL,
            "generation_80": load(ROOT / "RETAILRAG_QWEN37_FINAL_GENERATION_80.json").get("category_metrics"),
            "full_rag_350": category_summary(rows),
        },
    )

    lines = [
        "# RetailRAG qwen3.7-plus Final Failure Audit",
        "",
        f"Total non-strict outcomes: **{len(failures)}**.",
        "",
        "| Stage | Count | Percentage |",
        "|---|---:|---:|",
    ]
    lines.extend(
        f"| {stage} | {stage_counts[stage]} | {stage_counts[stage] / max(len(failures), 1):.2%} |"
        for stage in FAILURE_STAGES
    )
    lines.extend(
        [
            "",
            "| ID | Type | Classification | Stage | Reason |",
            "|---|---|---|---|---|",
        ]
    )
    for row in failures:
        values = [
            row["id"],
            row["question_type"],
            row["final_classification"],
            row["failure_stage"],
            row["failure_reason"],
        ]
        values = [str(value).replace("|", "\\|").replace("\n", " ") for value in values]
        lines.append("| " + " | ".join(values) + " |")
    FAILURES_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    component_names = (
        "faq",
        "query_rewrite",
        "query_classifier",
        "strategy_selector",
        "embedding",
        "milvus",
        "reranker",
        "evidence_gate",
        "llm_first_token",
        "llm_total",
        "full_rag",
    )
    data["latency_ms"] = {
        name: dist(
            [(row.get("component_latency_ms") or {}).get(name) for row in rows]
        )
        for name in component_names
    }
    data["failure_stage_counts"] = stage_counts
    data["evaluated_at"] = datetime.now().isoformat(timespec="seconds")
    persist(data)
    print(json.dumps(data["metrics"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
