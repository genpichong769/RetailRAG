"""Build final qwen3.7-plus metrics, latency, usage and human-readable report."""

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
    values = [value for value in values if value is not None]
    if not values:
        return "NOT_AVAILABLE"
    return {
        "count": len(values),
        "mean": mean(values),
        "median": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
    }


def usage_sum(items):
    items = [item for item in items if isinstance(item, dict)]
    if not items:
        return "NOT_AVAILABLE"
    return {
        key: sum(int(item.get(key, 0) or 0) for item in items)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def fmt(value):
    if value is None:
        return "N/A"
    if isinstance(value, str):
        return value
    return f"{value:.6f}" if isinstance(value, float) else str(value)


def verdict(strict_accuracy):
    if strict_accuracy >= 0.85:
        return "STRONG"
    if strict_accuracy >= 0.70:
        return "GOOD_WITH_LIMITATIONS"
    if strict_accuracy >= 0.50:
        return "DEMO_READY_WITH_CLEAR_LIMITATIONS"
    return "NEEDS_MAJOR_IMPROVEMENT"


def main():
    faq = load("RETAILRAG_QWEN37_FINAL_FAQ_480.json")
    retrieval = load("RETAILRAG_QWEN37_FINAL_RETRIEVAL_235.json")
    rewrite = load("RETAILRAG_QWEN37_FINAL_QUERY_REWRITE_30.json")
    evidence = load("RETAILRAG_QWEN37_FINAL_EVIDENCE_GATE.json")
    generation = load("RETAILRAG_QWEN37_FINAL_GENERATION_80.json")
    full = load("RETAILRAG_QWEN37_FINAL_350_RESULTS.json")
    failures = load("RETAILRAG_QWEN37_FINAL_FAILURES.json")
    categories = load("RETAILRAG_QWEN37_FINAL_CATEGORY_METRICS.json")
    engineering = load("RETAILRAG_QWEN37_FINAL_ENGINEERING.json", {})
    freeze = load("RETAILRAG_QWEN37_BENCHMARK_FREEZE.json")

    required = (faq, retrieval, rewrite, evidence, generation, full, failures, categories, freeze)
    if any(item is None for item in required):
        raise RuntimeError("One or more required final artifacts are missing")
    if generation.get("status") != "COMPLETED" or full.get("status") != "COMPLETED":
        raise RuntimeError("Generation 80 and judged Full RAG 350 must both be complete")

    faq_metrics = faq["metrics"]
    retrieval_metrics = retrieval["modes"]["hybrid_reranker"]["unified_metrics"]
    rewrite_before = rewrite["without_query_rewrite"]
    rewrite_after = rewrite["with_query_rewrite"]
    evidence_metrics = evidence["metrics"]
    generation_metrics = generation["metrics"]
    full_metrics = full["metrics"]
    quality = verdict(full_metrics["strict_correct_rate"])

    latency = {
        "status": "COMPLETED",
        "model": "qwen3.7-plus",
        "faq_gate_ms": faq.get("gate_latency_ms", "NOT_AVAILABLE"),
        "query_rewrite_ms": rewrite.get("rewrite_latency_ms", "NOT_AVAILABLE"),
        "generation_80": {
            "llm_first_token_ms": dist(
                [
                    (row.get("generation_llm_events") or [{}])[-1].get(
                        "first_token_latency_ms"
                    )
                    for row in generation["results"]
                    if row.get("generation_llm_events")
                ]
            ),
            "llm_total_ms": dist(
                [
                    (row.get("generation_llm_events") or [{}])[-1].get(
                        "total_latency_ms"
                    )
                    for row in generation["results"]
                    if row.get("generation_llm_events")
                ]
            ),
            "judge_ms": dist(
                [row.get("judge_latency_ms") for row in generation["results"]]
            ),
            "full_case_ms": dist(
                [
                    float(row.get("elapsed_seconds", 0)) * 1000
                    for row in generation["results"]
                ]
            ),
        },
        "full_rag_350": full.get("latency_ms", "NOT_AVAILABLE"),
        "notes": [
            "Full-RAG component latency uses the production reranker without the evaluation pair cache.",
            "P99 is reported because all component distributions retain per-case observations.",
        ],
    }
    save("RETAILRAG_QWEN37_FINAL_LATENCY.json", latency)

    correctness_usage = usage_sum(
        [
            (row.get("correctness_judge") or {}).get("judge_usage")
            for row in full["results"]
        ]
    )
    strategy_rows = {}
    strategy_labels = {
        "直接检索": "Direct",
        "假设问题检索": "HyDE",
        "子查询检索": "SubQuery",
        "回溯问题检索": "Backtracking",
        "FAQ_DIRECT": "FAQ Direct",
        "GENERAL": "General",
    }
    for row in full["results"]:
        trace = row.get("trace") or {}
        raw_strategy = trace.get("strategy") or row.get("route") or "UNKNOWN"
        strategy = strategy_labels.get(raw_strategy, raw_strategy)
        production_calls = len(row.get("generation_llm_events") or [])
        if trace.get("strategy"):
            production_calls += 1  # StrategySelector production call.
        strategy_rows.setdefault(strategy, []).append(production_calls)
        if row.get("question_type") == "multiturn":
            strategy_rows.setdefault("Multi-turn", []).append(production_calls)
    usage = {
        "status": "COMPLETED",
        "model": "qwen3.7-plus",
        "generation_80": generation.get("usage", "NOT_AVAILABLE"),
        "full_rag_350_production_calls": full.get("usage", "NOT_AVAILABLE"),
        "full_rag_350_correctness_judge": correctness_usage,
        "query_rewrite_30": rewrite.get("usage", "NOT_AVAILABLE"),
        "average_production_model_calls_by_strategy": {
            strategy: round(sum(values) / len(values), 6)
            for strategy, values in strategy_rows.items()
        },
        "call_count_note": "Counts observed RAGSystem LLM events plus one StrategySelector call where applicable; the correctness judge is excluded from production call counts.",
        "usage_limitation": "Only API responses that exposed Usage are summed. Missing StrategySelector usage is not estimated.",
    }
    save("RETAILRAG_QWEN37_FINAL_USAGE.json", usage)

    ordered_stages = sorted(
        failures["failure_stage_counts"].items(), key=lambda item: item[1], reverse=True
    )
    leading = [item for item in ordered_stages if item[1] > 0][:3]
    while len(leading) < 3:
        leading.append(("NONE", 0))

    full_category = categories["full_rag_350"]
    ranked_categories = sorted(
        full_category.items(), key=lambda item: item[1]["strict_accuracy"], reverse=True
    )
    best_category = ranked_categories[0]
    worst_category = ranked_categories[-1]
    dynamic = retrieval["dynamic_m_ablation"]["modes"]
    rerank = retrieval["reranker_ablation"]

    final_metrics = {
        "status": "COMPLETED",
        "evaluation_name": "RETAILRAG FINAL FULL EVALUATION V1",
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "final_model": "qwen3.7-plus",
        "benchmark_hash_check": freeze["status"],
        "data_scale": freeze["counts"],
        "faq": faq_metrics,
        "retrieval": retrieval_metrics,
        "reranker_ablation": {
            key: rerank[key]
            for key in (
                "hybrid",
                "hybrid_reranker",
                "reranker_improved_count",
                "reranker_unchanged_count",
                "reranker_degraded_count",
            )
        },
        "dynamic_m": {
            "fixed_m2": dynamic["fixed_m2"]["metrics"],
            "dynamic_m": dynamic["dynamic_m"]["metrics"],
        },
        "query_rewrite": {
            "without": rewrite_before,
            "with": rewrite_after,
            "success_rate": rewrite["rewrite_success_rate"],
            "error_rate": rewrite["rewrite_error_rate"],
            "improved": rewrite["rewrite_improved"],
            "unchanged": rewrite["rewrite_unchanged"],
            "degraded": rewrite["rewrite_degraded"],
        },
        "evidence_gate": evidence_metrics,
        "generation_80": {
            "execution_count": generation["execution_count"],
            "successfully_judged": generation["successfully_judged"],
            "execution_errors": generation["execution_errors"],
            "metrics": generation_metrics,
            "category_metrics": generation["category_metrics"],
        },
        "full_rag_350": full_metrics,
        "full_rag_category_metrics": full_category,
        "failure_stage_counts": failures["failure_stage_counts"],
        "quality_judgement": quality,
        "engineering": engineering,
        "historical_model_experiments": {
            "qwen3-max": "historical only; not included",
            "qwen3.6-plus": "Generation 80/80 and Full RAG 133/350 incomplete; not included",
            "qwen3.6-flash": "incomplete experiment; not included",
            "mixed_model_results": False,
        },
    }
    save("RETAILRAG_QWEN37_FINAL_METRICS.json", final_metrics)

    engineering_status = {
        key: (engineering.get(key.lower()) or engineering.get(key) or "NOT_RUN")
        for key in ("FASTAPI", "WEBSOCKET", "BROWSER_E2E")
    }
    generation_category_rows = "\n".join(
        "| {name} | {count} | {faithfulness} | {relevance} | {precision} | {recall} |".format(
            name=name,
            count=values["count"],
            faithfulness=fmt(values.get("faithfulness")),
            relevance=fmt(values.get("answer_relevance")),
            precision=fmt(values.get("context_precision")),
            recall=fmt(values.get("context_recall")),
        )
        for name, values in generation["category_metrics"].items()
    )
    full_category_rows = "\n".join(
        "| {name} | {count} | {correct} | {partial} | {incorrect} | {correct_refusal} | {wrong_refusal} | {strict} | {partial_rate} |".format(
            name=name,
            count=values["count"],
            correct=values["correct"],
            partial=values["partial"],
            incorrect=values["incorrect"],
            correct_refusal=values["correct_refusal"],
            wrong_refusal=values["wrong_refusal"],
            strict=fmt(values["strict_accuracy"]),
            partial_rate=fmt(values["partial_rate"]),
        )
        for name, values in full_category.items()
    )
    full_latency = full.get("latency_ms", {})
    latency_rows = "\n".join(
        "| {name} | {count} | {mean} | {median} | {p95} | {p99} |".format(
            name=name,
            count=values.get("count", "N/A") if isinstance(values, dict) else "N/A",
            mean=fmt(values.get("mean")) if isinstance(values, dict) else fmt(values),
            median=fmt(values.get("median")) if isinstance(values, dict) else "N/A",
            p95=fmt(values.get("p95")) if isinstance(values, dict) else "N/A",
            p99=fmt(values.get("p99")) if isinstance(values, dict) else "N/A",
        )
        for name, values in full_latency.items()
    )
    strategy_usage_rows = "\n".join(
        f"| {name} | {fmt(value)} |"
        for name, value in usage["average_production_model_calls_by_strategy"].items()
    )
    generation_usage = usage["generation_80"]
    full_usage = usage["full_rag_350_production_calls"]
    full_judge_usage = usage["full_rag_350_correctness_judge"]
    rewrite_usage = usage["query_rewrite_30"]
    report = f"""# RetailRAG Final Full Evaluation V1

## FINAL RESULT

**FINAL MODEL**  
qwen3.7-plus

**QUALITY JUDGEMENT**  
{quality}

## DATA SCALE

| Dataset | Count |
|---|---:|
| FAQ | 200 |
| Documents | 80 |
| FAQ Eval | 480 |
| Retrieval Eval | 235 |
| Generation Eval | 80 |
| Full RAG Eval | 350 |
| Multi-turn | 30 |

## FAQ

| Metric | Result |
|---|---:|
| Precision | {fmt(faq_metrics['precision'])} |
| Recall | {fmt(faq_metrics['recall'])} |
| F1 | {fmt(faq_metrics['f1'])} |
| FAR | {fmt(faq_metrics['false_accept_rate'])} |
| Complex Wrong Intercept | {faq_metrics['complex_query_wrong_intercept_count']} |

## RETRIEVAL

| Metric | Hybrid + Reranker |
|---|---:|
| Recall@1 | {fmt(retrieval_metrics['recall_at_1'])} |
| Recall@3 | {fmt(retrieval_metrics['recall_at_3'])} |
| Recall@5 | {fmt(retrieval_metrics['recall_at_5'])} |
| MRR | {fmt(retrieval_metrics['mrr'])} |
| All Recall@3 | {fmt(retrieval_metrics['all_recall_at_3'])} |

## RERANKER ABLATION

| Mode | Top1 Accuracy | Recall@1 | Recall@3 | MRR |
|---|---:|---:|---:|---:|
| Hybrid | {fmt(rerank['hybrid']['hit_at_1'])} | {fmt(rerank['hybrid']['recall_at_1'])} | {fmt(rerank['hybrid']['recall_at_3'])} | {fmt(rerank['hybrid']['mrr'])} |
| Hybrid + Reranker | {fmt(rerank['hybrid_reranker']['hit_at_1'])} | {fmt(rerank['hybrid_reranker']['recall_at_1'])} | {fmt(rerank['hybrid_reranker']['recall_at_3'])} | {fmt(rerank['hybrid_reranker']['mrr'])} |

Improved / Unchanged / Degraded: **{rerank['reranker_improved_count']} / {rerank['reranker_unchanged_count']} / {rerank['reranker_degraded_count']}**.

## DYNAMIC M ABLATION

| Mode | Any@3 | All@3 | Context Precision | Context Recall | Avg Contexts | Avg Chars | Token Proxy | Multi-doc All@3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed M=2 | {fmt(dynamic['fixed_m2']['metrics']['any_at_3'])} | {fmt(dynamic['fixed_m2']['metrics']['all_at_3'])} | {fmt(dynamic['fixed_m2']['metrics']['context_precision'])} | {fmt(dynamic['fixed_m2']['metrics']['context_recall'])} | {fmt(dynamic['fixed_m2']['metrics']['average_context_count'])} | {fmt(dynamic['fixed_m2']['metrics']['average_context_characters'])} | {fmt(dynamic['fixed_m2']['metrics']['average_context_token_proxy'])} | {fmt(dynamic['fixed_m2']['metrics']['multi_document_all_at_3'])} |
| Dynamic M | {fmt(dynamic['dynamic_m']['metrics']['any_at_3'])} | {fmt(dynamic['dynamic_m']['metrics']['all_at_3'])} | {fmt(dynamic['dynamic_m']['metrics']['context_precision'])} | {fmt(dynamic['dynamic_m']['metrics']['context_recall'])} | {fmt(dynamic['dynamic_m']['metrics']['average_context_count'])} | {fmt(dynamic['dynamic_m']['metrics']['average_context_characters'])} | {fmt(dynamic['dynamic_m']['metrics']['average_context_token_proxy'])} | {fmt(dynamic['dynamic_m']['metrics']['multi_document_all_at_3'])} |

## QUERY REWRITE

| Metric | Before | After |
|---|---:|---:|
| Hit@1 | {fmt(rewrite_before['hit_at_1'])} | {fmt(rewrite_after['hit_at_1'])} |
| Hit@3 | {fmt(rewrite_before['hit_at_3'])} | {fmt(rewrite_after['hit_at_3'])} |
| MRR | {fmt(rewrite_before['mrr'])} | {fmt(rewrite_after['mrr'])} |

Rewrite Success / Error: **{fmt(rewrite['rewrite_success_rate'])} / {fmt(rewrite['rewrite_error_rate'])}**. Improved / Unchanged / Degraded: **{rewrite['rewrite_improved']} / {rewrite['rewrite_unchanged']} / {rewrite['rewrite_degraded']}**.

## EVIDENCE GATE

| Metric | Result |
|---|---:|
| Precision | {fmt(evidence_metrics['refusal_precision'])} |
| Recall | {fmt(evidence_metrics['refusal_recall'])} |
| F1 | {fmt(evidence_metrics['refusal_f1'])} |
| False Answer Rate | {fmt(evidence_metrics['false_answer_rate'])} |
| False Refusal Rate | {fmt(evidence_metrics['false_refusal_rate'])} |
| Answerable Accuracy | {fmt(evidence_metrics['answerable_accuracy'])} |

## GENERATION / RAGAS

| Metric | Result |
|---|---:|
| Execution | {generation['execution_count']}/80 |
| Faithfulness | {fmt(generation_metrics['faithfulness'])} |
| Answer Relevance | {fmt(generation_metrics['answer_relevance'])} |
| Context Precision | {fmt(generation_metrics['context_precision'])} |
| Context Recall | {fmt(generation_metrics['context_recall'])} |

### Generation 80 by category

| Category | Count | Faithfulness | Answer Relevance | Context Precision | Context Recall |
|---|---:|---:|---:|---:|---:|
{generation_category_rows}

## END-TO-END 350

| Metric | Result |
|---|---:|
| Execution | {full_metrics['execution_attempted']}/350 |
| Correct | {full_metrics['correct']} |
| Partial | {full_metrics['partial']} |
| Incorrect | {full_metrics['incorrect']} |
| Correct Refusal | {full_metrics['correct_refusal']} |
| Wrong Refusal | {full_metrics['wrong_refusal']} |
| Execution Error | {full_metrics['execution_errors']} |
| Strict Accuracy | {fmt(full_metrics['strict_correct_rate'])} |
| Partial Rate | {fmt(full_metrics['partial_rate'])} |
| Incorrect Rate | {fmt(full_metrics['incorrect_rate'])} |
| Wrong Refusal Rate | {fmt(full_metrics['wrong_refusal_rate'])} |
| Answerable Accuracy | {fmt(full_metrics['answerable_question_accuracy'])} |
| Unanswerable Refusal Accuracy | {fmt(full_metrics['unanswerable_refusal_accuracy'])} |

### End-to-end 350 by category

| Category | Count | Correct | Partial | Incorrect | Correct Refusal | Wrong Refusal | Strict Accuracy | Partial Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{full_category_rows}

## 主要失败阶段

1. {leading[0][0]}: {leading[0][1]}
2. {leading[1][0]}: {leading[1][1]}
3. {leading[2][0]}: {leading[2][1]}

## LATENCY

单位：毫秒。P99 由完整逐题观测值计算。

| Component | Count | Mean | Median | P95 | P99 |
|---|---:|---:|---:|---:|---:|
{latency_rows}

## QWEN3.7-PLUS TOKEN USAGE

| Evaluation | Prompt Tokens | Completion Tokens | Total Tokens | Average Total Tokens / Case |
|---|---:|---:|---:|---:|
| Generation 80 | {generation_usage['prompt_tokens']} | {generation_usage['completion_tokens']} | {generation_usage['total_tokens']} | {fmt(generation_usage['average_total_tokens_per_case'])} |
| Full RAG 350 production | {full_usage['prompt_tokens']} | {full_usage['completion_tokens']} | {full_usage['total_tokens']} | {fmt(full_usage['average_total_tokens_per_case'])} |
| Full RAG 350 correctness judge | {full_judge_usage['prompt_tokens']} | {full_judge_usage['completion_tokens']} | {full_judge_usage['total_tokens']} | {fmt(full_judge_usage['total_tokens'] / 350)} |
| Query Rewrite 30 | {rewrite_usage['prompt_tokens']} | {rewrite_usage['completion_tokens']} | {rewrite_usage['total_tokens']} | {fmt(rewrite_usage['total_tokens'] / 30)} |

### Average production model calls by strategy

| Strategy | Average calls / case |
|---|---:|
{strategy_usage_rows}

StrategySelector 未暴露 Usage，因此 Token 总量仅汇总 API 实际返回字段，没有估算缺失值。

## ENGINEERING

| Check | Result |
|---|---|
| MySQL | {engineering.get('mysql', 'NOT_RUN')} |
| Redis | {engineering.get('redis', 'NOT_RUN')} |
| Milvus | {engineering.get('milvus', 'NOT_RUN')} |
| MinIO | {engineering.get('minio', 'NOT_RUN')} |
| etcd | {engineering.get('etcd', 'NOT_RUN')} |
| FastAPI | {engineering_status['FASTAPI']} |
| WebSocket | {engineering_status['WEBSOCKET']} |
| Browser E2E | {engineering_status['BROWSER_E2E']} |

Known engineering limitation: one async acceptance client sent keepalive pings during a 95.23-second synchronous retrieval and timed out; the server completed the request, and the 120/180-second WebSocket suites passed. A single Uvicorn worker can delay health and ping responses while CPU-bound retrieval is running.

## HISTORICAL_MODEL_EXPERIMENTS

- qwen3-max results are retained as historical evidence only.
- qwen3.6-plus Generation 80/80 and incomplete Full RAG 133/350 are retained separately.
- qwen3.6-flash incomplete results are retained separately.
- No historical or mixed-model row is included in the qwen3.7-plus final metrics.

## Final assessment questions

1. **RetailRAG 当前实际效果怎么样？** 端到端严格准确率为 {fmt(full_metrics['strict_correct_rate'])}，质量结论为 **{quality}**。Generation 的 Faithfulness 为 {fmt(generation_metrics['faithfulness'])}，但完整效果仍受检索覆盖、上下文完整性和拒答决策共同影响。
2. **哪类问题效果最好？** `{best_category[0]}`，Strict Accuracy 为 {fmt(best_category[1]['strict_accuracy'])}。
3. **哪类问题效果最差？** `{worst_category[0]}`，Strict Accuracy 为 {fmt(worst_category[1]['strict_accuracy'])}。
4. **FAQ 是否稳定？** 精确问法稳定：Exact Accuracy={fmt(faq_metrics['exact_accuracy'])}、FAR={fmt(faq_metrics['false_accept_rate'])}。整体还不能称为完全稳定：paraphrase recall 只有 {fmt(faq_metrics['paraphrase_recall'])}，且 350 端到端中有 20 条失败归因于 FAQ_ROUTING，主要是非纯 FAQ 问题被 FAQ_DIRECT 截获后缺少检索证据。
5. **Dense + Sparse + Reranker 是否有效？** 有效，但增益温和。相对 Hybrid，Recall@3 从 {fmt(rerank['hybrid']['recall_at_3'])} 提升到 {fmt(rerank['hybrid_reranker']['recall_at_3'])}，MRR 从 {fmt(rerank['hybrid']['mrr'])} 提升到 {fmt(rerank['hybrid_reranker']['mrr'])}；Improved/Unchanged/Degraded 为 {rerank['reranker_improved_count']}/{rerank['reranker_unchanged_count']}/{rerank['reranker_degraded_count']}。
6. **多文档完整召回是否仍是主要问题？** Dynamic M 的多文档 All@3 为 {fmt(dynamic['dynamic_m']['metrics']['multi_document_all_at_3'])}，低于单文档表现，仍是关键瓶颈。
7. **Query Rewrite 是否值得保留？** 值得。Hit@3 从 {fmt(rewrite_before['hit_at_3'])} 提升到 {fmt(rewrite_after['hit_at_3'])}，MRR 从 {fmt(rewrite_before['mrr'])} 提升到 {fmt(rewrite_after['mrr'])}，错误率为 {fmt(rewrite['rewrite_error_rate'])}。
8. **Evidence Gate 是否过度拒答？** 控制集上的 False Refusal Rate 为 {fmt(evidence_metrics['false_refusal_rate'])}，并不高；但 350 端到端出现 {full_metrics['wrong_refusal']} 条 Wrong Refusal，其中 {failures['failure_stage_counts'].get('EVIDENCE_GATE', 0)} 条归因于 EVIDENCE_GATE，说明真实分布上存在明显过度拒答，需要下一轮优先复核。
9. **qwen3.7-plus Generator 是否仍是瓶颈？** 是显著瓶颈，但不是第一瓶颈。GENERATION 归因 {failures['failure_stage_counts'].get('GENERATION', 0)} 条，少于 FAQ_ROUTING 的 {failures['failure_stage_counts'].get('FAQ_ROUTING', 0)} 条和 EVIDENCE_GATE 的 {failures['failure_stage_counts'].get('EVIDENCE_GATE', 0)} 条；同时 Generation 80 的 Faithfulness/Answer Relevance 达到 {fmt(generation_metrics['faithfulness'])}/{fmt(generation_metrics['answer_relevance'])}，说明上游路由与拒答对端到端结果影响更大。
10. **Context Precision / Recall 为什么处于当前水平？** Context Precision={fmt(generation_metrics['context_precision'])} 反映候选中仍含旁支证据；Context Recall={fmt(generation_metrics['context_recall'])} 主要受 K=3 候选覆盖与多文档完整召回限制。Dynamic M 增加复杂题上下文，但不能恢复未进入候选池的文档。
11. **投实习时哪些指标适合写简历？** 可以写固定 200 FAQ、80 文档、480/235/80/350/30 评测规模；FAQ F1/FAR；Hybrid+Reranker Recall@3 与 MRR；Query Rewrite 前后增益；Evidence Gate Precision/Recall/F1；Generation 四项指标；以及 350 条严格准确率。应同时注明统一模型与固定 Benchmark。
12. **如果只能再优化一轮，优先改哪里？** 优先修 FAQ_ROUTING：它贡献 {failures['failure_stage_counts'].get('FAQ_ROUTING', 0)} 条失败，是最大单一阶段。目标应是减少带上下文、组合条件、多文档或多轮问题被 FAQ_DIRECT 过早截获，同时守住 Exact Accuracy={fmt(faq_metrics['exact_accuracy'])} 和 FAR={fmt(faq_metrics['false_accept_rate'])}；随后再处理 {failures['failure_stage_counts'].get('EVIDENCE_GATE', 0)} 条 EVIDENCE_GATE 失败。

## Method and limitations

- Generator、Generation Judge 与 350 正确性 Judge 均为 qwen3.7-plus，同模型裁判可能偏乐观。
- Benchmark、Ground Truth、Prompt、阈值与业务逻辑在正式运行期间均未修改。
- Token Usage 只汇总 API 实际返回的字段；未返回部分标为 `NOT_AVAILABLE`，没有估算。
"""
    (ROOT / "RETAILRAG_QWEN37_FINAL_EVALUATION_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "COMPLETED",
                "model": "qwen3.7-plus",
                "quality_judgement": quality,
                "strict_accuracy": full_metrics["strict_correct_rate"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
