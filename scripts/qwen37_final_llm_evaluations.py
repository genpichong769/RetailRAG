"""Run the qwen3.7-plus Generation-80 and Full-RAG-350 final evaluations.

The frozen datasets and production prompts are read without modification. Results
are checkpointed to new PHASE 6 files so historical qwen3-max artifacts remain
intact.
"""

import argparse
import copy
import hashlib
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from base import Config
from evaluate_scale import CachedReranker, llm_client, mean
from new_main import IntegratedQASystem
from rag_qa.core.new_rag_system import RAGSystem
from rag_qa.core.strategy_selector import StrategySelector
from rag_qa.core.vector_store import VectorStore


GENERATION_OUT = ROOT / "RETAILRAG_QWEN37_FINAL_GENERATION_80.json"
FULL_OUT = ROOT / "RETAILRAG_QWEN37_FINAL_350_RESULTS.json"


def load(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def assert_frozen_inputs():
    audit = load("RETAILRAG_SCALE_DATA_AUDIT.json")
    mismatches = []
    for relative, expected in audit["sha256"].items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if actual.lower() != expected.lower():
            mismatches.append(relative)
    if mismatches:
        raise RuntimeError(f"Frozen inputs changed: {mismatches}")
    return {relative: value.lower() for relative, value in audit["sha256"].items()}


def ensure_selected_model(cfg):
    smoke = load("RETAILRAG_QWEN37_MODULE_SMOKE.json")
    if smoke.get("status") != "PASS" or smoke.get("selected_model") != cfg.LLM_MODEL:
        raise RuntimeError("Configured model does not match the passed migration smoke test")
    if cfg.LLM_MODEL != "qwen3.7-plus":
        raise RuntimeError(f"Unexpected final model: {cfg.LLM_MODEL}")


class TrackedStream:
    """Capture usage and first-token timing for RAGSystem streaming calls."""

    def __init__(self, client, cfg):
        self.client = client
        self.cfg = cfg
        self.local = threading.local()

    def begin(self):
        self.local.events = []

    def events(self):
        return copy.deepcopy(getattr(self.local, "events", []))

    def __call__(self, prompt):
        started = time.perf_counter()
        first_token_ms = None
        usage = None
        usage_supported = True
        try:
            response = self.client.chat.completions.create(
                model=self.cfg.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是 Nova Retail 智能知识助手。"},
                    {"role": "user", "content": str(prompt)},
                ],
                temperature=0.1,
                timeout=90,
                stream=True,
                stream_options={"include_usage": True},
            )
        except Exception as exc:
            # Some OpenAI-compatible endpoints do not expose stream usage.
            # Retry the same prompt and parameters without the optional field.
            if "stream_options" not in str(exc).lower() and "usage" not in str(exc).lower():
                raise
            usage_supported = False
            response = self.client.chat.completions.create(
                model=self.cfg.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是 Nova Retail 智能知识助手。"},
                    {"role": "user", "content": str(prompt)},
                ],
                temperature=0.1,
                timeout=90,
                stream=True,
            )
        try:
            for chunk in response:
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage is not None:
                    usage = chunk_usage.model_dump()
                if chunk.choices and chunk.choices[0].delta.content:
                    if first_token_ms is None:
                        first_token_ms = round(
                            (time.perf_counter() - started) * 1000, 3
                        )
                    yield chunk.choices[0].delta.content
        finally:
            event = {
                "first_token_latency_ms": first_token_ms,
                "total_latency_ms": round((time.perf_counter() - started) * 1000, 3),
                "usage": usage if usage is not None else "NOT_AVAILABLE",
                "stream_usage_supported": usage_supported and usage is not None,
            }
            if not hasattr(self.local, "events"):
                self.local.events = []
            self.local.events.append(event)


def judge_json_with_meta(client, cfg, prompt):
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=cfg.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        timeout=90,
    )
    text = response.choices[0].message.content
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("Judge did not return a JSON object")
    return {
        "scores": json.loads(match.group(0)),
        "usage": response.usage.model_dump() if response.usage else "NOT_AVAILABLE",
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
    }


def usage_summary(rows):
    usages = []
    for row in rows:
        for event in row.get("generation_llm_events", []):
            if isinstance(event.get("usage"), dict):
                usages.append(event["usage"])
        if isinstance(row.get("judge_usage"), dict):
            usages.append(row["judge_usage"])
    if not usages:
        return "NOT_AVAILABLE"
    totals = {
        key: sum(int(item.get(key, 0) or 0) for item in usages)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    totals.update(
        {
            "reported_call_count": len(usages),
            "average_total_tokens_per_case": round(
                totals["total_tokens"] / max(len(rows), 1), 6
            ),
            "coverage_note": "Includes RAGSystem streaming calls and Generation judge calls; StrategySelector uses its production client and does not expose usage.",
        }
    )
    return totals


def is_provider_fallback(answer):
    return any(
        marker in (answer or "")
        for marker in (
            "语言模型暂时不可用",
            "当前无法生成回答",
            "未配置语言模型 API Key",
        )
    )


def is_account_blocker(error):
    lowered = str(error or "").lower()
    return any(
        marker in lowered
        for marker in (
            "insufficient balance",
            "arrearage",
            "payment required",
            "account suspended",
        )
    )


def aggregate_generation(rows):
    valid = [row for row in rows if row.get("scores")]
    names = ("faithfulness", "answer_relevance", "context_precision", "context_recall")
    metrics = {
        name: round(mean([float(row["scores"][name]) for row in valid]), 6)
        for name in names
    } if valid else {}
    negative = [row for row in valid if row["expected_behavior"] == "reject"]
    partial = [row for row in valid if row["expected_behavior"] == "partial_answer"]
    result = {
        "execution_count": len(rows),
        "successfully_judged": len(valid),
        "execution_errors": sum(bool(row.get("error")) for row in rows),
        "metrics": metrics,
        "unanswerable_refusal_accuracy": round(
            mean(
                [
                    int(
                        row["scores"].get("refused", False)
                        and not row["scores"].get("unsupported_specifics", False)
                    )
                    for row in negative
                ]
            ),
            6,
        ) if negative else None,
        "partial_answer_accuracy": round(
            mean([int(row["scores"].get("partial_handled", False)) for row in partial]),
            6,
        ) if partial else None,
    }
    category_metrics = {}
    for category in sorted({row.get("question_type") for row in rows}):
        category_rows = [
            row for row in valid if row.get("question_type") == category
        ]
        category_metrics[category] = {
            "count": len([row for row in rows if row.get("question_type") == category]),
            **{
                name: round(
                    mean([float(row["scores"][name]) for row in category_rows]), 6
                )
                if category_rows
                else None
                for name in names
            },
        }
    result["category_metrics"] = category_metrics
    result["usage"] = usage_summary(rows)
    return result


def generation_judge_prompt(sample, context, answer):
    return f'''你是严格的RAG评测裁判。仅按给出的原始证据与参考判断，不利用外部知识替缺失证据补全。每项0到1。
faithfulness：回答事实由实际检索上下文支持的比例；不得因内容听起来合理而给满分。
answer_relevance：是否回应用户全部可答条件；遗漏、错误拒答、回避需要扣分。
context_precision：实际上下文与所问直接相关的比例；不能因为有一条相关就给全部满分。
context_recall：参考答案所需事实在实际上下文中的覆盖比例；不能用回答反推上下文已包含。
纯无答案且正确拒答时faithfulness与relevance可为1；检索到无关上下文应降低precision。
partial_answer必须回答有证据部分，并明确不能确认未知部分；把整个问题都拒答不算完整。
另外判断refused（是否明确拒绝回答缺乏证据部分），unsupported_specifics（是否编造未知具体事实），partial_handled（仅用于部分可回答）。
返回JSON：{{"faithfulness":0.0,"answer_relevance":0.0,"context_precision":0.0,"context_recall":0.0,"refused":false,"unsupported_specifics":false,"partial_handled":false,"rationale":"具体说明缺失或错误"}}
历史：{json.dumps(sample.get('history', []), ensure_ascii=False)}
问题：{sample['question']}
期望行为：{sample['expected_behavior']}
参考：{sample['ground_truth']}
实际检索上下文：{context or '无上下文'}
实际回答：{answer}'''


def run_generation():
    frozen_hashes = assert_frozen_inputs()
    cfg = Config()
    ensure_selected_model(cfg)
    samples = load("RETAILRAG_SCALE_GENERATION_SELECTION.json")["samples"]
    if len(samples) != 80:
        raise RuntimeError("Generation selection is not the frozen 80-case set")

    existing = load(GENERATION_OUT.name) if GENERATION_OUT.exists() else {}
    if existing and existing.get("generator_model") != cfg.LLM_MODEL:
        raise RuntimeError("Existing PHASE 6 generation checkpoint uses a different model")
    by_id = {row["id"]: row for row in existing.get("results", [])}

    store = VectorStore()
    store.reranker = CachedReranker(store.reranker, cfg)
    client = llm_client(cfg)
    tracker = TrackedStream(client, cfg)
    rag = RAGSystem(store, tracker)

    def persist_row(row):
        by_id[row["id"]] = row
        ordered = [by_id[item["id"]] for item in samples if item["id"] in by_id]
        summary = aggregate_generation(ordered)
        output = {
            "status": (
                "COMPLETED"
                if summary["execution_count"] == 80
                and summary["successfully_judged"] == 80
                and summary["execution_errors"] == 0
                else "RUNNING_OR_PARTIAL"
            ),
            "generator_model": cfg.LLM_MODEL,
            "judge_model": cfg.LLM_MODEL,
            "same_family_optimism_warning": True,
            "historical_qwen3_max_results_preserved": True,
            "frozen_input_sha256": frozen_hashes,
            **summary,
            "results": ordered,
        }
        save(GENERATION_OUT, output)

    def finalize_judge(index, sample, row, future):
        try:
            judged = future.result()
            row["scores"] = judged["scores"]
            row["judge_usage"] = judged["usage"]
            row["judge_latency_ms"] = judged["latency_ms"]
            for key in ("faithfulness", "answer_relevance", "context_precision", "context_recall"):
                row["scores"][key] = max(0.0, min(1.0, float(row["scores"][key])))
            row.pop("error", None)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        row["elapsed_seconds"] = round(time.perf_counter() - row.pop("_started"), 3)
        persist_row(row)
        print(
            f"Generation {index}/80 {sample['id']}: "
            f"{'PASS' if row.get('scores') and not row.get('error') else row.get('error')}",
            flush=True,
        )

    pending = []
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="phase6-judge") as executor:
        for index, sample in enumerate(samples, 1):
            previous = by_id.get(sample["id"])
            if previous and previous.get("scores") and not previous.get("error"):
                print(f"Generation {index}/80 {sample['id']}: checkpoint", flush=True)
                continue
            row = dict(sample)
            row["attempts"] = int((previous or {}).get("attempts", 0)) + 1
            row["_started"] = time.perf_counter()
            try:
                tracker.begin()
                answer = "".join(
                    rag.generate_answer(sample["question"], history=sample.get("history", []))
                )
                trace = copy.deepcopy(rag.last_generation_trace or {})
                if is_provider_fallback(answer):
                    raise RuntimeError("Generator returned the configured provider-error fallback")
                row["answer"] = answer
                row["trace"] = trace
                row["generation_llm_events"] = tracker.events()
                row["generator_model"] = cfg.LLM_MODEL
                row["judge_model"] = cfg.LLM_MODEL
                future = executor.submit(
                    judge_json_with_meta,
                    client,
                    cfg,
                    generation_judge_prompt(sample, trace.get("final_context", ""), answer),
                )
                pending.append((index, sample, row, future))
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                row["elapsed_seconds"] = round(time.perf_counter() - row.pop("_started"), 3)
                row["generator_model"] = cfg.LLM_MODEL
                row["judge_model"] = cfg.LLM_MODEL
                persist_row(row)
                print(f"Generation {index}/80 {sample['id']}: {row['error']}", flush=True)
            if len(pending) >= 2:
                finalize_judge(*pending.pop(0))
        for item in pending:
            finalize_judge(*item)
    final = load(GENERATION_OUT.name)
    print(json.dumps({key: final[key] for key in ("status", "generator_model", "judge_model", "execution_count", "successfully_judged", "execution_errors", "metrics")}, ensure_ascii=False, indent=2))
    if final["status"] != "COMPLETED":
        raise RuntimeError("Generation 80 did not complete without errors")


def aggregate_full(rows):
    route_counts = {"FAQ_DIRECT": 0, "RAG": 0, "GENERAL": 0}
    for row in rows:
        if row.get("route") in route_counts:
            route_counts[row["route"]] += 1
    return {
        "execution_count": len(rows),
        "execution_successes": sum(not row.get("error") for row in rows),
        "execution_errors": sum(bool(row.get("error")) for row in rows),
        "refusal_count": sum(bool(row.get("refused")) for row in rows),
        "faq_direct_count": route_counts["FAQ_DIRECT"],
        "rag_count": route_counts["RAG"],
        "general_count": route_counts["GENERAL"],
        "usage": usage_summary(rows),
    }


def run_full():
    frozen_hashes = assert_frozen_inputs()
    cfg = Config()
    ensure_selected_model(cfg)
    samples = load("RETAILRAG_RAG_EVALUATION_SET.json")["samples"]
    if len(samples) != 350:
        raise RuntimeError("RAG evaluation is not the frozen 350-case set")

    existing = load(FULL_OUT.name) if FULL_OUT.exists() else {}
    if existing and existing.get("final_llm_model") != cfg.LLM_MODEL:
        raise RuntimeError("Existing PHASE 6 full-RAG checkpoint uses a different model")
    by_id = {row["id"]: row for row in existing.get("results", [])}

    qa = IntegratedQASystem()
    # The official full run uses the production reranker directly so recorded
    # latency is not reduced by the evaluation-only pair cache.
    faq_lock = threading.Lock()
    classifier_lock = threading.Lock()
    retrieval_lock = threading.Lock()
    local = threading.local()

    class LockedClassifier:
        def __init__(self, classifier):
            self.classifier = classifier
            self.local = threading.local()

        def predict_details(self, query):
            started = time.perf_counter()
            with classifier_lock:
                result = self.classifier.predict_details(query)
            self.local.last_latency_ms = round(
                (time.perf_counter() - started) * 1000, 3
            )
            return result

        def last_latency_ms(self):
            return getattr(self.local, "last_latency_ms", None)

    class LockedVectorStore:
        def __init__(self, store):
            self.store = store
            self.local = threading.local()

        def begin(self):
            self.local.events = []

        def hybrid_search_with_rerank(self, *args, **kwargs):
            # Model inference and Milvus access are short relative to provider
            # calls; serialize them to keep the shared local models deterministic.
            with retrieval_lock:
                result = self.store.hybrid_search_with_rerank(*args, **kwargs)
                event = copy.deepcopy(
                    ((self.store.last_search_trace or {}).get("timings") or {})
                )
            if not hasattr(self.local, "events"):
                self.local.events = []
            self.local.events.append(event)
            return result

        def events(self):
            return copy.deepcopy(getattr(self.local, "events", []))

        def __getattr__(self, name):
            return getattr(self.store, name)

    locked_classifier = LockedClassifier(qa.rag_system.query_classifier)
    locked_store = LockedVectorStore(qa.vector_store)

    class TimedStrategySelector:
        def __init__(self):
            self.selector = StrategySelector()
            self.last_latency_ms = None

        def select_strategy(self, query):
            started = time.perf_counter()
            result = self.selector.select_strategy(query)
            self.last_latency_ms = round(
                (time.perf_counter() - started) * 1000, 3
            )
            return result

        def __getattr__(self, name):
            return getattr(self.selector, name)

    def thread_rag():
        if not hasattr(local, "rag"):
            rag = copy.copy(qa.rag_system)
            rag.vector_store = locked_store
            rag.query_classifier = locked_classifier
            rag.strategy_selector = TimedStrategySelector()
            tracker = TrackedStream(llm_client(cfg), cfg)
            rag.llm = tracker
            rag._usage_tracker = tracker
            original_rewrite = rag.rewrite_query
            original_evidence = rag._evidence_decision

            def timed_rewrite(query, history):
                started = time.perf_counter()
                result = original_rewrite(query, history)
                rag._last_rewrite_latency_ms = round(
                    (time.perf_counter() - started) * 1000, 3
                )
                return result

            def timed_evidence(query, documents):
                started = time.perf_counter()
                result = original_evidence(query, documents)
                rag._last_evidence_latency_ms = round(
                    (time.perf_counter() - started) * 1000, 3
                )
                return result

            rag.rewrite_query = timed_rewrite
            rag._evidence_decision = timed_evidence
            rag.last_retrieval_trace = None
            rag.last_generation_trace = None
            rag._active_retrieval_queries = []
            local.rag = rag
        return local.rag

    def execute(index, sample, previous):
        row = dict(sample)
        row["attempts"] = int((previous or {}).get("attempts", 0)) + 1
        started = time.perf_counter()
        try:
            faq_started = time.perf_counter()
            with faq_lock:
                faq = qa.bm25_search.search_with_details(
                    sample["question"], threshold=qa.config.FAQ_MATCH_THRESHOLD
                )
            faq_latency_ms = round((time.perf_counter() - faq_started) * 1000, 3)
            row["faq_routing"] = {
                key: faq.get(key)
                for key in (
                    "top1_question",
                    "score",
                    "threshold",
                    "above_threshold",
                    "hit",
                    "cached",
                    "gate_blocked",
                    "gate_reasons",
                    "algorithm_version",
                )
            }
            if faq.get("hit"):
                answer = faq.get("answer") or ""
                row["route"] = "FAQ_DIRECT"
                row["trace"] = None
                row["refused"] = False
                row["component_latency_ms"] = {
                    "faq": faq_latency_ms,
                    "query_rewrite": None,
                    "query_classifier": None,
                    "strategy_selector": None,
                    "embedding": None,
                    "milvus": None,
                    "reranker": None,
                    "evidence_gate": None,
                    "llm_first_token": None,
                    "llm_total": None,
                }
            else:
                rag = thread_rag()
                rag._usage_tracker.begin()
                locked_store.begin()
                answer = "".join(
                    rag.generate_answer(
                        sample["question"], history=sample.get("history", [])
                    )
                )
                trace = copy.deepcopy(rag.last_generation_trace or {})
                row["generation_llm_events"] = rag._usage_tracker.events()
                classification = (trace.get("classification") or {}).get(
                    "routed_prediction"
                )
                row["route"] = "GENERAL" if classification == "general" else "RAG"
                row["trace"] = trace
                evidence = trace.get("evidence_decision") or {}
                row["refused"] = evidence.get("mode") == "refuse"
                search_events = locked_store.events()
                final_llm_event = (
                    row["generation_llm_events"][-1]
                    if trace.get("llm_called") and row["generation_llm_events"]
                    else {}
                )
                row["component_latency_ms"] = {
                    "faq": faq_latency_ms,
                    "query_rewrite": getattr(
                        rag, "_last_rewrite_latency_ms", None
                    ),
                    "query_classifier": locked_classifier.last_latency_ms(),
                    "strategy_selector": rag.strategy_selector.last_latency_ms,
                    "embedding": round(
                        sum(float(item.get("embedding_ms", 0) or 0) for item in search_events),
                        3,
                    ) if search_events else None,
                    "milvus": round(
                        sum(float(item.get("retrieval_ms", 0) or 0) for item in search_events),
                        3,
                    ) if search_events else None,
                    "reranker": round(
                        sum(float(item.get("rerank_ms", 0) or 0) for item in search_events),
                        3,
                    ) if search_events else None,
                    "evidence_gate": getattr(
                        rag, "_last_evidence_latency_ms", None
                    ),
                    "llm_first_token": final_llm_event.get(
                        "first_token_latency_ms"
                    ),
                    "llm_total": final_llm_event.get("total_latency_ms"),
                }
                if trace.get("llm_called") and is_provider_fallback(answer):
                    raise RuntimeError("Generator returned the configured provider-error fallback")
                if (
                    classification == "internal"
                    and trace.get("strategy")
                    and rag.strategy_selector.last_call_status != "PASS"
                ):
                    raise RuntimeError(
                        "StrategySelector did not complete its real model call: "
                        f"{rag.strategy_selector.last_call_status}"
                    )
            if not answer:
                raise RuntimeError("Pipeline returned an empty answer")
            row["answer"] = answer
            row.pop("error", None)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        row["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        row.setdefault("component_latency_ms", {})["full_rag"] = round(
            row["elapsed_seconds"] * 1000, 3
        )
        row["final_llm_model"] = cfg.LLM_MODEL
        return index, sample, row

    def persist_full():
        ordered = [by_id[item["id"]] for item in samples if item["id"] in by_id]
        summary = aggregate_full(ordered)
        output = {
            "status": (
                "COMPLETED"
                if summary["execution_count"] == 350
                and summary["execution_successes"] == 350
                and summary["execution_errors"] == 0
                else "RUNNING_OR_PARTIAL"
            ),
            "final_llm_model": cfg.LLM_MODEL,
            "pipeline": [
                "FAQ Routing",
                "QueryClassifier",
                "StrategySelector",
                "Query Rewrite",
                "Retriever",
                "Dynamic M",
                "Evidence Gate",
                "Qwen Generation",
            ],
            "execution_concurrency": 3,
            "concurrency_note": "Provider calls overlap; shared local classifier, embedding, reranker and Milvus operations are serialized without changing evaluation parameters.",
            "historical_qwen3_max_results_preserved": True,
            "frozen_input_sha256": frozen_hashes,
            **summary,
            "results": ordered,
        }
        save(FULL_OUT, output)

    try:
        pending = []
        for index, sample in enumerate(samples, 1):
            previous = by_id.get(sample["id"])
            if previous and not previous.get("error"):
                print(f"Full RAG {index}/350 {sample['id']}: checkpoint", flush=True)
                continue
            pending.append((index, sample, previous))
        account_blocker = None
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="qwen37-full") as executor:
            completed = len(samples) - len(pending)
            for start in range(0, len(pending), 3):
                batch = pending[start : start + 3]
                future_map = {
                    executor.submit(execute, index, sample, previous): (index, sample)
                    for index, sample, previous in batch
                }
                for future in as_completed(future_map):
                    index, sample, row = future.result()
                    by_id[sample["id"]] = row
                    completed += 1
                    persist_full()
                    print(
                        f"Full RAG {completed}/350 source_index={index} {sample['id']}: "
                        f"{row.get('route', 'ERROR')} "
                        f"{'PASS' if not row.get('error') else row['error']}",
                        flush=True,
                    )
                    if is_account_blocker(row.get("error")):
                        account_blocker = row["error"]
                if account_blocker:
                    break
        if account_blocker:
            checkpoint = load(FULL_OUT.name)
            checkpoint["status"] = "CHECKPOINTED_ACCOUNT_BLOCKER"
            checkpoint["account_blocker"] = account_blocker
            save(FULL_OUT, checkpoint)
            raise RuntimeError(account_blocker)
    finally:
        qa.mysql_client.close()
    final = load(FULL_OUT.name)
    print(json.dumps({key: final[key] for key in ("status", "final_llm_model", "execution_count", "execution_successes", "execution_errors", "refusal_count", "faq_direct_count", "rag_count", "general_count")}, ensure_ascii=False, indent=2))
    if final["status"] != "COMPLETED":
        raise RuntimeError("Full RAG 350 did not complete without errors")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("generation", "full", "all"))
    args = parser.parse_args()
    if args.stage in {"generation", "all"}:
        run_generation()
    if args.stage in {"full", "all"}:
        run_full()


if __name__ == "__main__":
    main()
