"""Compare multi-turn retrieval without and with standalone-query rewriting."""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from base import Config
from rag_qa.core.prompts import RAGPrompts
from rag_qa.core.vector_store import VectorStore

DATASET = ROOT / "RETAILRAG_RAG_EVALUATION_SET.json"
OUT = ROOT / "RETAILRAG_QWEN37_FINAL_QUERY_REWRITE_30.json"


def score_result(docs, expected):
    names = [doc.metadata.get("document_name") for doc in docs]
    rank = next((index for index, name in enumerate(names, 1) if name in expected), None)
    return {
        "retrieved_documents": names,
        "hit_at_1": int(bool(names) and names[0] in expected),
        "hit_at_3": int(bool(set(names[:3]) & expected)),
        "reciprocal_rank": 1 / rank if rank else 0.0,
    }


def main():
    config = Config()
    if config.LLM_MODEL != "qwen3.7-plus":
        raise RuntimeError(f"Unexpected final model: {config.LLM_MODEL}")
    samples = [
        row for row in json.loads(DATASET.read_text(encoding="utf-8"))["samples"]
        if row["question_type"] == "multiturn"
    ]
    client = OpenAI(
        api_key=config.DASHSCOPE_API_KEY,
        base_url=config.DASHSCOPE_BASE_URL,
        http_client=httpx.Client(trust_env=False, timeout=60),
    )
    store = VectorStore()
    prepared = []
    rewrite_latencies = []
    usage_rows = []
    for index, sample in enumerate(samples, 1):
        history_text = "\n".join(
            f"Q: {item['question']}\nA: {item['answer']}" for item in sample["history"]
        )
        prompt = RAGPrompts.query_rewrite_prompt().format(
            history=history_text, question=sample["question"]
        )
        started = time.perf_counter()
        rewrite_error = None
        try:
            response = client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                timeout=60,
            )
            rewritten = response.choices[0].message.content.strip().strip('"“”')
            if not rewritten:
                raise RuntimeError("Empty rewritten query")
            usage_rows.append(response.usage.model_dump() if response.usage else None)
        except Exception as exc:
            rewritten = sample["question"]
            rewrite_error = f"{type(exc).__name__}: {exc}"
            usage_rows.append(None)
        rewrite_latency_ms = round((time.perf_counter() - started) * 1000, 3)
        rewrite_latencies.append(rewrite_latency_ms)
        expected = set(sample["reference_documents"])
        prepared.append(
            {
                "id": sample["id"],
                "question": sample["question"],
                "history": sample["history"],
                "rewritten_query": rewritten,
                "rewrite_error": rewrite_error,
                "rewrite_latency_ms": rewrite_latency_ms,
                "expected_documents": sample["reference_documents"],
                "expected_set": expected,
                "without_docs": store.search_documents(
                    sample["question"], mode="hybrid", rerank=False, candidate_k=5, limit=5
                ),
                "with_docs": store.search_documents(
                    rewritten, mode="hybrid", rerank=False, candidate_k=5, limit=5
                ),
            }
        )
        print(f"rewrite: {index}/{len(samples)}", flush=True)

    pairs = []
    for row in prepared:
        pairs.extend([[row["question"], doc.page_content] for doc in row["without_docs"]])
        pairs.extend([[row["rewritten_query"], doc.page_content] for doc in row["with_docs"]])
    scores = store.reranker.predict(pairs, batch_size=32, show_progress_bar=True)
    offset = 0
    rows = []
    for row in prepared:
        for key in ("without_docs", "with_docs"):
            docs = row[key]
            doc_scores = scores[offset : offset + len(docs)]
            offset += len(docs)
            for score, doc in zip(doc_scores, docs):
                doc.metadata["reranker_score"] = float(score)
            row[key] = sorted(docs, key=lambda doc: doc.metadata["reranker_score"], reverse=True)
        rows.append(
            {
                "id": row["id"],
                "question": row["question"],
                "history": row["history"],
                "rewritten_query": row["rewritten_query"],
                "rewrite_error": row["rewrite_error"],
                "rewrite_latency_ms": row["rewrite_latency_ms"],
                "expected_documents": row["expected_documents"],
                "without_rewrite": score_result(row["without_docs"], row["expected_set"]),
                "with_rewrite": score_result(row["with_docs"], row["expected_set"]),
            }
        )

    def aggregate(key):
        return {
            "hit_at_1": round(sum(row[key]["hit_at_1"] for row in rows) / len(rows), 6),
            "hit_at_3": round(sum(row[key]["hit_at_3"] for row in rows) / len(rows), 6),
            "mrr": round(sum(row[key]["reciprocal_rank"] for row in rows) / len(rows), 6),
        }

    before = aggregate("without_rewrite")
    after = aggregate("with_rewrite")
    improved = unchanged = degraded = 0
    for row in rows:
        before_rr = row["without_rewrite"]["reciprocal_rank"]
        after_rr = row["with_rewrite"]["reciprocal_rank"]
        outcome = "IMPROVED" if after_rr > before_rr else "DEGRADED" if after_rr < before_rr else "UNCHANGED"
        row["rewrite_outcome"] = outcome
        improved += int(outcome == "IMPROVED")
        unchanged += int(outcome == "UNCHANGED")
        degraded += int(outcome == "DEGRADED")
    successful = sum(not row["rewrite_error"] for row in rows)
    def percentile(values, p):
        values = sorted(values)
        position = (len(values) - 1) * p / 100
        lower = int(position); upper = min(lower + 1, len(values) - 1)
        weight = position - lower
        return round(values[lower] * (1 - weight) + values[upper] * weight, 6)
    token_usage = {
        key: sum(int(row.get(key, 0)) for row in usage_rows if row)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    } if any(usage_rows) else "NOT_AVAILABLE"
    output = {
        "status": "PASS",
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "model": config.LLM_MODEL,
        "sample_count": len(rows),
        "without_query_rewrite": before,
        "with_query_rewrite": after,
        "gain": {key: round(after[key] - before[key], 6) for key in before},
        "rewrite_success_rate": round(successful / len(rows), 6),
        "rewrite_error_rate": round((len(rows) - successful) / len(rows), 6),
        "rewrite_improved": improved,
        "rewrite_unchanged": unchanged,
        "rewrite_degraded": degraded,
        "rewrite_latency_ms": {
            "mean": round(sum(rewrite_latencies) / len(rewrite_latencies), 6),
            "median": percentile(rewrite_latencies, 50),
            "p95": percentile(rewrite_latencies, 95),
            "p99": percentile(rewrite_latencies, 99),
        },
        "usage": token_usage,
        "enable_recommendation": after["hit_at_3"] >= before["hit_at_3"] and after["mrr"] >= before["mrr"],
        "results": rows,
    }
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: output[key] for key in ("without_query_rewrite", "with_query_rewrite", "gain", "enable_recommendation")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
