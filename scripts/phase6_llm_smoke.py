"""Real module smoke tests for the PHASE 6 LLM-only migration."""

import json
import re
import sys
import time
from pathlib import Path

import httpx
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from base import Config
from rag_qa.core.new_rag_system import RAGSystem
from rag_qa.core.vector_store import VectorStore


OUT = ROOT / "RETAILRAG_PHASE6_LLM_SMOKE.json"


def main():
    cfg = Config()
    client = OpenAI(
        api_key=cfg.DASHSCOPE_API_KEY,
        base_url=cfg.DASHSCOPE_BASE_URL,
        http_client=httpx.Client(trust_env=False, timeout=90),
        max_retries=0,
    )

    def non_stream(prompt, system="你是 Nova Retail 智能知识助手。", temperature=0.0):
        response = client.chat.completions.create(
            model=cfg.LLM_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": str(prompt)},
            ],
            temperature=temperature,
            timeout=90,
        )
        return (response.choices[0].message.content or "").strip() if response.choices else ""

    def stream(prompt):
        response = client.chat.completions.create(
            model=cfg.LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是 Nova Retail 智能知识助手。"},
                {"role": "user", "content": str(prompt)},
            ],
            temperature=0.1,
            timeout=90,
            stream=True,
        )
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    checks = {}

    def record(name, started, passed, **details):
        checks[name] = {
            "status": "PASS" if passed else "FAIL",
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            **details,
        }
        OUT.write_text(
            json.dumps(
                {
                    "status": "RUNNING",
                    "selected_model": cfg.LLM_MODEL,
                    "credential_value_logged": False,
                    "checks": checks,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(name, checks[name]["status"], flush=True)

    started = time.perf_counter()
    text = non_stream("你好，只回复：MODEL_OK")
    record("ordinary_generation", started, bool(text), response=text)

    started = time.perf_counter()
    raw_json = non_stream(
        '只输出 JSON，不要解释：{"status":"JUDGE_OK","score":1}',
        system="你是严格的 JSON 评测裁判。",
    )
    match = re.search(r"\{.*\}", raw_json, re.S)
    parsed = json.loads(match.group(0)) if match else None
    record(
        "judge_json",
        started,
        bool(parsed and parsed.get("status") == "JUDGE_OK"),
        raw_response=raw_json,
        parsed=parsed,
    )

    started = time.perf_counter()
    chunks = list(stream("只回复：STREAM_OK"))
    streaming_text = "".join(chunks).strip()
    record(
        "provider_streaming",
        started,
        bool(chunks and streaming_text),
        chunk_count=len(chunks),
        response=streaming_text,
    )

    store = VectorStore()
    rag = RAGSystem(store, stream)

    started = time.perf_counter()
    strategy = rag.strategy_selector.select_strategy(
        "比较开店和闭店时现金、设备、库存分别要检查什么？"
    )
    record(
        "strategy_selector",
        started,
        rag.strategy_selector.last_call_status == "PASS"
        and strategy in rag.strategy_selector.VALID_STRATEGIES,
        parsed_strategy=strategy,
        raw_response=rag.strategy_selector.last_raw_response,
    )

    branch_cases = {
        "hyde": ("怎样做才能让会员觉得积分更好用？", "假设问题检索"),
        "subquery": (
            "比较开店和闭店时现金、设备、库存分别要检查什么？",
            "子查询检索",
        ),
        "backtracking": (
            "促销订单部分退货又用了赠品和券该怎么算？",
            "回溯问题检索",
        ),
    }
    for name, (question, branch) in branch_cases.items():
        started = time.perf_counter()
        documents = rag.retrieve_and_merge(question, strategy=branch)
        trace = dict(rag.last_retrieval_trace or {})
        transformed = trace.get("retrieval_queries") or []
        record(
            name,
            started,
            trace.get("strategy") == branch and bool(transformed) and bool(documents),
            strategy=trace.get("strategy"),
            retrieval_queries=transformed,
            retrieved_documents=[
                doc.metadata.get("document_name") for doc in documents
            ],
        )

    started = time.perf_counter()
    original_followup = "那没有小票呢？"
    rewritten = rag.rewrite_query(
        original_followup,
        [
            {
                "question": "顾客购买商品后想退货，需要什么条件？",
                "answer": "需要核验购买记录、商品状态与适用的退换货规则。",
            }
        ],
    )
    record(
        "query_rewrite",
        started,
        bool(rewritten and rewritten != original_followup and len(rewritten) <= 300),
        original=original_followup,
        rewritten=rewritten,
    )

    started = time.perf_counter()
    answer = "".join(rag.generate_answer("没有纸质小票退货需要哪些凭证？", history=[]))
    trace = dict(rag.last_generation_trace or {})
    record(
        "rag_final_generation",
        started,
        bool(answer)
        and trace.get("llm_called") is True
        and "语言模型暂时不可用" not in answer
        and "当前无法生成回答" not in answer,
        answer=answer,
        strategy=trace.get("strategy"),
        references=trace.get("references", []),
        llm_called=trace.get("llm_called"),
    )

    status = "PASS" if all(row["status"] == "PASS" for row in checks.values()) else "FAIL"
    output = {
        "status": status,
        "selected_model": cfg.LLM_MODEL,
        "credential_value_logged": False,
        "checks": checks,
    }
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": status, "model": cfg.LLM_MODEL, "checks": {k: v["status"] for k, v in checks.items()}}, ensure_ascii=False, indent=2))
    if status != "PASS":
        raise RuntimeError("One or more LLM migration smoke checks failed")


if __name__ == "__main__":
    main()
