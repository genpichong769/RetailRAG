"""Run each Qwen-selected retrieval strategy through its real Milvus branch."""

import json
import sys
import time
from pathlib import Path

import httpx
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from base import Config
from rag_qa.core.new_rag_system import RAGSystem
from rag_qa.core.vector_store import VectorStore


def main():
    config = Config()
    client = OpenAI(
        api_key=config.DASHSCOPE_API_KEY,
        base_url=config.DASHSCOPE_BASE_URL,
        http_client=httpx.Client(trust_env=False, timeout=30),
        max_retries=0,
    )

    def qwen(prompt):
        completion = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是 Nova Retail 检索查询转换器。"},
                {"role": "user", "content": str(prompt)},
            ],
            temperature=0.1,
            timeout=30,
        )
        return completion.choices[0].message.content if completion.choices else ""

    cases = [
        ("direct", "无小票退货需要什么凭证？", "直接检索"),
        ("hyde", "怎样做才能让会员觉得积分更好用？", "假设问题检索"),
        ("subquery", "比较开店和闭店时现金、设备、库存分别要检查什么？", "子查询检索"),
        ("backtracking", "促销订单部分退货又用了赠品和券该怎么算？", "回溯问题检索"),
    ]
    vector_store = VectorStore()
    rag = RAGSystem(vector_store, qwen)
    rows = []
    for case_id, query, expected in cases:
        started = time.perf_counter()
        parsed = rag.strategy_selector.select_strategy(query)
        raw = rag.strategy_selector.last_raw_response
        documents = rag.retrieve_and_merge(query, strategy=parsed)
        trace = rag.last_retrieval_trace or {}
        rows.append(
            {
                "case": case_id,
                "query": query,
                "expected_strategy": expected,
                "qwen_raw_response": raw,
                "parsed_strategy": parsed,
                "actual_branch": trace.get("strategy"),
                "retrieval_queries": trace.get("retrieval_queries", []),
                "retrieved_documents": trace.get("documents", []),
                "document_count": len(documents),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )
    result = {
        "status": "PASS" if all(
            row["parsed_strategy"] == row["expected_strategy"]
            and row["actual_branch"] == row["parsed_strategy"]
            and row["retrieval_queries"]
            and row["document_count"] > 0
            for row in rows
        ) else "FAIL",
        "scope": "real Qwen selection, branch execution, query transformation and Milvus retrieval",
        "results": rows,
    }
    (ROOT / "RETAILRAG_STRATEGY_PATHS_ONLINE_RESULTS.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=True, indent=2))
    if result["status"] != "PASS":
        raise RuntimeError("One or more strategy branches did not execute as selected")


if __name__ == "__main__":
    main()
