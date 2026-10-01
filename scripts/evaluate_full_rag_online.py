"""Run 15 real classifier-to-Qwen RetailRAG questions and retain diagnostics."""

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


QUESTIONS = [
    "金卡会员的保级条件是什么？",
    "积分什么时候过期？",
    "新人券包在什么条件下发放？",
    "促销赠品发完以后门店怎么处理？",
    "没有纸质小票退货需要哪些凭证？",
    "活动订单部分退货后赠品和优惠怎么处理？",
    "闭店前现金、设备和库存分别要检查什么？",
    "盘点出现三百元差异时如何升级上报？",
    "临期商品如何标识并处理？",
    "冷藏商品到货温度不合格怎么办？",
    "POS 显示扣款成功但订单待支付怎么处置？",
    "冷柜报警且不确定超温时长应该采取哪些动作？",
    "员工把完整支付信息发到公开群后如何升级？",
    "春季会员日活动期间门店人员和陈列有什么要求？",
    "会员日订单用了券和赠品，部分退货后积分如何处理？",
]


def main():
    config = Config()
    client = OpenAI(
        api_key=config.DASHSCOPE_API_KEY,
        base_url=config.DASHSCOPE_BASE_URL,
        http_client=httpx.Client(trust_env=False, timeout=45),
        max_retries=0,
    )

    def qwen_stream(prompt):
        completion = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是 Nova Retail 智能知识助手。"},
                {"role": "user", "content": str(prompt)},
            ],
            temperature=0.1,
            timeout=45,
            stream=True,
        )
        for chunk in completion:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    vector_store = VectorStore()
    rag = RAGSystem(vector_store, qwen_stream)
    results = []
    for index, question in enumerate(QUESTIONS, 1):
        started = time.perf_counter()
        answer = "".join(rag.generate_answer(question, history=[]))
        trace = dict(rag.last_generation_trace or {})
        references = trace.get("references", [])
        results.append(
            {
                "id": index,
                "question": question,
                "classifier": trace.get("classification"),
                "strategy": trace.get("strategy"),
                "retrieval_trace": trace.get("retrieval_trace"),
                "retrieved_sources": trace.get("retrieved_sources", []),
                "final_context": trace.get("final_context"),
                "answer": answer,
                "references": references,
                "answer_mentions_retrieved_document": any(name and name in answer for name in references),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )
        print(
            f"{index:02d}/{len(QUESTIONS)} category={trace.get('classification', {}).get('routed_prediction')} "
            f"strategy={trace.get('strategy')} refs={references} answer_chars={len(answer)}"
        )
    summary = {
        "status": "PASS" if all(
            row["classifier"]
            and row["classifier"]["routed_prediction"] == "internal"
            and row["strategy"]
            and row["retrieved_sources"]
            and row["final_context"]
            and row["answer"]
            for row in results
        ) else "FAIL",
        "quality_status": "PASS" if all(row["answer_mentions_retrieved_document"] for row in results) else "FAIL",
        "question_count": len(results),
        "internal_routes": sum(row["classifier"]["routed_prediction"] == "internal" for row in results),
        "answers_returned": sum(bool(row["answer"]) for row in results),
        "answers_with_document_name": sum(row["answer_mentions_retrieved_document"] for row in results),
        "strategies": sorted({row["strategy"] for row in results}),
    }
    (ROOT / "RETAILRAG_FULL_RAG_ONLINE_RESULTS.json").write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (ROOT / "RETAILRAG_FULL_RAG_ONLINE_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    if summary["status"] != "PASS":
        raise RuntimeError("Full online RAG verification failed")


if __name__ == "__main__":
    main()
