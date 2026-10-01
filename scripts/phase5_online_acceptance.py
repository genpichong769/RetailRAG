"""Small online regression suite for the PHASE 5 API and WebSocket path."""

import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

import httpx
import websockets

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "RETAILRAG_PHASE5_ONLINE_ACCEPTANCE.json"
BASE = "http://127.0.0.1:8003"
WS = "ws://127.0.0.1:8003/api/stream"


async def stream_answer(query, session_id):
    tokens = []
    messages = []
    started = time.time()
    async with websockets.connect(WS, open_timeout=20, close_timeout=10) as socket:
        await socket.send(json.dumps({"query": query, "session_id": session_id}, ensure_ascii=False))
        while True:
            message = json.loads(await asyncio.wait_for(socket.recv(), timeout=120))
            messages.append(message)
            if message.get("type") == "token":
                tokens.append(message.get("token", ""))
            if message.get("type") in {"end", "error"}:
                break
    return {
        "query": query,
        "answer": "".join(tokens),
        "messages": messages,
        "elapsed_seconds": round(time.time() - started, 3),
    }


async def main():
    results = []
    async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
        health = await client.get(f"{BASE}/health")
        root = await client.get(f"{BASE}/")
        results.append({"case": "health", "pass": health.status_code == 200 and health.json().get("status") == "healthy"})
        results.append({"case": "formal_page", "pass": root.status_code == 200 and "Nova Retail" in root.text and "your_phone_here" not in root.text})
        faq_cases = [
            ("faq_exact", "会员等级怎么划分？", False),
            ("faq_paraphrase", "会员等级是怎么分的？", False),
            ("faq_regression_boundary", "金卡会员的保级条件是什么？", True),
            ("faq_nonfaq_boundary", "法国的首都是什么？", True),
        ]
        for name, query, expected_streaming in faq_cases:
            response = await client.post(f"{BASE}/api/query", json={"query": query})
            payload = response.json()
            results.append(
                {
                    "case": name,
                    "query": query,
                    "pass": response.status_code == 200 and payload.get("is_streaming") is expected_streaming,
                    "response": payload,
                }
            )
        session = (await client.post(f"{BASE}/api/create_session")).json()["session_id"]

    refusal = await stream_answer("Nova Retail 今年营业收入多少？", session)
    refusal["case"] = "code_level_refusal"
    refusal["pass"] = "当前知识库信息不足" in refusal["answer"]
    results.append(refusal)
    version = await stream_answer("现在金卡积分怎么算？", str(uuid.uuid4()))
    version["case"] = "active_version_answer"
    version["pass"] = "1.5" in version["answer"] and "2026" in version["answer"] and "1 元累计 1 积分" not in version["answer"]
    results.append(version)

    output = {
        "status": "PASS" if all(row["pass"] for row in results) else "FAIL",
        "passed": sum(row["pass"] for row in results),
        "total": len(results),
        "results": results,
    }
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": output["status"], "passed": output["passed"], "total": output["total"], "cases": [{"case": row["case"], "pass": row["pass"]} for row in results]}, ensure_ascii=False, indent=2))
    if output["status"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
