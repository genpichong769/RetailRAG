"""Exercise RetailRAG HTTP and WebSocket endpoints against the live Uvicorn process."""

import json
import sys
import time
from pathlib import Path

import httpx
import websocket

ROOT = Path(__file__).resolve().parent.parent
BASE = "http://127.0.0.1:8003"
WS_URL = "ws://127.0.0.1:8003/api/stream"


def stream_query(query, session_id):
    ws = websocket.create_connection(WS_URL, timeout=120, http_proxy_host=None)
    ws.settimeout(120)
    events = []
    tokens = []
    started = time.perf_counter()
    try:
        ws.send(json.dumps({"query": query, "session_id": session_id}, ensure_ascii=False))
        while True:
            event = json.loads(ws.recv())
            events.append(event)
            if event.get("type") == "token":
                tokens.append(event.get("token", ""))
            if event.get("type") in {"end", "error"}:
                break
    finally:
        ws.close(status=1000)
    return {
        "query": query,
        "events": [event.get("type") for event in events],
        "token_events": sum(event.get("type") == "token" for event in events),
        "answer": "".join(tokens),
        "end_received": any(event.get("type") == "end" for event in events),
        "error_events": [event for event in events if event.get("type") == "error"],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def main():
    with httpx.Client(trust_env=False, timeout=30) as client:
        root = client.get(BASE + "/")
        health = client.get(BASE + "/health")
        session_response = client.post(BASE + "/api/create_session")
        session_id = session_response.json()["session_id"]

        faq = client.post(
            BASE + "/api/query",
            json={"query": "会员等级怎么划分？", "session_id": session_id},
        )
        rag_probe = client.post(
            BASE + "/api/query",
            json={"query": "顾客买的商品想退货", "session_id": session_id},
        )
        known_false_intercept = client.post(
            BASE + "/api/query",
            json={"query": "金卡会员的保级条件是什么？", "session_id": session_id},
        )
        general_probe = client.post(
            BASE + "/api/query",
            json={"query": "地球为什么围绕太阳公转？", "session_id": session_id},
        )

        rag_stream = stream_query("顾客买的商品想退货", session_id)
        general_stream = stream_query("地球为什么围绕太阳公转？", session_id)

        multi_session = client.post(BASE + "/api/create_session").json()["session_id"]
        multi_results = [
            stream_query(question, multi_session)
            for question in ["顾客买的商品想退货", "买了5天，有小票", "还是活动期间买的"]
        ]
        history_response = client.get(BASE + f"/api/history/{multi_session}")
        history = history_response.json().get("history", [])

    checks = {
        "root_200_retail_page": root.status_code == 200 and "RetailRAG" in root.text and "Nova Retail" in root.text,
        "health_200": health.status_code == 200 and health.json().get("status") == "healthy",
        "create_session_200": session_response.status_code == 200 and bool(session_id),
        "faq_http_200_answer": faq.status_code == 200 and not faq.json().get("is_streaming") and bool(faq.json().get("answer")),
        "rag_http_requests_stream": rag_probe.status_code == 200 and rag_probe.json().get("is_streaming") is True,
        "general_http_requests_stream": general_probe.status_code == 200 and general_probe.json().get("is_streaming") is True,
        "rag_websocket": rag_stream["end_received"] and rag_stream["token_events"] > 0 and not rag_stream["error_events"],
        "rag_answer_has_source_reference": "来源" in rag_stream["answer"] and "return_exchange" in rag_stream["answer"],
        "general_websocket": general_stream["end_received"] and general_stream["token_events"] > 0 and not general_stream["error_events"],
        "multiturn_websocket": all(row["end_received"] and row["token_events"] > 0 and not row["error_events"] for row in multi_results),
        "multiturn_history_three": history_response.status_code == 200 and len(history) == 3,
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "quality_status": "FAIL" if not known_false_intercept.json().get("is_streaming") else "PASS",
        "known_faq_false_intercept": {
            "query": "金卡会员的保级条件是什么？",
            "status_code": known_false_intercept.status_code,
            "response": known_false_intercept.json(),
            "expected_to_enter_rag": True,
        },
        "base_url": BASE,
        "checks": checks,
        "http": {
            "root_status": root.status_code,
            "health_status": health.status_code,
            "faq_status": faq.status_code,
            "faq_response": faq.json(),
            "rag_probe_status": rag_probe.status_code,
            "rag_probe_response": rag_probe.json(),
            "known_false_intercept_response": known_false_intercept.json(),
            "general_probe_status": general_probe.status_code,
            "general_probe_response": general_probe.json(),
        },
        "websocket": {
            "rag": rag_stream,
            "general": general_stream,
            "multiturn": multi_results,
        },
        "multiturn_session_id": multi_session,
        "multiturn_history": history,
    }
    (ROOT / "RETAILRAG_API_ONLINE_RESULTS.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "checks": checks}, ensure_ascii=True, indent=2))
    if result["status"] != "PASS":
        raise RuntimeError("FastAPI/WebSocket online verification failed")


if __name__ == "__main__":
    main()
