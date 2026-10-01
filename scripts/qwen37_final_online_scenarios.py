"""Run the remaining final online scenarios through HTTP and WebSocket."""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path

import httpx
import websocket


ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8003"
WS_URL = "ws://127.0.0.1:8003/api/stream"
OUT = ROOT / "RETAILRAG_QWEN37_FINAL_ONLINE_SCENARIOS.json"


def stream_query(query):
    session_id = str(uuid.uuid4())
    socket = websocket.create_connection(WS_URL, timeout=180, http_proxy_host=None)
    socket.settimeout(180)
    tokens = []
    events = []
    started = time.perf_counter()
    try:
        socket.send(json.dumps({"query": query, "session_id": session_id}, ensure_ascii=False))
        while True:
            event = json.loads(socket.recv())
            events.append(event.get("type"))
            if event.get("type") == "token":
                tokens.append(event.get("token", ""))
            if event.get("type") in {"end", "error"}:
                break
    finally:
        socket.close(status=1000)
    return {
        "query": query,
        "session_id": session_id,
        "answer": "".join(tokens),
        "events": events,
        "token_events": events.count("token"),
        "end_received": "end" in events,
        "error_received": "error" in events,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def main():
    with httpx.Client(trust_env=False, timeout=30) as client:
        faq_exact = client.post(f"{BASE}/api/query", json={"query": "会员等级怎么划分？"}).json()
        faq_paraphrase = client.post(f"{BASE}/api/query", json={"query": "会员等级是怎么分的？"}).json()

    scenarios = {
        "single_document_rag": stream_query("无小票退货需要什么凭证？"),
        "multi_document_complex_rag": stream_query("比较开店和闭店时现金、设备、库存分别要检查什么？"),
        "version_conflict": stream_query("现在金卡积分怎么算？"),
        "unanswerable": stream_query("Nova Retail 今年营业收入多少？"),
        "general_query": stream_query("地球为什么围绕太阳公转？"),
    }
    prior = json.loads((ROOT / "RETAILRAG_API_ONLINE_RESULTS.json").read_text(encoding="utf-8"))
    checks = {
        "faq_exact": bool(faq_exact.get("answer")) and not faq_exact.get("is_streaming"),
        "faq_paraphrase": bool(faq_paraphrase.get("answer")) and not faq_paraphrase.get("is_streaming"),
        "single_document_rag": scenarios["single_document_rag"]["end_received"] and scenarios["single_document_rag"]["token_events"] > 0,
        "multi_document_complex_rag": scenarios["multi_document_complex_rag"]["end_received"] and scenarios["multi_document_complex_rag"]["token_events"] > 0,
        "version_conflict": scenarios["version_conflict"]["end_received"] and "1.5" in scenarios["version_conflict"]["answer"] and "2026" in scenarios["version_conflict"]["answer"],
        "unanswerable": scenarios["unanswerable"]["end_received"] and "当前知识库信息不足" in scenarios["unanswerable"]["answer"],
        "general_query": scenarios["general_query"]["end_received"] and scenarios["general_query"]["token_events"] > 0,
        "multi_turn_query_rewrite": prior.get("checks", {}).get("multiturn_websocket") is True and prior.get("checks", {}).get("multiturn_history_three") is True,
        "websocket_streaming": all(row["end_received"] and not row["error_received"] for row in scenarios.values()),
    }
    output = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checked_at": datetime.now().astimezone().isoformat(),
        "model": "qwen3.7-plus",
        "checks": checks,
        "faq_exact": faq_exact,
        "faq_paraphrase": faq_paraphrase,
        "scenarios": scenarios,
        "transport_note": "websocket-client uses a 180-second receive timeout and does not send periodic keepalive pings.",
    }
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": output["status"], "checks": checks, "latency_seconds": {key: value["elapsed_seconds"] for key, value in scenarios.items()}}, ensure_ascii=False, indent=2))
    if output["status"] != "PASS":
        raise RuntimeError("One or more final online scenarios failed")


if __name__ == "__main__":
    main()
