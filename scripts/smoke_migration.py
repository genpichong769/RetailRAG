"""Minimal online API smoke checks; never runs the frozen full benchmark."""

import csv
import json
import uuid
from pathlib import Path

import httpx
import websocket


ROOT = Path(__file__).resolve().parents[1]
HTTP = "http://127.0.0.1:8003"
WS = "ws://127.0.0.1:8003/api/stream"


def load(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def stream(query, session_id):
    ws = websocket.create_connection(WS, timeout=120, http_proxy_host=None)
    try:
        ws.send(json.dumps({"query": query, "session_id": session_id}, ensure_ascii=False))
        types = []
        tokens = []
        while True:
            message = json.loads(ws.recv())
            types.append(message["type"])
            if message["type"] == "token":
                tokens.append(message.get("token", ""))
            if message["type"] in {"end", "error"}:
                break
        answer = "".join(tokens)
        return {"types": types, "answer": answer, "answer_length": len(answer), "error": message.get("error")}
    finally:
        ws.close()


def main():
    checks = {}
    with httpx.Client(timeout=30, trust_env=False) as client:
        checks["health"] = client.get(f"{HTTP}/health").status_code == 200
        checks["frontend"] = client.get(f"{HTTP}/").status_code == 200
        checks["sources"] = len(client.get(f"{HTTP}/api/sources").json()["sources"]) == 8
        with (ROOT / "mysql_qa/data/retail_faq.csv").open(encoding="utf-8-sig", newline="") as handle:
            faq_question = next(csv.DictReader(handle))["问题"]
        result = client.post(f"{HTTP}/api/query", json={"query": faq_question}).json()
        checks["faq_exact"] = bool(result["answer"]) and not result["is_streaming"]
        print("FAQ_EXACT", checks["faq_exact"], flush=True)

        paraphrases = [row["query"] for row in load("RETAILRAG_FAQ_EVALUATION_SET.json")["samples"]
                       if row.get("question_type") == "paraphrase"]
        checks["faq_paraphrase"] = False
        for question in paraphrases[:10]:
            result = client.post(f"{HTTP}/api/query", json={"query": question}).json()
            if bool(result["answer"]) and not result["is_streaming"]:
                checks["faq_paraphrase"] = True
                break
        print("FAQ_PARAPHRASE", checks["faq_paraphrase"], flush=True)

    samples = load("RETAILRAG_RAG_EVALUATION_SET.json")["samples"]
    frozen_results = load("RETAILRAG_QWEN37_FINAL_350_RESULTS.json")["results"]
    for kind in ("single_document", "multi_document", "unanswerable"):
        classification = "CORRECT_REFUSAL" if kind == "unanswerable" else "CORRECT"
        sample = next(row for row in frozen_results
                      if row["question_type"] == kind and row["classification"] == classification)
        result = stream(sample["question"], str(uuid.uuid4()))
        answer = result["answer"]
        expected_content = ("知识库信息不足" in answer if kind == "unanswerable"
                            else not answer.startswith(("当前知识库信息不足", "抱歉，语言模型暂时不可用")))
        checks[kind] = ("start" in result["types"] and "end" in result["types"]
                        and result["answer_length"] > 0 and expected_content)
        print(kind.upper(), checks[kind], result["answer_length"], result["error"], flush=True)

    multi = next(row for row in samples if row["question_type"] == "multiturn")
    session = str(uuid.uuid4())
    prior = stream(multi["history"][0]["question"], session)
    follow = stream(multi["question"], session)
    checks["query_rewrite_multiturn"] = (
        "end" in prior["types"] and "end" in follow["types"] and follow["answer_length"] > 0
    )
    print("QUERY_REWRITE_MULTITURN", checks["query_rewrite_multiturn"],
          follow["answer_length"], follow["error"], flush=True)
    print(json.dumps({"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks},
                     ensure_ascii=False, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
