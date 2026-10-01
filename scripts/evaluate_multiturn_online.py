"""Run the required three-turn conversation through real MySQL, Milvus and Qwen."""

import json
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from new_main import IntegratedQASystem


TURNS = [
    "顾客买的商品想退货",
    "买了5天，有小票",
    "还是活动期间买的",
]


def main():
    qa = IntegratedQASystem()
    session_id = str(uuid.uuid4())
    turn_results = []
    try:
        for question in TURNS:
            started = time.perf_counter()
            answer = ""
            for token, is_complete in qa.query(question, session_id=session_id):
                answer += token or ""
                if is_complete:
                    break
            trace = dict(qa.rag_system.last_generation_trace or {})
            turn_results.append(
                {
                    "question": question,
                    "answer": answer,
                    "classification": trace.get("classification"),
                    "strategy": trace.get("strategy"),
                    "retrieval_trace": trace.get("retrieval_trace"),
                    "prompt": trace.get("prompt"),
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                }
            )

        qa.mysql_client.cursor.execute(
            "SELECT question, answer, timestamp FROM conversations "
            "WHERE session_id=%s ORDER BY timestamp, id",
            (session_id,),
        )
        database_rows = [
            {"question": row[0], "answer": row[1], "timestamp": row[2].isoformat()}
            for row in qa.mysql_client.cursor.fetchall()
        ]
        recent_history = qa.get_session_history(session_id)
        final_prompt = turn_results[-1].get("prompt") or ""
        final_retrieval = turn_results[-1].get("retrieval_trace") or {}
        result = {
            "status": "PASS" if (
                len(database_rows) == 3
                and len(recent_history) == 3
                and TURNS[0] in final_prompt
                and TURNS[1] in final_prompt
                and final_retrieval.get("original_query") == TURNS[2]
                and all(row["answer"] for row in turn_results)
            ) else "FAIL",
            "session_id": session_id,
            "turn_count": len(turn_results),
            "database_row_count": len(database_rows),
            "recent_history_count": len(recent_history),
            "final_prompt_contains_turn_1": TURNS[0] in final_prompt,
            "final_prompt_contains_turn_2": TURNS[1] in final_prompt,
            "final_retrieval_original_query": final_retrieval.get("original_query"),
            "retriever_uses_current_turn_not_history_rewrite": final_retrieval.get("original_query") == TURNS[2],
            "database_rows": database_rows,
            "recent_history": recent_history,
            "turns": turn_results,
        }
    finally:
        qa.mysql_client.close()
    (ROOT / "RETAILRAG_MULTITURN_ONLINE_RESULTS.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in result.items() if key not in {"turns", "database_rows", "recent_history"}}, ensure_ascii=True, indent=2))
    if result["status"] != "PASS":
        raise RuntimeError("Online multi-turn verification failed")


if __name__ == "__main__":
    main()
