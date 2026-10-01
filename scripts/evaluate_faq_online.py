"""Run the Phase 4 FAQ cases through real MySQL and Redis clients."""

import csv
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mysql_qa import BM25Search, MySQLClient, RedisClient
from scripts.evaluate_faq_retrieval import EXACT_QUERIES, NON_FAQ_QUERIES, PARAPHRASE_QUERIES


def main():
    logging.disable(logging.INFO)
    mysql_client = MySQLClient()
    redis_client = RedisClient()
    try:
        redis_client.clear_faq_cache()
        search = BM25Search(redis_client, mysql_client)
        groups = {
            "exact": EXACT_QUERIES,
            "paraphrase": PARAPHRASE_QUERIES,
            "non_faq": NON_FAQ_QUERIES,
        }
        results = []
        for kind, queries in groups.items():
            for query in queries:
                details = search.search_with_details(query, threshold=0.85)
                results.append(
                    {
                        "kind": kind,
                        "query": query,
                        "top1_question": details.get("top1_question"),
                        "raw_bm25_score": details.get("raw_score"),
                        "softmax_score": details.get("score"),
                        "threshold": details.get("threshold"),
                        "above_threshold": details.get("above_threshold"),
                        "hit": details.get("hit"),
                        "cached": details.get("cached"),
                        "returned_answer": details.get("answer"),
                        "need_rag": details.get("need_rag"),
                    }
                )

        repeat = search.search_with_details(EXACT_QUERIES[0], threshold=0.85)
        keys = list(redis_client.client.scan_iter(match=redis_client.faq_key("*")))
        ttls = [redis_client.client.ttl(key) for key in keys]
        exact_hits = sum(bool(row["hit"]) for row in results if row["kind"] == "exact")
        paraphrase_hits = sum(bool(row["hit"]) for row in results if row["kind"] == "paraphrase")
        non_faq_rejected = sum(not bool(row["hit"]) for row in results if row["kind"] == "non_faq")
        summary = {
            "status": "PASS" if (
                len(search.original_questions) == 52
                and exact_hits == 10
                and paraphrase_hits == 2
                and non_faq_rejected == 9
                and repeat.get("cached") is True
                and keys
                and all(ttl > 0 for ttl in ttls)
            ) else "FAIL",
            "scope": "real MySQL + Redis + Jieba/BM25 online path",
            "threshold": 0.85,
            "mysql_faq_count": len(search.original_questions),
            "exact_hits": exact_hits,
            "paraphrase_hits": paraphrase_hits,
            "non_faq_rejected": non_faq_rejected,
            "repeat_query_cache_hit": repeat.get("cached"),
            "redis_key_count": len(keys),
            "all_redis_keys_have_ttl": bool(keys) and all(ttl > 0 for ttl in ttls),
        }
    finally:
        mysql_client.close()

    fields = [
        "kind", "query", "top1_question", "raw_bm25_score", "softmax_score",
        "threshold", "above_threshold", "hit", "cached", "returned_answer", "need_rag",
    ]
    with (ROOT / "RETAILRAG_FAQ_ONLINE_RESULTS.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    (ROOT / "RETAILRAG_FAQ_ONLINE_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["status"] != "PASS":
        raise RuntimeError("Online FAQ result differs from the fixed Phase 4 baseline")


if __name__ == "__main__":
    main()
