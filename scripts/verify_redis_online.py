"""Exercise the real RetailRAG Redis cache namespace and TTL."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mysql_qa import RedisClient


def main():
    client = RedisClient()
    client.clear_faq_cache()
    query = "phase4.1 Redis 在线诊断"
    expected = "JSON 解码正常"
    client.set_answer(query, expected)
    key = client.answer_key(query)
    actual = client.get_answer(query)
    ttl = client.client.ttl(key)
    keys_before_clear = list(client.client.scan_iter(match=client.faq_key("*")))
    cleared = client.clear_faq_cache()
    exists_after_clear = bool(client.client.exists(key))
    result = {
        "status": "PASS" if all(
            [
                client.client.ping(),
                key.startswith(client.namespace + ":"),
                actual == expected,
                0 < ttl <= client.ttl,
                key in keys_before_clear,
                cleared >= 1,
                not exists_after_clear,
            ]
        ) else "FAIL",
        "ping": bool(client.client.ping()),
        "namespace": client.namespace,
        "json_decoded_value_matches": actual == expected,
        "configured_ttl": client.ttl,
        "actual_ttl": ttl,
        "keys_before_clear": len(keys_before_clear),
        "keys_cleared": cleared,
        "exists_after_clear": exists_after_clear,
    }
    (ROOT / "RETAILRAG_REDIS_ONLINE_RESULTS.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "PASS":
        raise RuntimeError("Redis online verification failed")


if __name__ == "__main__":
    main()
