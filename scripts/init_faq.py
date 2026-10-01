"""Replace RetailRAG FAQ rows and invalidate only its Redis cache namespace."""

import sys
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mysql_qa import MySQLClient, RedisClient


def main():
    faq_path = ROOT / "mysql_qa" / "data" / "retail_faq.csv"
    mysql_client = MySQLClient()
    try:
        mysql_client.create_table()
        imported = mysql_client.insert_data(faq_path, replace=True)
        actual = mysql_client.count_faqs()
    finally:
        mysql_client.close()
    removed = RedisClient().clear_faq_cache()
    with faq_path.open(encoding="utf-8-sig", newline="") as handle:
        expected = len(list(csv.DictReader(handle)))
    if imported != expected or actual != expected:
        raise RuntimeError(f"FAQ count mismatch: imported={imported}, actual={actual}")
    print(f"FAQ_INIT=PASS faq_count={actual} redis_keys_removed={removed}")


if __name__ == "__main__":
    main()
