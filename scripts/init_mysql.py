"""Idempotently create RetailRAG MySQL tables and replace demo FAQ data."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mysql_qa import MySQLClient


def main():
    faq_path = ROOT / "mysql_qa" / "data" / "retail_faq.csv"
    client = MySQLClient()
    try:
        client.create_table()
        client.create_conversations_table()
        imported = client.insert_data(faq_path, replace=True)
        actual = client.count_faqs()
        if imported != 200 or actual != 200:
            raise RuntimeError(f"FAQ count mismatch: imported={imported}, actual={actual}")
        print("MYSQL_INIT=PASS tables=jpkb,conversations faq_count=200")
    finally:
        client.close()


if __name__ == "__main__":
    main()
