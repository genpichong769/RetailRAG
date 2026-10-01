"""Explicitly initialize RetailRAG FAQ data and invalidate its Redis cache."""

import argparse
from pathlib import Path

from mysql_qa import MySQLClient, RedisClient


def main():
    parser = argparse.ArgumentParser(description="导入 RetailRAG FAQ 数据")
    default_csv = Path(__file__).resolve().parent / "data" / "retail_faq.csv"
    parser.add_argument("--csv", type=Path, default=default_csv)
    parser.add_argument("--append", action="store_true", help="追加而不是替换现有 FAQ")
    args = parser.parse_args()

    mysql_client = MySQLClient()
    try:
        mysql_client.create_table()
        count = mysql_client.insert_data(args.csv, replace=not args.append)
        removed = RedisClient().clear_faq_cache()
        print(f"FAQ 导入完成：{count} 条；已清理 Redis 缓存键：{removed} 个")
    finally:
        mysql_client.close()


if __name__ == "__main__":
    main()
