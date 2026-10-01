"""Create the configured Milvus database when it does not yet exist."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pymilvus import MilvusClient

from base import Config


def main():
    config = Config()
    client = MilvusClient(uri=f"http://{config.MILVUS_HOST}:{config.MILVUS_PORT}", timeout=10)
    databases = client.list_databases()
    if config.MILVUS_DATABASE_NAME not in databases:
        client.create_database(db_name=config.MILVUS_DATABASE_NAME)
        print(f"Created Milvus database: {config.MILVUS_DATABASE_NAME}")
    else:
        print(f"Milvus database already exists: {config.MILVUS_DATABASE_NAME}")


if __name__ == "__main__":
    main()
