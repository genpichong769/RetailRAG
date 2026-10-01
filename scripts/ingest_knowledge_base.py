"""Idempotently upsert all RetailRAG demo documents into Milvus."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from base import Config
from rag_qa.core.document_processor import process_documents
from rag_qa.core.vector_store import VectorStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Recreate the RetailRAG collection before ingestion (schema/version migration).",
    )
    args = parser.parse_args()
    config = Config()
    data_root = ROOT / "rag_qa" / "data"
    markdown_files = [
        path
        for source in config.VALID_SOURCES
        for path in (data_root / f"{source}_data").glob("*.md")
    ]
    all_chunks = []
    per_category = {}
    for source in config.VALID_SOURCES:
        directory = data_root / f"{source}_data"
        chunks = process_documents(
            str(directory),
            config.PARENT_CHUNK_SIZE,
            config.CHILD_CHUNK_SIZE,
            config.CHUNK_OVERLAP,
        )
        per_category[source] = {
            "documents": len(list(directory.glob("*.md"))),
            "parent_chunks": len({chunk.metadata["parent_id"] for chunk in chunks}),
            "child_chunks": len(chunks),
        }
        all_chunks.extend(chunks)

    store = VectorStore()
    if args.replace:
        store.client.release_collection(store.collection_name)
        store.client.drop_collection(store.collection_name)
        store._create_or_load_collection()
    before = int(store.client.get_collection_stats(store.collection_name).get("row_count", 0))
    store.add_documents(all_chunks)
    store.client.flush(store.collection_name)
    compaction_id = store.client.compact(store.collection_name)
    compaction_state = "Executing"
    for _ in range(60):
        compaction_state = store.client.get_compaction_state(compaction_id)
        if str(compaction_state).lower() in {"completed", "failed"}:
            break
        time.sleep(1)
    if str(compaction_state).lower() != "completed":
        raise RuntimeError(f"Milvus compaction did not complete: {compaction_state}")
    after = int(store.client.get_collection_stats(store.collection_name).get("row_count", 0))
    visible = store.client.query(
        collection_name=store.collection_name,
        filter="",
        output_fields=["id"],
        limit=max(1000, len(all_chunks) * 2),
    )
    summary = {
        "collection": store.collection_name,
        "documents": len(markdown_files),
        "parent_chunks": len(
            {
                (chunk.metadata.get("source"), chunk.metadata.get("document_name"), chunk.metadata["parent_id"])
                for chunk in all_chunks
            }
        ),
        "child_chunks": len(all_chunks),
        "entities_before": before,
        "entities_after": after,
        "visible_entities_after": len(visible),
        "unique_entity_ids_after": len({row["id"] for row in visible}),
        "compaction_state": str(compaction_state),
        "replaced_collection": args.replace,
        "status_counts": {
            status: sum(
                1
                for chunk in all_chunks
                if chunk.metadata.get("status", "active") == status
            )
            for status in ("active", "expired", "superseded")
        },
        "per_category": per_category,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if (
        summary["documents"] != len(markdown_files)
        or summary["visible_entities_after"] != summary["child_chunks"]
        or summary["unique_entity_ids_after"] != summary["child_chunks"]
    ):
        raise RuntimeError(
            f"Knowledge-base count mismatch: documents={summary['documents']} "
            f"visible={summary['visible_entities_after']} chunks={summary['child_chunks']}"
        )


if __name__ == "__main__":
    main()
