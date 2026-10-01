"""Build RetailRAG chunks locally and audit metadata without Milvus."""

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from base import Config
from rag_qa.core.document_processor import process_documents


def main():
    logging.disable(logging.INFO)
    config = Config()
    data_root = ROOT / "rag_qa" / "data"
    total_documents = 0
    total_parent_chunks = 0
    total_child_chunks = 0
    per_category = {}
    missing_metadata = []

    for source in config.VALID_SOURCES:
        directory = data_root / f"{source}_data"
        documents = list(directory.glob("*.md"))
        chunks = process_documents(
            str(directory),
            config.PARENT_CHUNK_SIZE,
            config.CHILD_CHUNK_SIZE,
            config.CHUNK_OVERLAP,
        )
        parents = {
            (chunk.metadata.get("document_name"), chunk.metadata.get("parent_id"))
            for chunk in chunks
        }
        for chunk in chunks:
            required = ("source", "category", "document_name", "parent_id", "parent_content")
            absent = [key for key in required if not chunk.metadata.get(key)]
            if absent:
                missing_metadata.append(
                    {"source": source, "missing": absent, "metadata": chunk.metadata}
                )
        per_category[source] = {
            "documents": len(documents),
            "parent_chunks": len(parents),
            "child_chunks": len(chunks),
        }
        total_documents += len(documents)
        total_parent_chunks += len(parents)
        total_child_chunks += len(chunks)

    result = {
        "status": "PASS" if not missing_metadata else "FAIL",
        "parameters": {
            "parent_chunk_size": config.PARENT_CHUNK_SIZE,
            "child_chunk_size": config.CHILD_CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
        },
        "documents": total_documents,
        "parent_chunks": total_parent_chunks,
        "child_chunks": total_child_chunks,
        "missing_metadata_count": len(missing_metadata),
        "per_category": per_category,
    }
    output = ROOT / "RETAILRAG_DOCUMENT_CHUNK_AUDIT.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if total_documents != 38 or missing_metadata:
        raise RuntimeError("Document chunk audit failed")


if __name__ == "__main__":
    main()
