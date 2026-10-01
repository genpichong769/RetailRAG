"""Load and execute the local BGE-M3 and BGE Reranker models without Milvus."""

import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from milvus_model.hybrid import BGEM3EmbeddingFunction
from sentence_transformers import CrossEncoder

from base import Config


def infer_shape(value):
    if hasattr(value, "shape"):
        return list(value.shape)
    if not value:
        return [0]
    first = value[0]
    if hasattr(first, "shape"):
        return [len(value), int(first.shape[-1])]
    return [len(value), len(first)]


def main():
    config = Config()
    result = {"device": "cpu"}

    started = time.perf_counter()
    embedding = BGEM3EmbeddingFunction(
        model_name_or_path=config.bge_m3,
        use_fp16=False,
        device="cpu",
    )
    vectors = embedding([
        "会员积分什么时候过期？",
        "闭店盘点差异如何上报？",
    ])
    dense_shape = infer_shape(vectors["dense"])
    sparse_shape = infer_shape(vectors["sparse"])
    result["bge_m3"] = {
        "status": "PASS",
        "dense_shape": dense_shape,
        "sparse_shape": sparse_shape,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    del vectors, embedding
    gc.collect()

    started = time.perf_counter()
    reranker = CrossEncoder(config.bge_reranker, device="cpu")
    scores = reranker.predict([
        ["积分何时到期？", "积分在获得后的下一自然年 12 月 31 日到期。"],
        ["积分何时到期？", "闭店时应完成现金交接和重点商品盘点。"],
    ])
    result["reranker"] = {
        "status": "PASS",
        "scores": [float(score) for score in scores],
        "relevant_ranked_first": bool(float(scores[0]) > float(scores[1])),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    del reranker
    gc.collect()

    output = ROOT / "RETAILRAG_LOCAL_MODEL_CHECK.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
