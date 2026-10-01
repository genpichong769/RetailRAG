"""Compare fixed M=2, global M=3, and Dynamic M on unchanged K=3 retrieval."""

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from base import Config
from base.query_analysis import analyze_query
from rag_qa.core.vector_store import VectorStore
from evaluate_scale import CachedReranker


def load(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def save(name, value):
    path = ROOT / name
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def dynamic_select(question, ranked, cfg):
    analysis = analyze_query(question)
    budget = 3 if analysis["context_complex"] else 2
    selected = ranked[: min(2, budget)]
    diversity_applied = False
    if budget > 2 and len(ranked) > 2:
        top_score = float(ranked[0].metadata.get("reranker_score", 0.0))
        relevance_floor = max(cfg.EVIDENCE_RERANKER_THRESHOLD, max(top_score, 0.0) * 0.02)
        selected_names = {doc.metadata.get("document_name") for doc in selected}
        selected_categories = {
            doc.metadata.get("category") or doc.metadata.get("source") for doc in selected
        }
        eligible = [
            doc
            for doc in ranked[2:]
            if float(doc.metadata.get("reranker_score", float("-inf"))) >= relevance_floor
            and doc.metadata.get("document_name") not in selected_names
        ]
        best_extra_score = (
            float(eligible[0].metadata.get("reranker_score", float("-inf")))
            if eligible
            else float("-inf")
        )
        diverse = [
            doc
            for doc in eligible
            if float(doc.metadata.get("reranker_score", float("-inf")))
            >= best_extra_score * 0.8
            and (doc.metadata.get("category") or doc.metadata.get("source"))
            not in selected_categories
        ]
        if diverse or eligible:
            selected.append((diverse or eligible)[0])
            diversity_applied = bool(diverse)
    return selected, {
        "budget": budget,
        "complex": analysis["context_complex"],
        "diversity_applied": diversity_applied,
    }


def score(sample, documents, metadata):
    names = [doc.metadata.get("document_name") for doc in documents]
    expected = set(sample.get("reference_documents") or [])
    rank = next((index for index, name in enumerate(names, 1) if name in expected), None)
    found = expected & set(names)
    return {
        "id": sample["id"],
        "question": sample["question"],
        "question_type": sample["question_type"],
        "expected_documents": sorted(expected),
        "retrieved_documents": names,
        "any_at_1": int(bool(expected & set(names[:1]))),
        "any_at_3": int(bool(found)),
        "all_at_1": int(bool(expected) and expected.issubset(set(names[:1]))),
        "all_at_3": int(bool(expected) and expected.issubset(set(names))),
        "reciprocal_rank": 1 / rank if rank else 0.0,
        "context_recall": len(found) / len(expected) if expected else None,
        "context_precision": len(found) / len(names) if names and expected else None,
        "context_count": len(documents),
        "context_characters": sum(len(doc.page_content) for doc in documents),
        "context_token_proxy": round(sum(len(doc.page_content) for doc in documents) / 1.8),
        **metadata,
    }


def average(values):
    values = [value for value in values if value is not None]
    return round(sum(values) / len(values), 6) if values else None


def metrics(rows):
    reference_rows = [row for row in rows if row["expected_documents"]]
    multi = [row for row in reference_rows if len(row["expected_documents"]) > 1]
    keys = [
        "any_at_1",
        "any_at_3",
        "all_at_1",
        "all_at_3",
        "reciprocal_rank",
        "context_recall",
        "context_precision",
    ]
    result = {key: average([row[key] for row in reference_rows]) for key in keys}
    result.update(
        {
            "sample_count": len(rows),
            "reference_bearing_count": len(reference_rows),
            "multi_document_count": len(multi),
            "multi_document_all_at_3": average([row["all_at_3"] for row in multi]),
            "multi_document_context_recall": average([row["context_recall"] for row in multi]),
            "average_context_count": average([row["context_count"] for row in rows]),
            "average_context_characters": average(
                [row["context_characters"] for row in rows]
            ),
            "average_context_token_proxy": average(
                [row["context_token_proxy"] for row in rows]
            ),
            "dynamic_m3_ratio": average(
                [int(row.get("budget", 2) == 3) for row in rows]
            ),
            "diversity_applied_ratio": average(
                [int(row.get("diversity_applied", False)) for row in rows]
            ),
        }
    )
    result["mrr"] = result["reciprocal_rank"]
    return result


def evaluate_cohort(name, samples, store, cfg):
    modes = {"fixed_m2": [], "global_m3": [], "dynamic_m": []}
    timings = []
    for index, sample in enumerate(samples, 1):
        ranked = store.search_documents(
            sample["question"], mode="hybrid", rerank=True, candidate_k=3, limit=3
        )
        trace = store.last_search_trace or {}
        timings.append(trace.get("timings", {}))
        modes["fixed_m2"].append(score(sample, ranked[:2], {"budget": 2}))
        modes["global_m3"].append(score(sample, ranked[:3], {"budget": 3}))
        selected, metadata = dynamic_select(sample["question"], ranked, cfg)
        modes["dynamic_m"].append(score(sample, selected, metadata))
        if index % 25 == 0:
            print(f"{name} {index}/{len(samples)}", flush=True)
    latency_keys = ["embedding_ms", "retrieval_ms", "rerank_ms", "total_ms"]
    latency = {
        key: {
            "mean": average([item.get(key) for item in timings]),
            "p95": round(float(np_percentile([item.get(key) for item in timings], 95)), 6),
        }
        for key in latency_keys
    }
    return {
        "sample_count": len(samples),
        "candidate_k": 3,
        "modes": {
            mode: {"metrics": metrics(rows), "results": rows} for mode, rows in modes.items()
        },
        "shared_retrieval_latency_ms": latency,
        "latency_note": "All three modes slice the same K=3 candidate pool; timings are shared. Reranker pair cache was enabled for resumability, so these are evaluation-process timings, not uncached production latency.",
    }


def np_percentile(values, percentile):
    values = sorted(float(value) for value in values if value is not None)
    if not values:
        return 0.0
    position = (len(values) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def main():
    cfg = Config()
    assert cfg.RETRIEVAL_K == 3
    retrieval = load("RETAILRAG_SCALE_RETRIEVAL_SET.json")["samples"]
    rag = load("RETAILRAG_RAG_EVALUATION_SET.json")["samples"]
    store = VectorStore()
    store.reranker = CachedReranker(store.reranker, cfg)
    output = {
        "status": "COMPLETED",
        "design": {
            "candidate_k": 3,
            "simple_budget": 2,
            "complex_budget": 3,
            "complexity_source": "deterministic concept, condition, clause and arithmetic signals",
            "relevance_floor": "max(legacy evidence threshold, top1 reranker score * 0.02)",
            "selection": "relevance first; prefer a new category only when its score is at least 80% of the best eligible extra candidate",
        },
        "retrieval_235": evaluate_cohort("retrieval_235", retrieval, store, cfg),
        "rag_350": evaluate_cohort("rag_350", rag, store, cfg),
    }
    save("RETAILRAG_PHASE6_DYNAMIC_M_EVALUATION.json", output)
    compact = {
        cohort: {
            mode: value["metrics"]
            for mode, value in output[cohort]["modes"].items()
        }
        for cohort in ("retrieval_235", "rag_350")
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
