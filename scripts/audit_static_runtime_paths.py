"""Audit strategy and multi-turn code paths without claiming online execution."""

import ast
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from base import Config
from rag_qa.core.prompts import RAGPrompts


def contains_all(text, values):
    return all(value in text for value in values)


def main():
    strategy_source = (ROOT / "rag_qa/core/strategy_selector.py").read_text(encoding="utf-8")
    rag_source = (ROOT / "rag_qa/core/new_rag_system.py").read_text(encoding="utf-8")
    main_source = (ROOT / "new_main.py").read_text(encoding="utf-8")
    strategy_tree = ast.parse(strategy_source)
    rag_tree = ast.parse(rag_source)
    main_tree = ast.parse(main_source)

    strategies = ["直接检索", "假设问题检索", "子查询检索", "回溯问题检索"]
    method_names = {
        node.name
        for tree in (strategy_tree, rag_tree, main_tree)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    prompt = str(RAGPrompts.rag_prompt().template)
    config = Config()

    strategy_checks = {
        "selector_declares_four_strategies": contains_all(strategy_source, strategies),
        "retail_examples_in_selector_prompt": contains_all(
            strategy_source, ["Nova Retail", "无小票退货", "会员", "闭店", "促销订单"]
        ),
        "direct_branch_present": "hybrid_search_with_rerank" in rag_source,
        "hyde_branch_present": "_retrieve_with_hyde" in method_names,
        "subquery_branch_present": "_retrieve_with_subqueries" in method_names,
        "backtracking_branch_present": "_retrieve_with_backtracking" in method_names,
        "branch_dispatch_names_present": contains_all(rag_source, strategies[1:]),
    }
    multiturn_checks = {
        "conversation_insert_present": "INSERT INTO conversations" in main_source,
        "recent_history_limit_five": "LIMIT %s" in main_source and "(session_id, 5)" in main_source,
        "history_forwarded_to_rag": "history=history" in main_source,
        "prompt_has_history_slot": "{history}" in prompt,
        "rag_truncates_to_five": "history[-5:]" in rag_source,
        "retrieval_receives_current_query": "retrieve_and_merge(query" in rag_source,
        "history_not_used_to_rewrite_retrieval_query": "retrieve_and_merge(history" not in rag_source,
    }
    examples = {
        "direct": "无小票退货需要什么凭证？",
        "hyde": "怎样做才能让会员觉得积分更好用？",
        "subquery": "比较开店和闭店时现金、设备、库存分别要检查什么？",
        "backtracking": "促销订单部分退货又用了赠品和券该怎么算？",
        "multiturn": ["顾客买的商品想退货", "买了5天，有小票", "还是活动期间买的"],
    }
    has_key = bool(config.DASHSCOPE_API_KEY.strip())
    result = {
        "status": "STATIC_PASS" if all(strategy_checks.values()) and all(multiturn_checks.values()) else "FAIL",
        "scope": "code and prompt inspection only; no external service result is simulated",
        "qwen_runtime_status": "CONFIGURED_UNVERIFIED" if has_key else "BLOCKED_API_KEY",
        "strategy_runtime_status": "NOT_RUN" if has_key else "BLOCKED_API_KEY",
        "strategy_checks": strategy_checks,
        "multiturn_checks": multiturn_checks,
        "observed_multiturn_behavior": {
            "mysql_history_write": "implemented; online verification requires MySQL",
            "final_prompt_history": "last five Q/A pairs are included",
            "retrieval_query": "the current turn only; history-aware query rewriting is not implemented",
        },
        "examples_for_online_run": examples,
    }
    output = ROOT / "RETAILRAG_STATIC_RUNTIME_AUDIT.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] == "FAIL":
        raise RuntimeError("Static runtime path audit failed")


if __name__ == "__main__":
    main()
