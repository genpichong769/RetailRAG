"""Call Qwen with multiple natural examples for all four retrieval strategies."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from base import Config
from rag_qa.core.strategy_selector import StrategySelector


def main():
    config = Config()
    if not config.DASHSCOPE_API_KEY.strip():
        result = {"status": "BLOCKED_API_KEY", "credential_value_logged": False}
    else:
        cases = [
            ("direct_1", "无小票退货需要什么凭证？", "直接检索"),
            ("direct_2", "会员积分的有效期是多久？", "直接检索"),
            ("hyde_1", "怎样做才能让会员觉得积分更好用？", "假设问题检索"),
            ("hyde_2", "顾客总觉得优惠太复杂，门店可以从哪些方向改善体验？", "假设问题检索"),
            ("subquery_1", "比较开店和闭店时现金、设备、库存分别要检查什么？", "子查询检索"),
            ("subquery_2", "分别说明 POS 故障、冷柜报警和标签机错价应如何处理？", "子查询检索"),
            ("backtracking_1", "促销订单部分退货又用了赠品和券该怎么算？", "回溯问题检索"),
            ("backtracking_2", "会员日买的生鲜没有小票，现在还要部分退款，应该依据哪些基础规则？", "回溯问题检索"),
        ]
        selector = StrategySelector()
        rows = []
        for case_id, query, expected in cases:
            started = time.perf_counter()
            actual = selector.select_strategy(query)
            rows.append(
                {
                    "case": case_id,
                    "query": query,
                    "expected": expected,
                    "actual": actual,
                    "raw_response": selector.last_raw_response,
                    "call_status": selector.last_call_status,
                    "error": selector.last_error,
                    "matched": selector.last_call_status == "PASS" and actual == expected,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                }
            )
        if all(row["call_status"] == "PASS" for row in rows):
            overall_status = "PASS"
        elif any(row["call_status"].startswith("BLOCKED") for row in rows):
            overall_status = "BLOCKED_NETWORK_OR_ENDPOINT"
        else:
            overall_status = "FAIL"
        result = {
            "status": overall_status,
            "scope": "real StrategySelector calls to the configured Qwen endpoint",
            "model": config.LLM_MODEL,
            "credential_source": "host process environment",
            "credential_value_logged": False,
            "expected_matches": sum(row["matched"] for row in rows),
            "parsed_strategy_coverage": sorted({row["actual"] for row in rows}),
            "results": rows,
        }
    output = ROOT / "RETAILRAG_STRATEGY_SELECTOR_RESULTS.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))
    if result["status"] == "FAIL":
        raise RuntimeError("One or more strategy selections did not match the expected branch")


if __name__ == "__main__":
    main()
