"""Evaluate FAQ BM25 behavior from the checked-in CSV without external services."""

import csv
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mysql_qa.retrieval.bm25_search import BM25Search


EXACT_QUERIES = [
    "会员等级怎么划分？",
    "积分什么时候过期？",
    "会员日是哪天？",
    "新客券有效期多久？",
    "普通商品几天内可以退？",
    "没有纸质小票能退货吗？",
    "开店前先检查什么？",
    "现金差异超过多少要升级？",
    "POS 支付结果不明怎么办？",
    "冷藏商品验收温度是多少？",
]

PARAPHRASE_QUERIES = [
    "Nova Retail 会员要消费多少才能升到银卡或金卡？",
    "消费获得的点数最晚到什么时候失效？",
    "每个月哪一天是会员专享日？",
    "新人优惠券领到后能用多少天？",
    "没质量问题的普通商品购买后多久还能退？",
    "购物小票丢了还能申请退货处理吗？",
    "门店每天开门营业之前要检查哪些事项？",
    "收银钱款差多少金额就要向区域上报？",
    "顾客已经被扣款但收银系统订单没成功该怎么办？",
    "冷藏货品到店验收允许的温度范围是多少？",
]

NON_FAQ_QUERIES = [
    "法国的首都是什么？",
    "如何用 Python 读取 Excel 文件？",
    "员工请假需要谁审批？",
    "供应商合同在哪里归档？",
    "门店空调坏了联系哪个维修商？",
    "今年公司年会什么时候举行？",
    "如何申请办公电脑？",
    "食堂午餐几点开始？",
    "商品采购预算由谁审批？",
    "总部停车位怎么申请？",
]


class MemoryCache:
    def __init__(self):
        self.data = {}

    def faq_key(self, suffix):
        return "retailrag:v1:" + suffix

    def get_data(self, key):
        return self.data.get(key)

    def set_data(self, key, value):
        self.data[key] = value

    def get_answer(self, query):
        return self.data.get("answer:" + query)

    def set_answer(self, query, answer):
        self.data["answer:" + query] = answer

    def clear_faq_cache(self):
        self.data.clear()
        return 0


class CsvFAQStore:
    def __init__(self, rows):
        self.answers = {row["问题"]: row["答案"] for row in rows}

    def fetch_questions(self):
        return [(question,) for question in self.answers]

    def fetch_answer(self, question):
        return self.answers.get(question)


def main():
    logging.disable(logging.INFO)
    csv_path = ROOT / "mysql_qa" / "data" / "retail_faq.csv"
    with csv_path.open(encoding="utf-8-sig") as stream:
        faq_rows = list(csv.DictReader(stream))
    search = BM25Search(MemoryCache(), CsvFAQStore(faq_rows))
    test_groups = {
        "exact": EXACT_QUERIES,
        "paraphrase": PARAPHRASE_QUERIES,
        "non_faq": NON_FAQ_QUERIES,
    }
    results = []
    for kind, queries in test_groups.items():
        for query in queries:
            details = search.search_with_details(query, threshold=0.85)
            results.append({"kind": kind, **details})

    csv_output = ROOT / "RETAILRAG_FAQ_OFFLINE_RESULTS.csv"
    fields = [
        "kind", "query", "top1_question", "raw_score", "score", "threshold",
        "above_threshold", "hit", "cached", "answer", "need_rag",
    ]
    with csv_output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    summary = {
        "status": "STATIC_PASS",
        "reason": "Uses real Jieba/BM25 and checked-in FAQ data, but not MySQL/Redis services",
        "threshold": 0.85,
        "exact_hits": sum(row["hit"] for row in results if row["kind"] == "exact"),
        "paraphrase_hits": sum(row["hit"] for row in results if row["kind"] == "paraphrase"),
        "non_faq_rejected": sum(not row["hit"] for row in results if row["kind"] == "non_faq"),
        "total": len(results),
    }
    (ROOT / "RETAILRAG_FAQ_OFFLINE_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for row in results:
        score = "n/a" if row["score"] is None else f"{row['score']:.6f}"
        print(
            f"{row['kind']:10s} hit={str(row['hit']):5s} score={score} "
            f"top1={row['top1_question']} | {row['query']}"
        )


if __name__ == "__main__":
    main()
