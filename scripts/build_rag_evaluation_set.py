"""Build the independent 190-case PHASE 5 RAG evaluation set from live source docs."""

import csv
import json
import re
import sys
import time
from pathlib import Path

import httpx
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from base import Config

OUT = ROOT / "RETAILRAG_RAG_EVALUATION_SET.json"
FAQ_EVAL = ROOT / "RETAILRAG_FAQ_EVALUATION_SET.json"
FAQ_CSV = ROOT / "mysql_qa" / "data" / "retail_faq.csv"


def strip_front_matter(text):
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            return parts[2].strip()
    return text


def parse_json(text):
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("model did not return a JSON object")
    return json.loads(match.group(0))


def ask(client, model, prompt, retries=3):
    last = None
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是严谨的中文零售知识库评测集设计员，只输出 JSON。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                timeout=60,
            )
            return parse_json(response.choices[0].message.content)
        except Exception as exc:
            last = exc
            time.sleep(1 + attempt)
    raise RuntimeError(f"Qwen generation failed: {last}")


def load_documents():
    docs = {}
    categories = {}
    for path in sorted((ROOT / "rag_qa" / "data").glob("*_data/*.md")):
        text = path.read_text(encoding="utf-8")
        if "status: superseded" in text[:300]:
            continue
        docs[path.stem] = strip_front_matter(text)
        categories[path.stem] = path.parent.name.replace("_data", "")
    return docs, categories


MULTI_PAIRS = [
    ("member_adjustment", "account_registration"),
    ("coupon_rules", "promotion_stacking"),
    ("promotion_stacking", "full_reduction"),
    ("promo_partial_return", "gift_rules"),
    ("return_exchange", "special_goods"),
    ("no_receipt", "after_sales_update"),
    ("closing_checklist", "cash_handover"),
    ("opening_checklist", "daily_inventory"),
    ("daily_inventory", "inventory_variance"),
    ("monthly_inventory", "inventory_terminal"),
    ("cold_chain", "receiving_inspection"),
    ("cold_cabinet", "cold_chain"),
    ("display_price", "label_printer"),
    ("pos_troubleshooting", "refund_channels"),
    ("privacy_security", "incident_escalation"),
    ("knowledge_governance", "spring_member_day"),
    ("member_day_campaign", "spring_member_day"),
    ("expiry_loss", "food_fresh"),
]

UNANSWERABLE = [
    "Nova Retail 今年营业收入是多少？",
    "公司创始人的出生日期是哪天？",
    "明年计划新开多少家海外门店？",
    "公司股票代码和最新股价是什么？",
    "总部办公楼的产权证编号是多少？",
    "现任首席财务官的手机号码是什么？",
    "所有门店员工的平均工资是多少？",
    "今年广告投放预算具体是多少万元？",
    "竞争对手下季度的促销计划是什么？",
    "公司银行账户余额现在有多少？",
    "董事会下次会议的未公开议程是什么？",
    "新一代收银系统的采购合同金额是多少？",
    "门店监控录像默认保存多少年？",
    "品牌代言人的合同什么时候到期？",
    "供应商甲的法人身份证号码是什么？",
    "公司去年的净利润精确到元是多少？",
    "内部尚未发布的春节奖金方案是什么？",
    "哪个员工本月绩效排名最后？",
    "未来三年的并购目标企业有哪些？",
    "公司服务器管理员密码是什么？",
]

VERSION_QUESTIONS = [
    "现在金卡会员每消费一元累计多少积分？",
    "当前生效的金卡积分倍率是多少？",
    "2026 年金卡消费积分怎么计算？",
    "请按最新制度说明金卡的积分累计比例。",
    "金卡用户今天消费 100 元基础可积多少分？",
    "最新金卡积分口径是一元一分还是一点五分？",
    "现行规则下金卡会员的基础积分倍数是多少？",
    "目前门店应按什么比例给金卡会员记积分？",
    "旧的一元一积分是否还是金卡现行标准？",
    "金卡积分新规生效后消费金额怎样换算？",
    "现在给金卡顾客补录积分应采用哪个倍率？",
    "按当前版本，金卡消费 20 元的基础积分是多少？",
    "金卡的最新积分累计政策何时开始生效？",
    "2025 年的金卡积分标准现在还能用吗？",
    "新版本金卡积分规则的有效期从哪天开始？",
    "门店系统当前应配置的金卡积分系数是多少？",
    "请说明已经生效的金卡消费积分规则。",
    "会员问金卡现在一元能积几分，应如何回答？",
    "当前有效版本对金卡基础积分有什么调整？",
    "比较旧规则后，现行金卡积分倍率变成了多少？",
]

MULTITURN = [
    ([{"question": "普通商品怎么退？", "answer": "按常规退换货政策核验。"}], "如果是活动商品呢？", "promo_partial_return"),
    ([{"question": "活动商品退货怎么处理？", "answer": "需重算优惠。"}], "用了优惠券怎么办？", "promo_partial_return"),
    ([{"question": "会员换手机号怎么办？", "answer": "先核验身份。"}], "那重复账户呢？", "account_registration"),
    ([{"question": "冷藏商品到货怎么验？", "answer": "检查温度与包装。"}], "温度超限呢？", "cold_chain"),
    ([{"question": "POS 支付状态不明怎么办？", "answer": "先查交易状态。"}], "还是查不到呢？", "pos_troubleshooting"),
    ([{"question": "盘点有差异怎么办？", "answer": "复盘并登记。"}], "超过门槛后呢？", "inventory_variance"),
    ([{"question": "价签错了怎么处理？", "answer": "暂停错误展示并核验。"}], "打印机也打不对呢？", "label_printer"),
    ([{"question": "顾客没有小票能退吗？", "answer": "可以尝试查找订单。"}], "高价值商品呢？", "after_sales_update"),
    ([{"question": "顾客信息能发工作群吗？", "answer": "应最小化处理。"}], "已经发到公开群了呢？", "incident_escalation"),
    ([{"question": "会员日按哪个活动执行？", "answer": "按当前有效通知执行。"}], "春季那次有什么不同？", "spring_member_day"),
]


def add(samples, prefix, question, answer, refs, category, qtype, behavior="answer", history=None):
    item = {
        "id": f"{prefix}_{len([s for s in samples if s['question_type'] == qtype]) + 1:03d}",
        "question": question.strip(),
        "ground_truth": answer.strip(),
        "reference_documents": refs,
        "category": category,
        "question_type": qtype,
        "expected_behavior": behavior,
    }
    if history is not None:
        item["history"] = history
    samples.append(item)


def main():
    config = Config()
    if not config.DASHSCOPE_API_KEY:
        raise RuntimeError("DASHSCOPE_API_KEY is required to build grounded cases")
    client = OpenAI(
        api_key=config.DASHSCOPE_API_KEY,
        base_url=config.DASHSCOPE_BASE_URL,
        http_client=httpx.Client(trust_env=False, timeout=60),
    )
    docs, categories = load_documents()
    generated = {}
    for name, content in docs.items():
        generated[name] = ask(
            client,
            config.LLM_MODEL,
            """根据下方唯一资料生成三组可由资料直接回答的问答。问题不能照抄文档标题；答案必须只使用资料事实。
single 用自然业务问法；keyword 必须含资料里的一个明确术语；paraphrase 要口语化且避免标题词序。
只输出：{"single":{"question":"","answer":""},"keyword":{...},"paraphrase":{...}}
资料：\n""" + content[:7000],
        )

    samples = []
    names = list(docs)
    for name in names[:35]:
        row = generated[name]["single"]
        add(samples, "single", row["question"], row["answer"], [name], categories[name], "single_document")
    for name in names[:20]:
        row = generated[name]["keyword"]
        add(samples, "keyword", row["question"], row["answer"], [name], categories[name], "keyword_exact")
    for name in names[-20:]:
        row = generated[name]["paraphrase"]
        add(samples, "paraphrase", row["question"], row["answer"], [name], categories[name], "semantic_paraphrase")

    for pair_index, (left, right) in enumerate(MULTI_PAIRS):
        count = 1 if pair_index == len(MULTI_PAIRS) - 1 else 2
        result = ask(
            client,
            config.LLM_MODEL,
            f"""根据两份资料生成 {count} 组必须综合两份资料才能完整回答的门店业务问答。问题不能照抄标题，答案只用资料事实。
只输出：{{"items":[{{"question":"","answer":""}}]}}
资料A：\n{docs[left][:5500]}\n资料B：\n{docs[right][:5500]}""",
        )
        for row in result["items"][:count]:
            add(samples, "multi", row["question"], row["answer"], [left, right], categories[left], "multi_document")

    faq_eval = json.loads(FAQ_EVAL.read_text(encoding="utf-8"))
    with FAQ_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        faq_rows = list(csv.DictReader(handle))
    faq_answers = {f"faq_{i:03d}": row["答案"] for i, row in enumerate(faq_rows, 1)}
    faq_cases = [s for s in faq_eval["samples"] if s["question_type"] == "paraphrase"][:30]
    for row in faq_cases:
        faq_id = row["expected_faq_id"]
        add(samples, "faq", row["query"], faq_answers[faq_id], [f"faq:{faq_id}"], "faq", "faq")

    for question in VERSION_QUESTIONS:
        add(
            samples,
            "version",
            question,
            "当前生效的 2.0 版规则自 2026-01-01 起执行：金卡会员基础消费每 1 元累计 1.5 积分；2025 年 1.0 版已被替代。",
            ["member_points_version_v2_2026"],
            "membership",
            "version_conflict",
        )
    for question in UNANSWERABLE:
        add(samples, "unanswerable", question, "当前知识库信息不足。", [], "unknown", "unanswerable", "reject")
    for history, question, ref in MULTITURN:
        row = generated[ref]["single"]
        add(samples, "multiturn", question, row["answer"], [ref], categories[ref], "multiturn", history=history)

    counts = {}
    for row in samples:
        counts[row["question_type"]] = counts.get(row["question_type"], 0) + 1
    expected = {
        "single_document": 35,
        "multi_document": 35,
        "keyword_exact": 20,
        "semantic_paraphrase": 20,
        "faq": 30,
        "version_conflict": 20,
        "unanswerable": 20,
        "multiturn": 10,
    }
    questions = [row["question"] for row in samples]
    titles = {re.sub(r"^#\s+", "", text.splitlines()[0]).strip() for text in docs.values()}
    errors = []
    if len(samples) != 190 or counts != expected:
        errors.append(f"count mismatch: total={len(samples)} counts={counts}")
    if len(set(questions)) != len(questions):
        errors.append("duplicate questions")
    copied_titles = sorted(set(questions) & titles)
    if copied_titles:
        errors.append(f"questions copied titles: {copied_titles}")
    missing = sorted(
        {ref for row in samples for ref in row["reference_documents"] if not ref.startswith("faq:")}
        - set(docs)
    )
    if missing:
        errors.append(f"missing references: {missing}")
    output = {
        "status": "PASS" if not errors else "FAIL",
        "description": "PHASE 5 independent evaluation set; questions are generated from source evidence but do not copy titles.",
        "sample_count": len(samples),
        "counts": counts,
        "validation_errors": errors,
        "samples": samples,
    }
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: output[k] for k in ("status", "sample_count", "counts", "validation_errors")}, ensure_ascii=False, indent=2))
    if errors:
        raise RuntimeError(errors)


if __name__ == "__main__":
    main()
