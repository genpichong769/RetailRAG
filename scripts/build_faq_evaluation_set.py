"""Build an independent 280-sample FAQ evaluation set with Qwen-assisted rewrites."""

import argparse
import csv
import json
import sys
from pathlib import Path

import httpx
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from base import Config

FAQ_PATH = ROOT / "mysql_qa" / "data" / "retail_faq.csv"
OUTPUT_PATH = ROOT / "RETAILRAG_FAQ_EVALUATION_SET.json"

NON_FAQ = [
    "法国的首都是什么？",
    "如何计算圆柱体的体积？",
    "今天北京天气怎么样？",
    "请写一首关于秋天的诗。",
    "Python 列表和元组有什么区别？",
    "太阳为什么会发光？",
    "怎样给家里的路由器改密码？",
    "一公斤等于多少克？",
    "帮我制定一周健身计划。",
    "地球到月球大约有多远？",
    "写正式邮件时主题行应该怎么写？",
    "什么是通货膨胀？",
    "番茄炒蛋怎么做？",
    "为什么冬天白昼比较短？",
    "请解释勾股定理。",
    "旅行时如何选择登机箱？",
    "常见的英语时态有哪些？",
    "电脑蓝屏通常有哪些原因？",
    "猫一天通常睡多久？",
    "水在标准大气压下多少度沸腾？",
]

# 逐条人工定义：与对应 FAQ 共享业务词，但询问 FAQ 与其他正式 FAQ 都未提供的信息。
HARD_NEGATIVES = [
    "会员等级的实体卡面颜色分别是什么？",
    "会员升级时页面的庆祝动画播放多久？",
    "金卡会员的保级条件是什么？",
    "会员积分明细页默认用什么字体？",
    "积分过期提醒短信的发送号码是多少？",
    "积分商城的实物礼品目录每月几号更新？",
    "退款后积分变动记录在页面上显示什么颜色？",
    "会员更换手机号的工单平均几个小时办结？",
    "会员账户合并完成后会播放哪个提示音？",
    "生日券的券面背景图由哪个团队设计？",
    "会员日门店应播放哪首背景音乐？",
    "优惠结算完成后小票上的促销字段用什么字号？",
    "优惠券条码一共由多少位数字组成？",
    "满减活动海报的标题必须使用哪种字体？",
    "新客券包在会员中心里的卡片背景是什么颜色？",
    "整单取消后退券通知由哪个短信号码发出？",
    "赠品缺货登记表的纸张规格是 A4 还是 A5？",
    "赠品退回时应放入什么颜色的周转箱？",
    "春季会员日门店海报的成品尺寸是多少？",
    "普通商品退货柜台每天几点停止取号？",
    "质量问题退货的默认检验员工号是多少？",
    "无小票查询页的搜索按钮在屏幕左边还是右边？",
    "无小票人工核查工单的平均处理时长是多少？",
    "生鲜退货拍摄的问题照片最低要多少像素？",
    "食品质量事件上报表的文件名格式是什么？",
    "促销订单部分退货的售后单用多宽的打印纸？",
    "部分退货未退券时页面会显示什么颜色的标识？",
    "退款到账短信的发送号码是多少？",
    "特殊商品价签上的例外图标是什么颜色？",
    "开店前门店应播放哪段迎宾音乐？",
    "闭店后最后一个照明开关位于门店什么位置？",
    "收银交接使用的现金封袋是什么材质？",
    "现金差异记录表的表头默认是什么颜色？",
    "现金差异升级时应拨打哪个内线分机？",
    "每日盘点任务在终端里默认按什么顺序排列？",
    "月度全盘当天系统默认几点锁定收货功能？",
    "盘点差异调整页的审批按钮是什么颜色？",
    "盘点差异升级报表必须保存成哪种电子表格格式？",
    "激烈客诉处理区应设在门店的哪个楼层？",
    "POS 机的指定硬件品牌和型号是什么？",
    "POS 日结失败提示框的背景色必须是什么？",
    "扫码枪的备用数据线具体是什么型号？",
    "条码扫出错误商品时告警弹窗使用什么字体？",
    "标签打印机需要更换哪个型号的碳带？",
    "盘点机离线任务最多能占用多少本地存储空间？",
    "盘点机异店数据警告页的图标是什么样式？",
    "冷藏验收用温度计的指定品牌是什么？",
    "冷冻柜温度传感器的指定型号是什么？",
    "冷链商品入库使用的周转车必须是什么颜色？",
    "过期商品隔离时使用的专用袋是什么颜色？",
    "价签与 POS 价格不一致时更换的价签用什么字体？",
    "供应商送货车辆进店验收时车身应是什么颜色？",
]


def _parse_json_response(text):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
    parsed = json.loads(cleaned)
    return parsed.get("items", parsed) if isinstance(parsed, dict) else parsed


def _generate_batch(client, model, rows, all_questions):
    payload = [
        {
            "faq_id": row["faq_id"],
            "category": row["知识类别"],
            "question": row["问题"],
            "answer": row["答案"],
        }
        for row in rows
    ]
    prompt = f"""
你正在为虚构连锁零售知识库构建独立 FAQ 评测集。针对输入中的每条 FAQ：
1. 写 3 条自然、口语化、语义等价的中文改写，不得复制原句，不得泄露答案。
2. 写 1 条 hard negative：与原问题共享部分词语或业务对象，但询问的是不同信息，不能由该 FAQ 的答案回答，也不要与其他正式 FAQ 同义。
3. 不添加事实，不改变原 FAQ 意图。

全部正式 FAQ 问题如下，用于避免 hard negative 与其他 FAQ 同义：
{json.dumps(all_questions, ensure_ascii=False)}

当前批次：
{json.dumps(payload, ensure_ascii=False)}

只返回严格 JSON 数组。每项格式：
{{"faq_id":"faq_001","paraphrases":["...","...","..."],"hard_negative":"..."}}
""".strip()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是严谨的中文检索评测数据设计员。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        timeout=60,
    )
    return _parse_json_response(response.choices[0].message.content)


def _repair_duplicates(client, model, samples):
    seen = set()
    duplicates = []
    for sample in samples:
        query = sample["query"]
        if query in seen:
            duplicates.append(
                {
                    "id": sample["id"],
                    "question_type": sample["question_type"],
                    "expected_behavior": sample["expected_behavior"],
                    "expected_question": sample.get("expected_question"),
                    "current_query": query,
                }
            )
        else:
            seen.add(query)
    if not duplicates:
        return samples

    prompt = f"""
下列 FAQ 评测样本与其他样本文本重复。请为每条只生成一个新 query：
- paraphrase 必须与 expected_question 语义等价。
- hard_negative/non_faq 必须继续应当拒绝，不得变成 expected_question 的同义问。
- 新 query 不得出现在已有问题列表中，不得添加编号或括号备注来制造唯一性。

待替换样本：
{json.dumps(duplicates, ensure_ascii=False)}

已有问题：
{json.dumps(sorted(seen), ensure_ascii=False)}

只返回严格 JSON 数组：[{{"id":"...","query":"..."}}]
""".strip()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是严谨的中文评测数据去重编辑。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
        timeout=60,
    )
    replacements = {
        item["id"]: str(item["query"]).strip()
        for item in _parse_json_response(response.choices[0].message.content)
    }
    for sample in samples:
        if sample["id"] in replacements:
            sample["query"] = replacements[sample["id"]]
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reuse-paraphrases",
        action="store_true",
        help="Reuse the already generated paraphrases and only rebuild negatives.",
    )
    args = parser.parse_args()
    config = Config()
    if not config.DASHSCOPE_API_KEY:
        raise RuntimeError("DASHSCOPE_API_KEY is required to build semantic rewrites")
    with FAQ_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 52:
        raise RuntimeError(f"Expected 52 FAQ rows, got {len(rows)}")
    for index, row in enumerate(rows, 1):
        row["faq_id"] = f"faq_{index:03d}"

    client = OpenAI(
        api_key=config.DASHSCOPE_API_KEY,
        base_url=config.DASHSCOPE_BASE_URL,
        http_client=httpx.Client(trust_env=False, timeout=60),
    )
    all_questions = [row["问题"] for row in rows]
    generated = {}
    if args.reuse_paraphrases and OUTPUT_PATH.exists():
        existing = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))["samples"]
        for row in rows:
            faq_id = row["faq_id"]
            generated[faq_id] = {
                "faq_id": faq_id,
                "paraphrases": [
                    sample["query"]
                    for sample in existing
                    if sample["expected_faq_id"] == faq_id
                    and sample["question_type"] == "paraphrase"
                ],
            }
    else:
        for start in range(0, len(rows), 13):
            batch = rows[start : start + 13]
            for item in _generate_batch(client, config.LLM_MODEL, batch, all_questions):
                generated[item["faq_id"]] = item

    samples = []
    for row in rows:
        faq_id = row["faq_id"]
        item = generated.get(faq_id)
        if not item or len(item.get("paraphrases", [])) != 3:
            raise RuntimeError(f"Incomplete generated item: {faq_id}")
        base = {
            "expected_faq_id": faq_id,
            "expected_question": row["问题"],
            "category": row["知识类别"],
        }
        samples.append(
            {
                "id": f"{faq_id}_exact",
                "query": row["问题"],
                **base,
                "expected_behavior": "answer",
                "question_type": "exact",
            }
        )
        for index, query in enumerate(item["paraphrases"], 1):
            samples.append(
                {
                    "id": f"{faq_id}_paraphrase_{index}",
                    "query": str(query).strip(),
                    **base,
                    "expected_behavior": "answer",
                    "question_type": "paraphrase",
                }
            )
        samples.append(
            {
                "id": f"{faq_id}_hard_negative",
                "query": HARD_NEGATIVES[int(faq_id.split("_")[1]) - 1],
                "expected_faq_id": None,
                "expected_question": None,
                "category": row["知识类别"],
                "expected_behavior": "reject",
                "question_type": "hard_negative",
                "confusable_with": faq_id,
            }
        )
    for index, query in enumerate(NON_FAQ, 1):
        samples.append(
            {
                "id": f"non_faq_{index:03d}",
                "query": query,
                "expected_faq_id": None,
                "expected_question": None,
                "category": "general",
                "expected_behavior": "reject",
                "question_type": "non_faq",
            }
        )

    samples = _repair_duplicates(client, config.LLM_MODEL, samples)
    queries = [sample["query"] for sample in samples]
    if len(samples) != 280 or len(set(queries)) != len(queries):
        raise RuntimeError(
            f"Dataset validation failed: samples={len(samples)}, unique={len(set(queries))}"
        )
    output = {
        "status": "PASS",
        "model_used_for_rewrites": config.LLM_MODEL,
        "faq_count": 52,
        "sample_count": len(samples),
        "counts": {
            "exact": 52,
            "paraphrase": 156,
            "hard_negative": 52,
            "non_faq": 20,
        },
        "samples": samples,
    }
    OUTPUT_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: output[key] for key in ("status", "faq_count", "sample_count", "counts")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
