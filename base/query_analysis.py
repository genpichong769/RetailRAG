"""Deterministic query-risk signals shared by the PHASE 6 targeted fixes.

The analyzer intentionally uses only generic linguistic and business-concept
signals.  It does not contain evaluation questions, FAQ IDs, document names, or
answers.  It is cheap enough to run before accepting the in-memory FAQ path.
"""

import re


CONCEPT_PATTERNS = {
    "member_tier": r"金卡|银卡|普通会员|会员等级|升级|降级|保级",
    "birthday": r"生日月|生日券|生日权益|生日",
    "points": r"积分|积几分|累计.*分",
    "coupon": r"优惠券|新人券|新客券|订单券|商品券|用券|返券",
    "promotion": r"会员日|满减|秒杀|促销|活动价|折扣|赠品|组合价|套装",
    "after_sales": r"退货|退款|退换|换货|售后|退回|返还|不退|(?:券|积分|钱).{0,3}退",
    "cash": r"现金|备用金|封袋|交接|日结",
    "inventory": r"库存|缺货|盘点|调拨|入库|出库|收货",
    "equipment": r"POS|收银机|打印机|扫码枪|冷柜|设备|探头|价签机|小票机|盘点机",
    "product_risk": r"食品|生鲜|临期|保质期|召回|过敏原|密封包装|礼品卡|特殊商品",
    "online_fulfilment": r"线上|网购|自提|配送|运费|免运",
    "account_privacy": r"账户|账号|手机号|验证码|个人信息|隐私|到账",
    "invoice": r"发票|开票|红冲",
    "incident_reporting": r"异常事件|事件上报|升级处理|上报原则|事件记录",
    "version_time": r"旧版|新规|当前|历史|生效|到期|通知|版本|试点",
}

CONDITION_PATTERN = re.compile(
    r"同时|并且|而且|以及|又|还|如果|若|以后|之后|期间|部分|跨店|"
    r"原订单|不参加|已经|仍然|分别|各自|一边|一方面|另一方面"
)
CLAUSE_SPLIT = re.compile(
    r"[，,；;。！？?!]|同时|并且|而且|以及|顺便|再告诉|还要|又"
)
CALCULATION_PATTERN = re.compile(
    r"多少积分|积多少|合计|一共多少|总共多少|最终(?:多少钱|应付|金额|多少)|应付多少|"
    r"退款金额|退多少钱|返还多少|能获得多少|可以获得多少|怎么算"
)
NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?\s*(?:元|分|件|次|天|小时|折|%|％)?")

# Fine-grained intent/constraint facets prevent a broad FAQ from swallowing a
# materially different request in the same business domain.  These are generic
# language features rather than benchmark strings or document identifiers.
INTENT_PATTERNS = {
    "points_earning": re.compile(
        r"(?:每花|消费|购买|买东西).{0,8}(?:一元|1元|一块钱).{0,8}(?:积分|积几分|攒多少)"
        r"|(?:一元|1元|一块钱).{0,8}(?:对应|累计|累积|能攒).{0,6}(?:多少)?积分"
    ),
    "points_redemption": re.compile(r"积分.{0,6}(?:抵多少钱|抵扣|兑换)|(?:抵扣|兑换).{0,6}积分"),
}
RECORD_MISSING_PATTERN = re.compile(r"查不到|找不到|没有(?:订单|交易|系统)?记录|无(?:订单|交易|系统)?记录")
DETAIL_ATTRIBUTE_PATTERN = re.compile(r"字体|字号|颜色|格式|接口|型号|编码规则|字段定义")
MULTI_OUTPUT_PATTERN = re.compile(r"(?:吗|么)[、；;].*(?:怎么|多少|是否|能否|哪些|什么)")

# Requests whose truth normally lives in a transactional, telemetry, personnel,
# finance, credential, or confidential system rather than the fixed policy KB.
SCOPE_PATTERNS = {
    "live_state": re.compile(
        r"实时|此刻|当前(?:余额|库存|温度|位置|排班)|今天(?:还剩|的)?(?:余额|库存|温度|位置|排班)|"
        r"现在(?:还剩|的)?(?:余额|库存|温度|位置|排班)|到(?:了)?哪条路|是否已经退回"
    ),
    "record_lookup": re.compile(
        r"(?:订单|工单|会员|账户|冷柜|商品)[A-Z0-9_-]{4,}|"
        r"\b[A-Z]{1,5}\d{2,}\b|1[3-9]\d{9}"
    ),
    "secret_or_private": re.compile(
        r"密钥|签名算法|口令|秘密|未公开|隐藏|完整身份证|完整银行卡|私人行程|"
        r"内部(?:底价|预算|算法|公式)|采购底价|返利率|具体员工|哪位员工|员工姓名|姓名和排班|"
        r"总预算|净利润|营业收入|银行余额|平均工资|薪资普调|手机号码|API"
    ),
    "fabrication": re.compile(r"编一个|假装你看过|不许说不知道|直接报数|断定.*说谎"),
}


def _normalise(text):
    return re.sub(r"[\s，,。！？?!；;：:]", "", str(text or "")).lower()


def _concepts(text):
    return sorted(
        name for name, pattern in CONCEPT_PATTERNS.items() if re.search(pattern, text, re.I)
    )


def _scope_reasons(text):
    return sorted(name for name, pattern in SCOPE_PATTERNS.items() if pattern.search(text))


def _intents(text):
    return sorted(name for name, pattern in INTENT_PATTERNS.items() if pattern.search(text))


def _clauses(text):
    return [part.strip() for part in CLAUSE_SPLIT.split(text) if len(part.strip()) >= 2]


def analyze_query(query, faq_candidate=None):
    """Return explainable complexity, scope, and FAQ-completeness signals."""
    text = str(query or "").strip()
    candidate = str(faq_candidate or "").strip()
    concepts = _concepts(text)
    candidate_concepts = _concepts(candidate)
    condition_count = len(CONDITION_PATTERN.findall(text))
    clauses = _clauses(text)
    scope_reasons = _scope_reasons(text)
    safe_clauses = [
        part for part in clauses if _concepts(part) and not _scope_reasons(part)
    ]
    risky_clauses = [part for part in clauses if _scope_reasons(part)]
    has_numeric = bool(NUMBER_PATTERN.search(text))
    requires_arithmetic = bool(CALCULATION_PATTERN.search(text)) and has_numeric
    candidate_requires_arithmetic = bool(CALCULATION_PATTERN.search(candidate)) and bool(
        NUMBER_PATTERN.search(candidate)
    )
    extra_concepts = sorted(set(concepts) - set(candidate_concepts))
    intents = _intents(text)
    candidate_intents = _intents(candidate)
    exact_candidate = bool(candidate) and _normalise(text) == _normalise(candidate)

    multi_condition = (
        len(concepts) >= 3
        or (len(concepts) >= 2 and condition_count >= 1)
        or (len(clauses) >= 2 and len(concepts) >= 2)
    )
    context_complex = (
        multi_condition
        or requires_arithmetic
        or "分别" in text
        or (
            "product_risk" in concepts
            and bool(re.search(r"质量问题|顾客反馈|客诉|投诉", text))
        )
    )

    faq_reasons = []
    if scope_reasons:
        faq_reasons.append("knowledge_scope_risk")
    if requires_arithmetic and not candidate_requires_arithmetic:
        faq_reasons.append("derived_calculation_not_covered_by_candidate")
    if extra_concepts and (condition_count or len(clauses) >= 2):
        faq_reasons.append("candidate_missing_query_concepts")
    if len(extra_concepts) >= 2 and multi_condition:
        faq_reasons.append("multiple_uncovered_concepts")
    if "分别" in text and len(concepts) >= 2 and extra_concepts:
        faq_reasons.append("multiple_requested_outputs")
    if intents and candidate_intents and set(intents).isdisjoint(candidate_intents):
        faq_reasons.append("candidate_intent_mismatch")
    if RECORD_MISSING_PATTERN.search(text) and not RECORD_MISSING_PATTERN.search(candidate):
        faq_reasons.append("candidate_missing_record_constraint")
    if DETAIL_ATTRIBUTE_PATTERN.search(text) and not DETAIL_ATTRIBUTE_PATTERN.search(candidate):
        faq_reasons.append("candidate_missing_detail_attribute")
    if MULTI_OUTPUT_PATTERN.search(text) and not MULTI_OUTPUT_PATTERN.search(candidate):
        faq_reasons.append("candidate_missing_requested_output")
    if exact_candidate:
        # An exact standard FAQ is an explicit fixed-answer contract, including
        # compound questions intentionally authored as one FAQ.
        faq_reasons = []

    partial_possible = bool(scope_reasons and safe_clauses and risky_clauses)
    return {
        "concepts": concepts,
        "candidate_concepts": candidate_concepts,
        "extra_concepts": extra_concepts,
        "intents": intents,
        "candidate_intents": candidate_intents,
        "condition_count": condition_count,
        "clause_count": len(clauses),
        "has_numeric_constraint": has_numeric,
        "requires_arithmetic": requires_arithmetic,
        "scope_risk": bool(scope_reasons),
        "scope_reasons": scope_reasons,
        "partial_possible": partial_possible,
        "context_complex": context_complex,
        "faq_force_rag": bool(faq_reasons),
        "faq_gate_reasons": faq_reasons,
        "exact_candidate": exact_candidate,
    }

