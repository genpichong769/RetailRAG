"""Binary BERT classifier with a conservative RetailRAG safety route."""

import os
import re

import torch
from transformers import BertForSequenceClassification, BertTokenizer

from base import logger


class QueryClassifier:
    """Classify public knowledge vs internal retail knowledge.

    BERT remains the primary classifier. A keyword rule can only upgrade a result
    from ``general`` to ``internal``; it can never downgrade an internal result.
    """

    INTERNAL_PATTERNS = (
        r"会员|积分|点数|等级|成长值|金卡|银卡|生日礼|生日券|会员日|账户合并|账号合并",
        r"优惠券|满减|促销|折扣|赠品|活动价|活动规则|新客券|新人券|叠加",
        r"退货|换货|退款|售后|小票|纸质凭证|价保|拒收",
        r"门店|开店|闭店|交班|交接班|盘点|库存|收银|钱箱|现金差异|客诉|店长",
        r"POS|扫码枪|扫码设备|标签打印机|标签机|盘点机|盘点终端|设备故障",
        r"冷链|冷柜|收货验收|保质期|临期|储存|陈列|价签|商品编码|SKU|损耗|报损",
        r"内部通知|内部制度|审批|升级上报|区域运营|值班经理|SOP|流程",
        r"Nova\s*Retail|诺瓦零售",
    )

    def __init__(self, model_path="retail-query-classifier"):
        self.model_path = model_path
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.label_map = {"general": 0, "internal": 1}
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)
        self.model.to(self.device)
        self.model.eval()
        logger.info("Retail 二分类 BERT 已加载 (%s)，设备: %s", model_path, self.device)

    @classmethod
    def matched_internal_rule(cls, query):
        normalized = re.sub(r"\s+", "", query or "")
        for pattern in cls.INTERNAL_PATTERNS:
            match = re.search(pattern, normalized, re.IGNORECASE)
            if match:
                return match.group(0)
        return None

    @classmethod
    def is_obviously_internal(cls, query):
        return cls.matched_internal_rule(query) is not None

    def predict_model(self, query):
        encoding = self.tokenizer(
            query,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        )
        encoding = {key: value.to(self.device) for key, value in encoding.items()}
        with torch.no_grad():
            probabilities = torch.softmax(self.model(**encoding).logits, dim=1)[0]
        prediction = int(torch.argmax(probabilities).item())
        return {
            "model_prediction": "internal" if prediction == 1 else "general",
            "model_confidence": float(probabilities[prediction].item()),
            "general_probability": float(probabilities[0].item()),
            "internal_probability": float(probabilities[1].item()),
        }

    def predict_details(self, query):
        rule_match = self.matched_internal_rule(query)
        model_result = self.predict_model(query)
        routed_prediction = (
            "internal" if rule_match else model_result["model_prediction"]
        )
        details = {
            **model_result,
            "rule_match": rule_match,
            "rule_overrode_model": bool(
                rule_match and model_result["model_prediction"] == "general"
            ),
            "routed_prediction": routed_prediction,
        }
        logger.info(
            "查询分类: model=%s confidence=%.4f rule=%s routed=%s",
            details["model_prediction"],
            details["model_confidence"],
            rule_match or "none",
            routed_prediction,
        )
        return details

    def predict_category(self, query):
        return self.predict_details(query)["routed_prediction"]


if __name__ == "__main__":
    classifier = QueryClassifier(os.getenv("QUERY_CLASSIFIER_MODEL_PATH", "retail-query-classifier"))
    for sample in ("会员积分过期怎么处理？", "水的沸点是多少？", "闭店盘点差异如何上报？"):
        print(sample, "->", classifier.predict_details(sample))
