"""LLM-based selection among the four inherited retrieval strategies."""

from langchain.prompts import PromptTemplate
import httpx
from openai import OpenAI

from base import Config, logger


class StrategySelector:
    VALID_STRATEGIES = ("直接检索", "假设问题检索", "子查询检索", "回溯问题检索")

    def __init__(self):
        config = Config()
        self.model = config.LLM_MODEL
        self.client = None
        self.last_call_status = "NOT_RUN"
        self.last_error = None
        self.last_raw_response = None
        if config.DASHSCOPE_API_KEY:
            self.client = OpenAI(
                api_key=config.DASHSCOPE_API_KEY,
                base_url=config.DASHSCOPE_BASE_URL,
                http_client=httpx.Client(trust_env=False, timeout=30),
            )
        else:
            logger.warning("DASHSCOPE_API_KEY 未配置，检索策略运行阶段阻断并回退直接检索")
        self.strategy_prompt_template = self._get_strategy_prompt()

    def call_dashscope(self, prompt):
        if self.client is None:
            self.last_call_status = "BLOCKED_API_KEY"
            self.last_raw_response = None
            return "直接检索"
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你只负责选择检索策略。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                timeout=30,
            )
            raw = completion.choices[0].message.content if completion.choices else "直接检索"
            self.last_call_status = "PASS"
            self.last_error = None
            self.last_raw_response = raw
            return raw
        except Exception as exc:
            self.last_call_status = "BLOCKED_NETWORK_OR_ENDPOINT"
            self.last_error = str(exc)
            self.last_raw_response = None
            logger.error("检索策略选择失败: %s", exc)
            return "直接检索"

    @staticmethod
    def _get_strategy_prompt():
        return PromptTemplate(
            template="""分析 Nova Retail 员工查询，只返回下列一个策略名称，不要解释：
- 直接检索：单一、明确的制度事实，例如“无小票退货需要什么凭证？”
- 假设问题检索：表达抽象或口语化，例如“怎样做才能让会员觉得积分更好用？”
- 子查询检索：同时询问多个实体或流程，例如“比较开店和闭店时现金、设备、库存分别要检查什么？”
- 回溯问题检索：问题包含复杂个案，需要先还原基础制度，例如“促销订单部分退货又用了赠品和券该怎么算？”

查询：{query}
策略：""",
            input_variables=["query"],
        )

    def select_strategy(self, query):
        raw = self.call_dashscope(self.strategy_prompt_template.format(query=query)).strip()
        strategy = next((name for name in self.VALID_STRATEGIES if name in raw), "直接检索")
        logger.info("查询选择检索策略：%s", strategy)
        return strategy


if __name__ == "__main__":
    print(StrategySelector().select_strategy("会员日订单用了券，部分退货后积分和赠品怎么处理？"))
