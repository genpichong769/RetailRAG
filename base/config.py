import ast
import configparser
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    # python-dotenv is included in the project requirements. Keeping this
    # optional lets configuration-only tooling run before dependencies install.
    pass


class Config:
    """Load local configuration without embedding credentials in source code."""

    def __init__(self, config_file=None):
        configured_path = config_file or os.getenv("RETAILRAG_CONFIG_FILE")
        self.config_file = Path(configured_path) if configured_path else PROJECT_ROOT / "config.ini"

        self.config = configparser.ConfigParser()
        self.config.read(self.config_file, encoding="utf-8")

        # Environment variables take precedence over config.ini.
        self.MYSQL_HOST = self._get("MYSQL_HOST", "mysql", "host", "localhost")
        self.MYSQL_PORT = int(self._get("MYSQL_PORT", "mysql", "port", "3307"))
        self.MYSQL_USER = self._get("MYSQL_USER", "mysql", "user", "retailrag")
        self.MYSQL_PASSWORD = self._get("MYSQL_PASSWORD", "mysql", "password", "")
        self.MYSQL_DATABASE = self._get("MYSQL_DATABASE", "mysql", "database", "retailrag")

        self.REDIS_HOST = self._get("REDIS_HOST", "redis", "host", "localhost")
        self.REDIS_PORT = int(self._get("REDIS_PORT", "redis", "port", "6379"))
        self.REDIS_PASSWORD = self._get("REDIS_PASSWORD", "redis", "password", "") or None
        self.REDIS_DB = int(self._get("REDIS_DB", "redis", "db", "0"))

        self.MILVUS_HOST = self._get("MILVUS_HOST", "milvus", "host", "localhost")
        self.MILVUS_PORT = self._get("MILVUS_PORT", "milvus", "port", "19530")
        self.MILVUS_DATABASE_NAME = self._get(
            "MILVUS_DATABASE_NAME", "milvus", "database_name", "retailrag"
        )
        self.MILVUS_COLLECTION_NAME = self._get(
            "MILVUS_COLLECTION_NAME", "milvus", "collection_name", "retailrag_v1"
        )

        # DashScope exposes an OpenAI-compatible API. API_KEY is kept for
        # backwards compatibility with existing local environments.
        self.LLM_MODEL = self._get("LLM_MODEL", "llm", "model", "qwen3.7-plus")
        self.DASHSCOPE_API_KEY = (
            os.getenv("DASHSCOPE_API_KEY")
            or os.getenv("API_KEY")
            or self.config.get("llm", "dashscope_api_key", fallback="")
        )
        self.DASHSCOPE_BASE_URL = self._get(
            "DASHSCOPE_BASE_URL",
            "llm",
            "dashscope_base_url",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        self.PARENT_CHUNK_SIZE = int(
            self._get("PARENT_CHUNK_SIZE", "retrieval", "parent_chunk_size", "1200")
        )
        self.CHILD_CHUNK_SIZE = int(
            self._get("CHILD_CHUNK_SIZE", "retrieval", "child_chunk_size", "300")
        )
        self.CHUNK_OVERLAP = int(
            self._get("CHUNK_OVERLAP", "retrieval", "chunk_overlap", "50")
        )
        self.RETRIEVAL_K = int(self._get("RETRIEVAL_K", "retrieval", "retrieval_k", "3"))
        self.CANDIDATE_M = int(self._get("CANDIDATE_M", "retrieval", "candidate_m", "2"))
        self.EVIDENCE_RERANKER_THRESHOLD = float(
            self._get(
                "EVIDENCE_RERANKER_THRESHOLD",
                "retrieval",
                "evidence_reranker_threshold",
                "0.020635",
            )
        )
        self.QUERY_REWRITE_ENABLED = self._get(
            "QUERY_REWRITE_ENABLED", "retrieval", "query_rewrite_enabled", "true"
        ).strip().lower() in {"1", "true", "yes", "on"}

        sources = self._get(
            "VALID_SOURCES",
            "app",
            "valid_sources",
            '["membership", "promotion", "after_sales", "store_operation", '
            '"product", "equipment", "policy", "notice"]',
        )
        self.VALID_SOURCES = ast.literal_eval(sources)
        self.SOURCE_LABELS = {
            "membership": "会员制度",
            "promotion": "促销活动",
            "after_sales": "售后政策",
            "store_operation": "门店运营",
            "product": "商品知识",
            "equipment": "设备操作",
            "policy": "企业制度",
            "notice": "内部通知",
        }
        self.CUSTOMER_SERVICE_PHONE = self._get(
            "CUSTOMER_SERVICE_PHONE", "app", "customer_service_phone", "门店值班经理或内部服务台"
        )
        self.LOG_FILE = self._get("LOG_FILE", "logger", "log_file", "logs/app.log")
        self.FAQ_CACHE_TTL = int(
            self._get("FAQ_CACHE_TTL", "redis", "faq_cache_ttl", "86400")
        )
        self.FAQ_CACHE_NAMESPACE = self._get(
            "FAQ_CACHE_NAMESPACE", "redis", "faq_cache_namespace", "retailrag:v1"
        )
        self.FAQ_RETRIEVAL_MODE = self._get(
            "FAQ_RETRIEVAL_MODE", "faq", "retrieval_mode", "dense"
        ).strip().lower()
        self.FAQ_MATCH_THRESHOLD = float(
            self._get("FAQ_MATCH_THRESHOLD", "faq", "match_threshold", "0.84")
        )
        self.FAQ_ALGORITHM_VERSION = self._get(
            "FAQ_ALGORITHM_VERSION", "faq", "algorithm_version", "dense-bgem3-v1"
        )

        # Local model artifacts are intentionally excluded from Git.
        local_m3 = PROJECT_ROOT / "rag_qa" / "models" / "bge-m3"
        local_reranker = PROJECT_ROOT / "rag_qa" / "models" / "bge-reranker-large"
        self.bge_m3 = os.getenv("BGE_M3_PATH") or (
            str(local_m3) if local_m3.is_dir() else "BAAI/bge-m3"
        )
        self.bge_reranker = os.getenv("BGE_RERANKER_PATH") or (
            str(local_reranker) if local_reranker.is_dir() else "BAAI/bge-reranker-large"
        )
        self.nlp_bert_doc_seg = os.getenv(
            "DOCUMENT_SEGMENTATION_MODEL_PATH",
            str(PROJECT_ROOT / "rag_qa" / "models" / "nlp_bert_document-segmentation_chinese-base"),
        )
        self.bert_intent_cls = os.getenv(
            "QUERY_CLASSIFIER_MODEL_PATH",
            str(PROJECT_ROOT / "rag_qa" / "models" / "retail-query-classifier"),
        )

    def _get(self, env_name, section, option, fallback):
        value = os.getenv(env_name)
        if value is not None:
            return value
        return self.config.get(section, option, fallback=fallback)


if __name__ == "__main__":
    conf = Config()
    print(conf.MYSQL_HOST)
