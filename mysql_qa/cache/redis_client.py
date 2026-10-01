"""Redis cache helpers for RetailRAG FAQ retrieval."""

import json

import redis

from base import Config, logger


class RedisClient:
    """Store FAQ indexes and answers as JSON with a bounded lifetime."""

    def __init__(self):
        self.logger = logger
        self.config = Config()
        self.namespace = self.config.FAQ_CACHE_NAMESPACE
        self.ttl = self.config.FAQ_CACHE_TTL
        try:
            self.client = redis.StrictRedis(
                host=self.config.REDIS_HOST,
                port=self.config.REDIS_PORT,
                password=self.config.REDIS_PASSWORD,
                db=self.config.REDIS_DB,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=10,
            )
            self.logger.info("Redis 客户端初始化完成")
        except redis.RedisError as exc:
            self.logger.error("Redis 连接失败: %s", exc)
            raise

    def faq_key(self, suffix):
        return f"{self.namespace}:{suffix}"

    def set_data(self, key, value, ttl=None):
        """Serialize *value* once and store it with the FAQ TTL."""
        try:
            self.client.set(
                key,
                json.dumps(value, ensure_ascii=False),
                ex=self.ttl if ttl is None else ttl,
            )
            self.logger.info("写入 Redis 缓存: %s", key)
        except redis.RedisError as exc:
            self.logger.error("Redis 存储失败: %s", exc)

    def get_data(self, key):
        """Return deserialized JSON; tolerate invalid legacy values as misses."""
        try:
            raw = self.client.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except (redis.RedisError, json.JSONDecodeError, TypeError) as exc:
            self.logger.warning("Redis 缓存读取失败，按未命中处理 (%s): %s", key, exc)
            return None

    def answer_key(self, query, variant=None):
        variant = variant or self.config.FAQ_ALGORITHM_VERSION
        return self.faq_key(f"answer:{variant}:{query}")

    def set_answer(self, query, answer, variant=None):
        self.set_data(self.answer_key(query, variant), answer)

    def get_answer(self, query, variant=None):
        answer = self.get_data(self.answer_key(query, variant))
        if answer is not None:
            self.logger.info("命中 Redis FAQ 答案缓存")
        return answer

    def clear_faq_cache(self):
        """Delete only keys owned by the current RetailRAG FAQ namespace."""
        pattern = self.faq_key("*")
        keys = list(self.client.scan_iter(match=pattern, count=200))
        if keys:
            self.client.delete(*keys)
        self.logger.info("已清理 %d 个 FAQ 缓存键 (%s)", len(keys), pattern)
        return len(keys)


if __name__ == "__main__":
    client = RedisClient()
    client.set_answer("会员日是哪天？", "以当期内部活动通知为准。")
    print(client.get_answer("会员日是哪天？"))
