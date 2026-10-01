"""BM25 FAQ retrieval with Redis-backed index and answer caches."""

import numpy as np
from rank_bm25 import BM25Okapi

from base import Config, logger
from base.query_analysis import analyze_query
from mysql_qa.utils.preprocess import preprocess_text


class BM25Search:
    def __init__(self, redis_client, mysql_client):
        self.logger = logger
        self.config = Config()
        self.redis_client = redis_client
        self.mysql_client = mysql_client
        self.original_key = redis_client.faq_key("qa_original_questions")
        self.tokenized_key = redis_client.faq_key("qa_tokenized_questions")
        self.bm25 = None
        self.questions = None
        self.original_questions = None
        self.embedding_function = None
        self.faq_dense_vectors = None
        self._load_data()

    def set_embedding_function(self, embedding_function):
        """Reuse RAG's BGE-M3 instance for a lightweight 52-row memory index."""
        self.embedding_function = embedding_function
        if self.original_questions:
            self.faq_dense_vectors = np.asarray(
                embedding_function(self.original_questions)["dense"], dtype=np.float32
            )
            self.logger.info(
                "Dense FAQ 索引初始化完成，共 %d 条，算法版本=%s",
                len(self.original_questions),
                self.config.FAQ_ALGORITHM_VERSION,
            )

    def _load_data(self):
        self.original_questions = self.redis_client.get_data(self.original_key)
        tokenized_questions = self.redis_client.get_data(self.tokenized_key)

        if not self.original_questions or not tokenized_questions:
            rows = self.mysql_client.fetch_questions()
            self.original_questions = [row[0] for row in rows]
            if not self.original_questions:
                self.logger.warning("未加载到 FAQ 问题，BM25 暂不可用")
                return
            tokenized_questions = [preprocess_text(question) for question in self.original_questions]
            self.redis_client.set_data(self.original_key, self.original_questions)
            self.redis_client.set_data(self.tokenized_key, tokenized_questions)

        self.questions = tokenized_questions
        self.bm25 = BM25Okapi(self.questions)
        self.logger.info("BM25 模型初始化完成，共 %d 条 FAQ", len(self.original_questions))

    def reload(self):
        """Rebuild the in-process index after an explicit FAQ import."""
        self.redis_client.clear_faq_cache()
        self._load_data()

    @staticmethod
    def _softmax(scores):
        exp_scores = np.exp(scores - np.max(scores))
        return exp_scores / exp_scores.sum()

    def search(self, query, threshold=None):
        details = self.search_with_details(query, threshold=threshold)
        return details["answer"], details["need_rag"]

    def search_with_details(self, query, threshold=None):
        """Return the existing FAQ decision plus diagnostics for runtime audits."""
        threshold = self.config.FAQ_MATCH_THRESHOLD if threshold is None else float(threshold)
        if not query or not isinstance(query, str):
            self.logger.error("无效查询")
            return {
                "query": query,
                "top1_question": None,
                "raw_score": None,
                "score": None,
                "threshold": threshold,
                "above_threshold": False,
                "hit": False,
                "cached": False,
                "answer": None,
                "need_rag": False,
            }
        # Gate semantics are part of the cache identity so answers accepted by
        # the pre-gate implementation cannot bypass the new decision.
        cache_variant = f"{self.config.FAQ_ALGORITHM_VERSION}-phase6-gate-v2"
        cached_answer = self.redis_client.get_answer(query, variant=cache_variant)
        if cached_answer is not None:
            analysis = analyze_query(query)
            return {
                "query": query,
                "top1_question": None,
                "raw_score": None,
                "score": None,
                "threshold": threshold,
                "above_threshold": True,
                "hit": True,
                "cached": True,
                "answer": cached_answer,
                "need_rag": False,
                "gate_blocked": False,
                "gate_reasons": [],
                "query_analysis": analysis,
                "algorithm_version": cache_variant,
            }
        if self.bm25 is None or not self.original_questions:
            return {
                "query": query,
                "top1_question": None,
                "raw_score": None,
                "score": None,
                "threshold": threshold,
                "above_threshold": False,
                "hit": False,
                "cached": False,
                "answer": None,
                "need_rag": True,
            }

        try:
            mode = self.config.FAQ_RETRIEVAL_MODE
            if mode == "dense":
                if self.embedding_function is None or self.faq_dense_vectors is None:
                    raise RuntimeError("Dense FAQ embedding function has not been initialized")
                query_vector = np.asarray(
                    self.embedding_function([query])["dense"][0], dtype=np.float32
                )
                scores = np.matmul(self.faq_dense_vectors, query_vector)
                best_idx = int(scores.argmax())
                raw_score = float(scores[best_idx])
                best_score = raw_score
            elif mode == "bm25":
                scores = self.bm25.get_scores(preprocess_text(query))
                softmax_scores = self._softmax(scores)
                best_idx = int(softmax_scores.argmax())
                raw_score = float(scores[best_idx])
                best_score = float(softmax_scores[best_idx])
            else:
                raise ValueError(f"Unsupported FAQ retrieval mode: {mode}")
            original_question = self.original_questions[best_idx]
            self.logger.info("FAQ(%s) 最高分 [%.3f]: %s", mode, best_score, original_question)
            above_threshold = best_score >= threshold
            gate = analyze_query(query, original_question)
            gate_blocked = bool(above_threshold and gate["faq_force_rag"])
            answer = (
                self.mysql_client.fetch_answer(original_question)
                if above_threshold and not gate_blocked
                else None
            )
            hit = bool(answer)
            if hit:
                self.redis_client.set_answer(query, answer, variant=cache_variant)
            return {
                "query": query,
                "top1_question": original_question,
                "raw_score": raw_score,
                "score": best_score,
                "threshold": threshold,
                "above_threshold": above_threshold,
                "hit": hit,
                "cached": False,
                "answer": answer,
                "need_rag": not hit,
                "gate_blocked": gate_blocked,
                "gate_reasons": gate["faq_gate_reasons"],
                "query_analysis": gate,
                "mode": mode,
                "algorithm_version": cache_variant,
            }
        except Exception as exc:
            self.logger.error("BM25 搜索失败: %s", exc)
            return {
                "query": query,
                "top1_question": None,
                "raw_score": None,
                "score": None,
                "threshold": threshold,
                "above_threshold": False,
                "hit": False,
                "cached": False,
                "answer": None,
                "need_rag": True,
                "error": str(exc),
            }
