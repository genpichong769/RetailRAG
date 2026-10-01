"""RetailRAG retrieval orchestration and grounded answer generation."""

import json
import math
import re
import time
from pathlib import Path

from base import Config, logger
from base.query_analysis import analyze_query
from .prompts import RAGPrompts
from .query_classifier import QueryClassifier
from .strategy_selector import StrategySelector

conf = Config()


class RAGSystem:
    def __init__(self, vector_store, llm):
        self.vector_store = vector_store
        self.llm = llm
        self.rag_prompt = RAGPrompts.rag_prompt()
        self.query_classifier = QueryClassifier(conf.bert_intent_cls)
        self.strategy_selector = StrategySelector()
        self.max_prompt_length = 4096
        self.last_retrieval_trace = None
        self.last_generation_trace = None
        self._active_retrieval_queries = []
        self.evidence_gate_config = self._load_evidence_gate_config()

    def rewrite_query(self, query, history):
        """Resolve short follow-ups into one standalone retrieval query."""
        if not conf.QUERY_REWRITE_ENABLED or not history:
            return query
        history_text = "\n".join(
            f"Q: {item['question']}\nA: {item['answer']}"
            for item in history[-3:]
            if isinstance(item, dict) and item.get("question") and item.get("answer")
        )
        if not history_text:
            return query
        try:
            rewritten = self._llm_text(
                RAGPrompts.query_rewrite_prompt().format(
                    history=history_text[:2400], question=query
                )
            ).strip().strip('"“”')
            return rewritten if 2 <= len(rewritten) <= 300 else query
        except Exception as exc:
            logger.error("多轮查询改写失败，回退原问题: %s", exc)
            return query

    @staticmethod
    def _load_evidence_gate_config():
        path = Path(__file__).resolve().parents[2] / "RETAILRAG_PHASE6_EVIDENCE_GATE_CONFIG.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("PHASE 6 证据门配置不可用，回退旧阈值: %s", exc)
            return None

    @staticmethod
    def _text_bigrams(text):
        compact = re.sub(r"\W+", "", str(text).lower())
        return {compact[index : index + 2] for index in range(max(0, len(compact) - 1))}

    @classmethod
    def _evidence_features(cls, query, valid):
        scores = sorted(
            [float(doc.metadata.get("reranker_score", 0.0)) for doc in valid], reverse=True
        )[:3]
        padded = scores + [0.0] * (3 - len(scores))
        query_terms = cls._text_bigrams(query)
        context_terms = cls._text_bigrams(" ".join(doc.page_content for doc in valid))
        analysis = analyze_query(query)
        return {
            "top1_score": padded[0],
            "top2_score": padded[1],
            "top3_score": padded[2],
            "score_mean": sum(scores) / max(len(scores), 1),
            "top1_margin": padded[0] - padded[1],
            "valid_context_count": float(len(valid)),
            "distinct_document_count": float(
                len({doc.metadata.get("document_name") for doc in valid})
            ),
            "distinct_category_count": float(
                len({doc.metadata.get("category") or doc.metadata.get("source") for doc in valid})
            ),
            "query_context_bigram_overlap": len(query_terms & context_terms)
            / max(len(query_terms), 1),
            "scope_risk": float(analysis["scope_risk"]),
        }, analysis

    def _evidence_decision(self, query=None, documents=None):
        # Backward-compatible class call: RAGSystem._evidence_decision(documents).
        if not isinstance(self, RAGSystem):
            documents = self
            query = "" if query is None else query
            config = RAGSystem._load_evidence_gate_config()
        else:
            if documents is None:
                documents = query
                query = ""
            config = self.evidence_gate_config
        valid = [
            doc
            for doc in documents
            if str(doc.metadata.get("status", "active")).lower() == "active"
            and doc.metadata.get("source")
            and doc.metadata.get("document_name")
        ]
        features, analysis = RAGSystem._evidence_features(query, valid)
        top_score = features["top1_score"] if valid else float("-inf")
        reasons = []
        if not valid:
            reasons.append("no_valid_context")
        probability = None
        probability_threshold = None
        if valid and config:
            if config.get("model") == "calibrated_top1_with_deterministic_scope_override":
                probability_threshold = config["reranker_score_threshold"]
                accepted = features[config["score_feature"]] >= probability_threshold
            else:
                ordered = [features[name] for name in config["feature_order"]]
                standardized = [
                    (value - mean) / scale
                    for value, mean, scale in zip(
                        ordered, config["scaler_mean"], config["scaler_scale"]
                    )
                ]
                logit = sum(
                    value * coefficient
                    for value, coefficient in zip(standardized, config["coefficient"])
                ) + config["intercept"]
                probability = 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, logit))))
                probability_threshold = config["answer_probability_threshold"]
                accepted = probability >= probability_threshold
            if not accepted:
                reasons.append("calibrated_insufficient_evidence")
        else:
            accepted = bool(valid) and top_score >= conf.EVIDENCE_RERANKER_THRESHOLD
            probability_threshold = conf.EVIDENCE_RERANKER_THRESHOLD
            if valid and not accepted:
                reasons.append("low_reranker_score")

        mode = "answer" if accepted else "refuse"
        if analysis["scope_risk"]:
            if analysis["partial_possible"] and valid and accepted:
                mode = "partial"
                reasons.append("scope_limited_partial_answer")
            else:
                accepted = False
                mode = "refuse"
                reasons.append("knowledge_scope_risk")
        return {
            "accepted": accepted,
            "mode": mode,
            "top_reranker_score": None if top_score == float("-inf") else top_score,
            "threshold": probability_threshold,
            "answer_probability": probability,
            "valid_context_count": len(valid),
            "features": features,
            "query_analysis": analysis,
            "reasons": reasons,
        }

    def _llm_text(self, prompt):
        value = self.llm(prompt)
        return "".join(value) if not isinstance(value, str) else value

    def _retrieve_with_hyde(self, query, source_filter=None, context_limit=None):
        try:
            hypothetical = self._llm_text(RAGPrompts.hyde_prompt().format(query=query))
            self._active_retrieval_queries.append(hypothetical)
            return self.vector_store.hybrid_search_with_rerank(
                hypothetical,
                k=conf.RETRIEVAL_K,
                source_filter=source_filter,
                limit=context_limit,
            )
        except Exception as exc:
            logger.error("HyDE 策略失败: %s", exc)
            return []

    def _retrieve_with_subqueries(self, query, source_filter=None, context_limit=None):
        try:
            text = self._llm_text(RAGPrompts.subquery_prompt().format(query=query))
            subqueries = [line.strip(" -0123456789.、") for line in text.splitlines() if line.strip()][:3]
            self._active_retrieval_queries.extend(subqueries)
            documents = []
            for subquery in subqueries:
                documents.extend(
                    self.vector_store.hybrid_search_with_rerank(
                        subquery,
                        k=conf.RETRIEVAL_K,
                        source_filter=source_filter,
                        limit=context_limit,
                    )
                )
            best_by_content = {}
            for doc in documents:
                previous = best_by_content.get(doc.page_content)
                score = float(doc.metadata.get("reranker_score", float("-inf")))
                previous_score = (
                    float(previous.metadata.get("reranker_score", float("-inf")))
                    if previous is not None
                    else float("-inf")
                )
                if previous is None or score > previous_score:
                    best_by_content[doc.page_content] = doc
            return sorted(
                best_by_content.values(),
                key=lambda doc: float(doc.metadata.get("reranker_score", float("-inf"))),
                reverse=True,
            )
        except Exception as exc:
            logger.error("子查询策略失败: %s", exc)
            return []

    def _retrieve_with_backtracking(self, query, source_filter=None, context_limit=None):
        try:
            simplified = self._llm_text(RAGPrompts.backtracking_prompt().format(query=query))
            self._active_retrieval_queries.append(simplified)
            return self.vector_store.hybrid_search_with_rerank(
                simplified,
                k=conf.RETRIEVAL_K,
                source_filter=source_filter,
                limit=context_limit,
            )
        except Exception as exc:
            logger.error("回溯问题策略失败: %s", exc)
            return []

    def retrieve_and_merge(self, query, source_filter=None, strategy=None):
        retrieval_started = time.perf_counter()
        strategy = strategy or self.strategy_selector.select_strategy(query)
        query_analysis = analyze_query(query)
        context_budget = 3 if (
            query_analysis["context_complex"] or strategy == "子查询检索"
        ) else conf.CANDIDATE_M
        self._active_retrieval_queries = []
        if strategy == "回溯问题检索":
            documents = self._retrieve_with_backtracking(query, source_filter, context_budget)
        elif strategy == "子查询检索":
            documents = self._retrieve_with_subqueries(query, source_filter, context_budget)
        elif strategy == "假设问题检索":
            documents = self._retrieve_with_hyde(query, source_filter, context_budget)
        else:
            self._active_retrieval_queries.append(query)
            documents = self.vector_store.hybrid_search_with_rerank(
                query,
                k=conf.RETRIEVAL_K,
                source_filter=source_filter,
                limit=context_budget,
            )
        logger.info("策略 %s 返回 %d 个候选文档", strategy, len(documents))
        ranked = sorted(
            documents,
            key=lambda doc: float(doc.metadata.get("reranker_score", float("-inf"))),
            reverse=True,
        )
        final_documents = ranked[: min(conf.CANDIDATE_M, context_budget)]
        diversity_applied = False
        if context_budget > conf.CANDIDATE_M and len(ranked) > conf.CANDIDATE_M:
            top_score = float(ranked[0].metadata.get("reranker_score", 0.0))
            relevance_floor = max(conf.EVIDENCE_RERANKER_THRESHOLD, max(top_score, 0.0) * 0.02)
            selected_names = {
                doc.metadata.get("document_name") for doc in final_documents
            }
            selected_categories = {
                doc.metadata.get("category") or doc.metadata.get("source")
                for doc in final_documents
            }
            eligible = [
                doc for doc in ranked[conf.CANDIDATE_M :]
                if float(doc.metadata.get("reranker_score", float("-inf"))) >= relevance_floor
                and doc.metadata.get("document_name") not in selected_names
            ]
            best_extra_score = (
                float(eligible[0].metadata.get("reranker_score", float("-inf")))
                if eligible
                else float("-inf")
            )
            diverse = [
                doc for doc in eligible
                if float(doc.metadata.get("reranker_score", float("-inf")))
                >= best_extra_score * 0.8
                and (doc.metadata.get("category") or doc.metadata.get("source"))
                not in selected_categories
            ]
            if diverse or eligible:
                final_documents.append((diverse or eligible)[0])
                diversity_applied = bool(diverse)
        self.last_retrieval_trace = {
            "original_query": query,
            "strategy": strategy,
            "retrieval_queries": list(self._active_retrieval_queries),
            "query_analysis": query_analysis,
            "selected_context_budget": context_budget,
            "candidate_pool_count": len(ranked),
            "selected_context_count": len(final_documents),
            "diversity_applied": diversity_applied,
            "context_character_count": sum(len(doc.page_content) for doc in final_documents),
            "context_token_proxy": round(
                sum(len(doc.page_content) for doc in final_documents) / 1.8
            ),
            "latency_ms": round((time.perf_counter() - retrieval_started) * 1000, 3),
            "documents": [
                {
                    "source": doc.metadata.get("source"),
                    "category": doc.metadata.get("category"),
                    "document_name": doc.metadata.get("document_name"),
                    "milvus_score": doc.metadata.get("milvus_score"),
                    "reranker_score": doc.metadata.get("reranker_score"),
                }
                for doc in final_documents
            ],
        }
        return final_documents

    @staticmethod
    def _format_context(documents):
        blocks = []
        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            category = doc.metadata.get("category", source)
            source_label = conf.SOURCE_LABELS.get(category, category)
            name = doc.metadata.get("document_name", "未命名文档")
            blocks.append(f"[来源文档：《{name}》｜知识类别：{source_label}]\n{doc.page_content}")
        return "\n\n".join(blocks)

    def generate_answer(self, query, source_filter=None, history=None):
        started_at = time.time()
        history = history[-5:] if isinstance(history, list) else []
        valid_history = [
            item
            for item in history
            if isinstance(item, dict) and "question" in item and "answer" in item
        ]
        history_context = "\n".join(
            f"Q: {item['question']}\nA: {item['answer']}" for item in valid_history
        )
        rewritten_query = self.rewrite_query(query, valid_history)

        classification = self.query_classifier.predict_details(query)
        query_category = classification["routed_prediction"]
        context_docs = []
        strategy = None
        if query_category == "internal":
            strategy = self.strategy_selector.select_strategy(rewritten_query)
            context_docs = self.retrieve_and_merge(rewritten_query, source_filter, strategy)
        context = self._format_context(context_docs)

        evidence = self._evidence_decision(rewritten_query, context_docs) if query_category == "internal" else {
            "accepted": True,
            "mode": "answer",
            "top_reranker_score": None,
            "threshold": None,
            "valid_context_count": 0,
            "reasons": [],
        }
        if query_category == "internal" and not evidence["accepted"]:
            refusal = (
                "当前知识库信息不足，暂时无法可靠回答该内部问题。"
                f"请联系{conf.CUSTOMER_SERVICE_PHONE}核实。"
            )
            self.last_generation_trace = {
                "question": query,
                "rewritten_query": rewritten_query,
                "classification": classification,
                "strategy": strategy,
                "retrieval_trace": self.last_retrieval_trace,
                "retrieved_sources": [],
                "evidence_decision": evidence,
                "final_context": context,
                "prompt": None,
                "answer": refusal,
                "references": [],
                "llm_called": False,
                "total_latency_ms": round((time.time() - started_at) * 1000, 3),
            }
            logger.info("代码级证据门拒答: %s", evidence)
            yield refusal
            return

        if evidence.get("mode") == "partial":
            context = (
                "[回答范围指令] 仅回答下列证据能确认的政策部分；对实时、账户、人员、"
                "交易记录或其他未提供的部分必须明确说明无法确认。\n\n" + context
            )
        prompt_input = self.rag_prompt.format(
            query_category=query_category,
            context=context or "（无可用检索证据）",
            history=history_context or "（无）",
            question=query,
            phone=conf.CUSTOMER_SERVICE_PHONE,
        )
        if len(prompt_input) > self.max_prompt_length:
            prompt_input = prompt_input[: self.max_prompt_length]
            logger.warning("提示超过 %d 字符，已截断", self.max_prompt_length)

        logger.info(
            "RAG 提示已构建，类别=%s，文档数=%d，耗时=%.2fs",
            query_category,
            len(context_docs),
            time.time() - started_at,
        )
        collected_tokens = []
        try:
            for token in self.llm(prompt_input):
                collected_tokens.append(token)
                yield token
        except Exception as exc:
            logger.error("LLM 调用失败: %s", exc)
            fallback = f"抱歉，当前无法生成回答，请联系人工支持：{conf.CUSTOMER_SERVICE_PHONE}"
            collected_tokens.append(fallback)
            yield fallback
        finally:
            self.last_generation_trace = {
                "question": query,
                "rewritten_query": rewritten_query,
                "classification": classification,
                "strategy": strategy,
                "retrieval_trace": self.last_retrieval_trace if query_category == "internal" else None,
                "retrieved_sources": [
                    {
                        "source": doc.metadata.get("source"),
                        "category": doc.metadata.get("category"),
                        "document_name": doc.metadata.get("document_name"),
                        "milvus_score": doc.metadata.get("milvus_score"),
                        "reranker_score": doc.metadata.get("reranker_score"),
                    }
                    for doc in context_docs
                ],
                "final_context": context,
                "evidence_decision": evidence,
                "prompt": prompt_input,
                "answer": "".join(collected_tokens),
                "references": [doc.metadata.get("document_name") for doc in context_docs],
                "llm_called": True,
                "total_latency_ms": round((time.time() - started_at) * 1000, 3),
            }
