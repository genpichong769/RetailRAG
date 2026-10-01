# -*- coding:utf-8 -*-
# 导入 BGE-M3 嵌入函数，用于生成文档和查询的向量表示
import torch.cuda
from milvus_model.hybrid import BGEM3EmbeddingFunction
# 导入 Milvus 相关类，用于操作向量数据库
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker
# 导入 Document 类，用于创建文档对象
from langchain.docstore.document import Document
# 导入 CrossEncoder，用于重排序和 NLI 判断
from sentence_transformers import CrossEncoder
# 导入 hashlib 模块，用于生成唯一 ID 的哈希值
import hashlib
import time
from datetime import date
from rag_qa.core.document_processor import *
import sys, os

# 获取当前文件所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# print(f'current_dir--》{current_dir}')
# 获取core文件所在的目录的绝对路径
rag_qa_path = os.path.dirname(current_dir)
# print(f'rag_qa_path--》{rag_qa_path}')
sys.path.insert(0, rag_qa_path)
# 获取根目录文件所在的绝对位置
project_root = os.path.dirname(rag_qa_path)
sys.path.insert(0, project_root)
from base import logger, Config

conf = Config()

VERSION_FIELDS = [
    "version",
    "effective_date",
    "expire_date",
    "status",
    "version_group",
]
OUTPUT_FIELDS = [
    "text",
    "parent_id",
    "parent_content",
    "source",
    "category",
    "document_name",
    "timestamp",
    *VERSION_FIELDS,
]


# core/vector_store.py
# 定义 VectorStore 类，封装向量存储和检索功能
class VectorStore:
    # 初始化方法，设置向量存储的基本参数
    def __init__(self,
                 collection_name=conf.MILVUS_COLLECTION_NAME,
                 host=conf.MILVUS_HOST,
                 port=conf.MILVUS_PORT,
                 database=conf.MILVUS_DATABASE_NAME):
        # 设置 Milvus 集合名称
        self.collection_name = collection_name
        # 设置 Milvus 主机地址
        self.host = host
        # 设置 Milvus 端口号
        self.port = port
        # 设置 Milvus 数据库名称
        self.database = database
        # 设置日志记录器
        self.logger = logger
        self.last_search_trace = None
        self._query_embedding_cache = {}
        # 检查CUDA是否可用
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # 日志提醒使用的是什么设备
        self.logger.info(f"使用设置：{self.device}")
        # 初始化 BGE-Reranker 模型，用于重排序检索结果
        reranker_path = conf.bge_reranker
        # print(f'reranker_path--》{reranker_path}')
        self.reranker = CrossEncoder(reranker_path, device=self.device)
        # 初始化 BGE-M3 嵌入函数，使用 CPU 设备，不启用 FP16
        m3_path = conf.bge_m3
        self.embedding_function = BGEM3EmbeddingFunction(model_name_or_path=m3_path, use_fp16=(self.device == 'cuda'),
                                                         device=self.device)
        # 获取稠密向量的维度# 1024
        self.dense_dim = self.embedding_function.dim["dense"]
        # 初始化 Milvus 客户端，连接到指定主机和数据库
        self.client = MilvusClient(
            uri=f"http://{self.host}:{self.port}", db_name=self.database, timeout=10
        )
        # 调用方法创建或加载 Milvus 集合
        self._create_or_load_collection()

    # 类私有化方法
    def _create_or_load_collection(self):
        # 检查指定集合是否已经存在
        if not self.client.has_collection(self.collection_name):
            # 创建集合 Schema，禁用自动 ID，启用动态字段
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
            # 添加 ID 字段，作为主键，VARCHAR 类型，最大长度 100
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
            # 添加文本字段，VARCHAR 类型，最大长度 65535
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
            # 添加稠密向量字段，FLOAT_VECTOR 类型，维度由嵌入函数指定
            schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=self.dense_dim)
            # 添加稀疏向量字段，SPARSE_FLOAT_VECTOR 类型
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
            # 添加父块 ID 字段，VARCHAR 类型，最大长度 100
            schema.add_field(field_name="parent_id", datatype=DataType.VARCHAR, max_length=100)
            # 添加父块内容字段，VARCHAR 类型，最大长度 65535
            schema.add_field(field_name="parent_content", datatype=DataType.VARCHAR, max_length=65535)
            # 添加零售知识类别字段，VARCHAR 类型，最大长度 50
            schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=50)
            schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=50)
            # 最小可解释性元数据：用于在最终答案中展示来源文档名
            schema.add_field(field_name="document_name", datatype=DataType.VARCHAR, max_length=255)
            schema.add_field(field_name="version", datatype=DataType.VARCHAR, max_length=50)
            schema.add_field(field_name="effective_date", datatype=DataType.VARCHAR, max_length=20)
            schema.add_field(field_name="expire_date", datatype=DataType.VARCHAR, max_length=20)
            schema.add_field(field_name="status", datatype=DataType.VARCHAR, max_length=20)
            schema.add_field(field_name="version_group", datatype=DataType.VARCHAR, max_length=255)
            # 添加时间戳字段，VARCHAR 类型，最大长度 50
            schema.add_field(field_name="timestamp", datatype=DataType.VARCHAR, max_length=50)

            # 创建索引参数对象
            index_params = self.client.prepare_index_params()
            # 为稠密向量字段添加 IVF_FLAT 索引，度量类型为内积 (IP)
            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_index",
                index_type="IVF_FLAT",
                metric_type="IP",
                params={"nlist": 128}
            )
            # 为稀疏向量字段添加 SPARSE_INVERTED_INDEX 索引，度量类型为内积 (IP)
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"drop_ratio_build": 0.2}
            )

            # 创建 Milvus 集合，应用定义的 Schema 和索引参数
            self.client.create_collection(collection_name=self.collection_name, schema=schema,
                                          index_params=index_params)
            # 记录创建集合的日志
            logger.info(f"已创建集合 {self.collection_name}")
        # 如果集合已存在
        else:
            # 记录加载集合的日志
            logger.info(f"已加载集合 {self.collection_name}")
        # 将集合加载到内存，确保可立即查询
        self.client.load_collection(self.collection_name)

    # 定义方法，向向量存储添加文档
    def add_documents(self, documents):
        # print(f'documents--》{documents[0]}')
        # 提取所有文档的内容列表
        texts = [doc.page_content for doc in documents]

        # 使用 BGE-M3 嵌入函数生成文档的嵌入
        embeddings = self.embedding_function(texts)
        # print(f'embeddings--》{embeddings}')
        # print(f'embeddings--》{embeddings.keys()}')
        # 初始化空列表，存储插入的数据
        data = []
        # 遍历每个文档，带上索引i
        for i, doc in enumerate(documents):
            # 生成文档内容的哈希值作为唯一的ID
            identity = "|".join([
                doc.metadata.get("source", "unknown"),
                doc.metadata.get("document_name", "未命名文档"),
                doc.metadata.get("parent_id", "unknown"),
                doc.page_content,
            ])
            text_hash = hashlib.md5(identity.encode('utf-8')).hexdigest()
            # print(f'text_hash--》{text_hash}')
            # print(f'text_hash--》{type(text_hash)}')
            # 初始化一个稀疏向量的字典（Milvus要求存储稀疏向量的格式）
            sparse_vector = {}

            try:
                # 新版本 milvus-model 使用 coo_array 格式
                row = embeddings["sparse"][i]
                if hasattr(row, 'col'):  # coo_array 格式
                    indices = row.col
                    values = row.data
                else:  # csr_matrix 格式
                    indices = row.indices
                    values = row.data
            except Exception as e:
                # 兼容旧版本 milvus-model
                row = embeddings["sparse"].getrow(i)
                indices = row.indices
                values = row.data

            # 获取第i行对应的稀疏向量数据[0.4, 0.2, 0, 0, 0.1]
            # row = embeddings["sparse"].getrow(i)
            # row = embeddings["sparse"][i]  # 新版本milvus-model，支持这种获取稀疏向量的形式
            # indices = row.row
            # print(f'row--》{row}')
            # print(f'row--》{row.shape}')
            # 获取稀疏向量的非零值的索引
            # indices = row.col
            # print(f'indics--》{indics}')
            # 获取稀疏向量的非零值
            # values = row.data

            # 将索引和值进行配对，存储到字典中
            for idx, value in zip(indices, values):
                sparse_vector[idx] = value
            # print(f'sparse_vector--》{sparse_vector}')
            # print(f'sparse_vector--》{len(sparse_vector)}')
            # print(embeddings["dense"][i])
            # print(embeddings["dense"][i].shape)
            # 创建数据字典，包含所有字段
            data.append({
                "id": text_hash,
                "text": doc.page_content,
                "dense_vector": embeddings["dense"][i],
                "sparse_vector": sparse_vector,
                "parent_id": doc.metadata["parent_id"],
                "parent_content": doc.metadata["parent_content"],
                "source": doc.metadata.get("source", "unknown"),
                "category": doc.metadata.get("category", doc.metadata.get("source", "unknown")),
                "document_name": doc.metadata.get("document_name", "未命名文档"),
                "timestamp": doc.metadata.get("timestamp", "unknown"),
                "version": str(doc.metadata.get("version", "1.0")),
                "effective_date": str(doc.metadata.get("effective_date", "2000-01-01")),
                "expire_date": str(doc.metadata.get("expire_date", "9999-12-31")),
                "status": str(doc.metadata.get("status", "active")),
                "version_group": str(
                    doc.metadata.get("version_group", doc.metadata.get("document_name", "unknown"))
                ),
            })
        # 检查是否有数据需要插入
        if data:
            # 使用 upsert 操作插入数据，覆盖重复 ID
            self.client.upsert(collection_name=self.collection_name, data=data)
            # 记录插入或更新的文档数量日志
            logger.info(f"已插入或更新 {len(data)} 个文档")

    @staticmethod
    def _to_sparse_vector(row):
        try:
            indices = row.col if hasattr(row, "col") else row.indices
            values = row.data
        except AttributeError:
            row = row.getrow(0)
            indices = row.indices
            values = row.data
        return {int(index): float(value) for index, value in zip(indices, values)}

    @staticmethod
    def _version_identity(doc):
        metadata = doc.metadata
        return (
            str(metadata.get("effective_date", "2000-01-01")),
            str(metadata.get("version", "1.0")),
            str(metadata.get("document_name", "unknown")),
        )

    def _filter_current_versions(self, documents, active_only=True, as_of=None):
        """Apply generic status/date/latest-version filtering without document-name rules."""
        if not active_only:
            return documents
        current_date = (as_of or date.today()).isoformat()
        eligible = []
        for doc in documents:
            metadata = doc.metadata
            if str(metadata.get("status", "active")).lower() != "active":
                continue
            effective = str(metadata.get("effective_date", "2000-01-01"))
            expires = str(metadata.get("expire_date", "9999-12-31"))
            if effective <= current_date <= expires:
                eligible.append(doc)

        latest_by_group = {}
        for doc in eligible:
            group = str(
                doc.metadata.get("version_group")
                or doc.metadata.get("document_name")
                or "unknown"
            )
            identity = self._version_identity(doc)
            if group not in latest_by_group or identity > latest_by_group[group]:
                latest_by_group[group] = identity
        return [
            doc
            for doc in eligible
            if self._version_identity(doc)
            == latest_by_group.get(
                str(
                    doc.metadata.get("version_group")
                    or doc.metadata.get("document_name")
                    or "unknown"
                )
            )
        ]

    @staticmethod
    def _build_filter(source_filter=None, active_only=True):
        clauses = []
        if source_filter:
            safe_source = str(source_filter).replace("'", "\\'")
            clauses.append(f"source == '{safe_source}'")
        if active_only:
            clauses.append("status == 'active'")
        return " and ".join(clauses)

    def search_documents(
        self,
        query,
        mode="hybrid",
        rerank=True,
        candidate_k=conf.RETRIEVAL_K,
        limit=conf.CANDIDATE_M,
        source_filter=None,
        active_only=True,
        as_of=None,
    ):
        """Evaluation-capable dense/sparse/hybrid retrieval; online defaults stay unchanged."""
        search_started = time.perf_counter()
        mode = str(mode).lower()
        if mode not in {"dense", "sparse", "hybrid"}:
            raise ValueError(f"Unsupported retrieval mode: {mode}")
        candidate_k = max(int(candidate_k), 1)
        limit = max(int(limit), 1)
        embedding_started = time.perf_counter()
        query_embeddings = self._query_embedding_cache.get(query)
        embedding_cache_hit = query_embeddings is not None
        if query_embeddings is None:
            query_embeddings = self.embedding_function([query])
            # Evaluation invokes several retrieval modes for the same query.
            if len(self._query_embedding_cache) >= 512:
                self._query_embedding_cache.clear()
            self._query_embedding_cache[query] = query_embeddings
        embedding_ms = (time.perf_counter() - embedding_started) * 1000
        dense_vector = query_embeddings["dense"][0]
        sparse_embeddings = query_embeddings["sparse"]
        try:
            sparse_row = sparse_embeddings[0]
        except NotImplementedError:
            sparse_row = sparse_embeddings.getrow(0)
        sparse_vector = self._to_sparse_vector(sparse_row)
        filter_expr = self._build_filter(source_filter, active_only)

        retrieval_started = time.perf_counter()
        if mode == "hybrid":
            dense_request = AnnSearchRequest(
                data=[dense_vector],
                anns_field="dense_vector",
                param={"metric_type": "IP", "params": {"nprobe": 10}},
                limit=candidate_k,
                expr=filter_expr,
            )
            sparse_request = AnnSearchRequest(
                data=[sparse_vector],
                anns_field="sparse_vector",
                param={"metric_type": "IP", "params": {}},
                limit=candidate_k,
                expr=filter_expr,
            )
            results = self.client.hybrid_search(
                collection_name=self.collection_name,
                reqs=[dense_request, sparse_request],
                ranker=WeightedRanker(1.0, 0.7),
                limit=candidate_k,
                output_fields=OUTPUT_FIELDS,
            )[0]
        else:
            vector = dense_vector if mode == "dense" else sparse_vector
            field = "dense_vector" if mode == "dense" else "sparse_vector"
            params = (
                {"metric_type": "IP", "params": {"nprobe": 10}}
                if mode == "dense"
                else {"metric_type": "IP", "params": {}}
            )
            results = self.client.search(
                collection_name=self.collection_name,
                data=[vector],
                anns_field=field,
                search_params=params,
                limit=candidate_k,
                filter=filter_expr,
                output_fields=OUTPUT_FIELDS,
            )[0]
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000

        child_hits = []
        for hit in results:
            entity = hit["entity"]
            distance = hit.get("distance") if hasattr(hit, "get") else getattr(hit, "distance", None)
            child_hits.append(
                {
                    "text": entity.get("text"),
                    "parent_id": entity.get("parent_id"),
                    "source": entity.get("source"),
                    "category": entity.get("category", entity.get("source")),
                    "document_name": entity.get("document_name"),
                    "version": entity.get("version"),
                    "effective_date": entity.get("effective_date"),
                    "expire_date": entity.get("expire_date"),
                    "status": entity.get("status"),
                    "version_group": entity.get("version_group"),
                    "milvus_score": float(distance) if distance is not None else None,
                }
            )

        parent_docs = self._get_unique_parent_docs(
            [self._doc_from_hit(hit) for hit in results]
        )
        parent_docs = self._filter_current_versions(parent_docs, active_only, as_of)
        rerank_started = time.perf_counter()
        if rerank and parent_docs:
            scores = self.reranker.predict([[query, doc.page_content] for doc in parent_docs])
            for score, doc in zip(scores, parent_docs):
                doc.metadata["reranker_score"] = float(score)
            parent_docs = sorted(
                parent_docs,
                key=lambda doc: doc.metadata.get("reranker_score", float("-inf")),
                reverse=True,
            )
        rerank_ms = (time.perf_counter() - rerank_started) * 1000 if rerank else 0.0

        final_docs = parent_docs[:limit]
        total_ms = (time.perf_counter() - search_started) * 1000
        self._record_search_trace(
            query,
            candidate_k,
            source_filter,
            child_hits,
            final_docs,
            candidate_m=limit,
            candidate_parent_count=len(parent_docs),
            timings={
                "embedding_ms": round(embedding_ms, 3),
                "retrieval_ms": round(retrieval_ms, 3),
                "rerank_ms": round(rerank_ms, 3),
                "total_ms": round(total_ms, 3),
                "embedding_cache_hit": embedding_cache_hit,
            },
        )
        self.last_search_trace.update(
            {
                "mode": mode,
                "rerank": bool(rerank),
                "active_only": bool(active_only),
                "as_of": (as_of or date.today()).isoformat(),
                "requested_parent_limit": limit,
            }
        )
        return final_docs

    # Online keeps Hybrid + Reranker and K=3.  The caller may select M=2/3.
    def hybrid_search_with_rerank(
        self, query, k=conf.RETRIEVAL_K, source_filter=None, limit=None
    ):
        parent_limit = conf.CANDIDATE_M if limit is None else max(1, min(int(limit), k))
        return self.search_documents(
            query,
            mode="hybrid",
            rerank=True,
            candidate_k=k,
            limit=parent_limit,
            source_filter=source_filter,
            active_only=True,
        )

    # 保留 PHASE 4.1 实现作为对照，不再被线上调用。
    def _legacy_hybrid_search_with_rerank(self, query, k=conf.RETRIEVAL_K, source_filter=None):
        # 使用 BGE-M3 嵌入函数生成查询的嵌入
        query_embeddings = self.embedding_function([query])
        # 获取查询的稠密向量
        dense_query_vector = query_embeddings["dense"][0]
        # print(f'dense_query_vector--》{dense_query_vector.shape}')
        # 初始化查询的稀疏向量字典
        sparse_query_vector = {}
        # ====================
        # 获取查询稀疏向量的第 0 行数据
        # row = query_embeddings["sparse"][0]
        # # 获取稀疏向量的非零值索引
        # indices = row.indices
        # # 获取稀疏向量的非零值
        # values = row.data
        # ====================
        try:
            # 新版本 milvus-model 使用 coo_array 格式
            row = query_embeddings["sparse"][0]
            if hasattr(row, 'col'):  # coo_array 格式
                indices = row.col
                values = row.data
            else:  # csr_matrix 格式
                indices = row.indices
                values = row.data
        except Exception as e:
            # 兼容旧版本 milvus-model
            row = query_embeddings["sparse"].getrow(0)
            indices = row.indices
            values = row.data

        # 将索引和值配对，填充稀疏向量字典
        for idx, value in zip(indices, values):
            sparse_query_vector[idx] = value
        # print(f'sparse_query_vector-->{sparse_query_vector}')
        # 初始化过滤表达式，默认不过滤
        filter_expr = f"source == '{source_filter}'" if source_filter else ""
        # print(f'filter_expr--》{filter_expr}')
        # 创建稠密向量搜索请求
        dense_request = AnnSearchRequest(
            data=[dense_query_vector],
            anns_field="dense_vector",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=k,
            expr=filter_expr
        )
        # 创建稀疏向量搜索请求
        sparse_request = AnnSearchRequest(
            data=[sparse_query_vector],
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {}},
            limit=k,
            expr=filter_expr
        )

        # 创建加权排序器，稀疏向量权重 0.7，稠密向量权重 1.0
        ranker = WeightedRanker(1.0, 0.7)
        # 执行混合搜索，返回 Top-K 结果
        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_request, sparse_request],
            ranker=ranker,
            limit=k,
            output_fields=["text", "parent_id", "parent_content", "source", "category", "document_name", "timestamp"]
        )[0]
        child_hits = []
        for hit in results:
            entity = hit["entity"]
            distance = hit.get("distance") if hasattr(hit, "get") else getattr(hit, "distance", None)
            child_hits.append({
                "text": entity.get("text"),
                "parent_id": entity.get("parent_id"),
                "source": entity.get("source"),
                "category": entity.get("category", entity.get("source")),
                "document_name": entity.get("document_name"),
                "milvus_score": float(distance) if distance is not None else None,
            })
        # print(f'results--》{results}')
        # print(f'results--》{type(results)}')
        # print(f'results--》{len(results)}')
        # 将上述搜索到的结果进行Document对象封装，便于查询使用
        sub_chunks = [self._doc_from_hit(hit) for hit in results]
        # print(f'sub_chunks--》{len(sub_chunks)}')
        # 从子块中提取去重的父文档
        parent_docs = self._get_unique_parent_docs(sub_chunks)
        # print(f'parent_docs--》{parent_docs}')
        # print(f'parent_docs--》{len(parent_docs)}')
        # # 如果只有1个文档或者没有，直接返回跳过重排序
        if len(parent_docs) < 2:
            final_docs = parent_docs[:conf.CANDIDATE_M]
            self._record_search_trace(query, k, source_filter, child_hits, final_docs)
            return final_docs
            # 如果有父文档，进行重排序
        if parent_docs:
            # 创建查询与文档内容的配对列表
            pairs = [[query, doc.page_content] for doc in parent_docs]
            # 使用 BGE-Reranker 计算每个配对的得分
            scores = self.reranker.predict(pairs)
            # print(f'scores--》{scores}')
            # 根据得分从高到低排序文档
            scored_docs = []
            for score, doc in zip(scores, parent_docs):
                doc.metadata["reranker_score"] = float(score)
                scored_docs.append((float(score), doc))
            ranked_parent_docs = [doc for _, doc in sorted(scored_docs, key=lambda item: item[0], reverse=True)]
        # 如果没有父文档，返回空列表
        # 如果没有父文档，返回空列表
        else:
            ranked_parent_docs = []

        # 返回前 m 个重排序后的文档
        final_docs = ranked_parent_docs[:conf.CANDIDATE_M]
        self._record_search_trace(query, k, source_filter, child_hits, final_docs)
        return final_docs

    def _record_search_trace(
        self,
        query,
        k,
        source_filter,
        child_hits,
        parent_docs,
        candidate_m=None,
        candidate_parent_count=None,
        timings=None,
    ):
        self.last_search_trace = {
            "query": query,
            "source_filter": source_filter,
            "dense_sparse": True,
            "weighted_ranker": [1.0, 0.7],
            "requested_top_k": k,
            "candidate_m": conf.CANDIDATE_M if candidate_m is None else candidate_m,
            "candidate_parent_count": (
                len(parent_docs) if candidate_parent_count is None else candidate_parent_count
            ),
            "timings": timings or {},
            "child_hits": child_hits,
            "parent_results": [
                {
                    "parent_content": doc.page_content,
                    "parent_id": doc.metadata.get("parent_id"),
                    "source": doc.metadata.get("source"),
                    "category": doc.metadata.get("category"),
                    "document_name": doc.metadata.get("document_name"),
                    "version": doc.metadata.get("version"),
                    "effective_date": doc.metadata.get("effective_date"),
                    "expire_date": doc.metadata.get("expire_date"),
                    "status": doc.metadata.get("status"),
                    "version_group": doc.metadata.get("version_group"),
                    "milvus_score": doc.metadata.get("milvus_score"),
                    "reranker_score": doc.metadata.get("reranker_score"),
                }
                for doc in parent_docs
            ],
        }

    def _get_unique_parent_docs(self, sub_chunks):
        # 初始化集合，用于存储已处理的父块内容（去重）
        parent_contents = set()
        # 初始化列表，用于存储唯一父文档
        unique_docs = []
        # 遍历所有子块
        for chunk in sub_chunks:
            # 获取子块的父块内容，默认为子块内容
            parent_content = chunk.metadata.get("parent_content", chunk.page_content)
            # 检查父块内容是否非空且未重复
            if parent_content and parent_content not in parent_contents:
                # 创建新的 Document 对象，包含父块内容和元数据
                unique_docs.append(Document(page_content=parent_content, metadata=chunk.metadata))
                # 将父块内容添加到去重集合
                parent_contents.add(parent_content)
            # 返回去重后的父文档列表
        return unique_docs

    # 定义类似私有方法，从 Milvus 查询结果创建 Document 对象
    def _doc_from_hit(self, hit):
        # 创建并返回 Document 对象，填充内容和元数据
        entity = hit["entity"]
        hybrid_score = hit.get("distance") if hasattr(hit, "get") else getattr(hit, "distance", None)
        return Document(
            page_content=entity.get("text"),
            metadata={
                "parent_id": entity.get("parent_id"),
                "parent_content": entity.get("parent_content"),
                "source": entity.get("source"),
                "category": entity.get("category", entity.get("source")),
                "document_name": entity.get("document_name"),
                "timestamp": entity.get("timestamp"),
                "version": entity.get("version", "1.0"),
                "effective_date": entity.get("effective_date", "2000-01-01"),
                "expire_date": entity.get("expire_date", "9999-12-31"),
                "status": entity.get("status", "active"),
                "version_group": entity.get(
                    "version_group", entity.get("document_name")
                ),
                "milvus_score": float(hybrid_score) if hybrid_score is not None else None,
            }
        )


if __name__ == "__main__":
    vector_store = VectorStore()  # 建表
    directory_path = os.path.join(rag_qa_path, "data", "membership_data")
    print(f"embedding_function.dim--》{vector_store.embedding_function.dim}")
    documents = process_documents(directory_path)  # 文档读取与切块
    vector_store.add_documents(documents)  # 子块编码并且写入Milvus

    # query = "会员积分什么时候过期？"
    # results = vector_store.hybrid_search_with_rerank(query, source_filter='membership')
    # print(f'results-->{results}')
    # print(f'results-->{len(results)}')
