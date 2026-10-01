# RetailRAG

**连锁零售企业私有知识库智能问答系统**

## 项目背景与使用场景

RetailRAG 使用自建模拟零售企业 Nova Retail 的私有知识数据构建，是单机演示项目。资料中的规则、金额、时限和活动均为演示内容，不代表真实企业承诺，也不是真实公司的生产数据。系统面向会员、促销、售后、门店运营、商品、设备、制度与通知问答，可展示 FAQ 快速命中、知识检索、证据约束回答和可复现评测；尚未进行生产环境验收。

## 架构与技术栈

- FAQ：MySQL 保存 200 条冻结问答，Redis 保存版本化缓存；jieba/BM25、BGE-M3 Dense 与 FAQ Complexity Gate 负责匹配与分流。
- RAG：QueryClassifier → 多轮 Query Rewrite → StrategySelector（Direct、HyDE、SubQuery、Backtracking）→ BGE-M3 Dense + Sparse → Milvus Hybrid Search → Parent-Child Chunking 的父块恢复与去重 → BGE Reranker → Dynamic M → Version Logic → Evidence Gate → Qwen。
- 服务：FastAPI 提供 HTTP、WebSocket 流式回答及静态前端；依赖 MySQL、Redis、Milvus、etcd 和 MinIO。代码以项目根目录计算本地路径。

```text
用户问题 → FAQ/Redis/MySQL
       ├─ 可靠命中：直接回答
       └─ 需要检索：分类/改写/策略 → Milvus 混合检索 → 父块与版本过滤
                                      → 重排/动态 M/证据门 → qwen3.7-plus
```

## 目录

| 路径 | 内容 |
|---|---|
| `app.py`, `new_main.py` | FastAPI 与问答入口 |
| `base/`, `mysql_qa/`, `rag_qa/` | 配置、FAQ、检索与生成核心代码 |
| `mysql_qa/data/retail_faq.csv` | 冻结的 200 条 FAQ |
| `rag_qa/data/*_data/` | 八个业务域的 80 篇冻结文档，含版本冲突样本 |
| `rag_qa/classify_data/` | 1000 条分类数据及拆分 |
| `rag_qa/models/retail-query-classifier/` | 正式分类器，权重使用 Git LFS |
| `scripts/` | 初始化、入库、评测与 Smoke Test |
| `static/` | 当前静态前端 |
| `docs/history/` | 精选历史实验 |
| `RETAILRAG_QWEN37_FINAL_*` | 不可改写的 qwen3.7-plus 正式评测结果 |

## 环境准备

Windows 的已记录依赖在 `requirements-windows.txt`；根目录 `requirements.txt` 引用它。macOS 可参照 `requirements-mac.txt`，但最终迁移验证在 Windows 上完成。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
git lfs pull
Copy-Item .env.example .env
```

在 `.env` 填写 MySQL、MinIO 与 DashScope 凭据。该文件被 Git 忽略。当前正式模型为 `qwen3.7-plus`。BGE-M3 与 BGE Reranker 不随仓库分发；默认从模型 ID 下载到标准 Hugging Face 缓存，也可用 `BGE_M3_PATH`、`BGE_RERANKER_PATH` 指向已有模型目录。首次下载约需数 GB 空间。分类器权重由 Git LFS 获取。

## Docker 与数据初始化

在新机器、端口未被占用且 `.env` 已配置后，从项目根目录执行：

```powershell
docker compose up -d
python scripts/init_mysql.py
python scripts/init_faq.py
python scripts/init_milvus.py
python scripts/ingest_knowledge_base.py --replace
```

根目录 Compose 启动 MySQL、Redis、Milvus、MinIO、etcd；持久化目录位于 `docker/` 下并被 Git 忽略。MySQL 初始化脚本创建表并导入 200 条 FAQ；FAQ 脚本还清理 RetailRAG 命名空间缓存；Milvus 脚本从 80 篇 Markdown 重建向量集合。已有数据的环境不要无意运行 `--replace`。

## FastAPI 与 WebSocket

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8003
```

打开 [http://127.0.0.1:8003/](http://127.0.0.1:8003/)。`GET /health` 是进程级检查，`POST /api/create_session` 创建会话，`POST /api/query` 处理非流式请求，`WS /api/stream` 提供流式回答，`GET /api/history/{session_id}` 查询历史，`GET /api/sources` 返回业务域。WebSocket 消息格式可参考 `static/index.html` 和 `scripts/test_api_online.py`。

## 冻结评测与结果

冻结数据：FAQ 480、检索 235、RAG 350、多轮 30、生成 80。FAQ、知识文档和评测输入共 84 个文件的 SHA-256 基线记录于 `RETAILRAG_SCALE_DATA_AUDIT.json`。运行 `python scripts/verify_frozen_assets.py` 可只读核对计数、哈希和正式交付物，无需调用 Qwen。

正式结果使用 `qwen3.7-plus`，状态为 **GOOD_WITH_LIMITATIONS**。详见 `RETAILRAG_QWEN37_FINAL_EVALUATION_REPORT.md` 和 `RETAILRAG_QWEN37_FINAL_METRICS.json`。

| 指标 | 正式结果 |
|---|---:|
| FAQ Precision / Recall / F1 / FAR | 1.0000 / 0.7600 / 0.8636 / 0 |
| Retrieval Recall@3 / MRR | 0.9011 / 0.8774 |
| Multi-document All Recall@3 | 0.6774 |
| Query Rewrite Hit@3 | 0.4000 → 1.0000 |
| Query Rewrite MRR | 0.3500 → 0.8722 |
| Evidence Gate Precision / Recall / F1 | 0.9574 / 1.0000 / 0.9783 |
| Generation Faithfulness / Answer Relevance | 0.9813 / 0.9863 |
| Full RAG 执行 / Strict Accuracy | 350/350 / 0.7914 |

Full RAG 分类为 Correct 229、Partial 43、Incorrect 9、Correct Refusal 48、Wrong Refusal 21，Execution Error 0。主要失败阶段为 FAQ_ROUTING 20、EVIDENCE_GATE 19、GENERATION 15。

## 已知限制

多文档检索 All Recall@3 为 0.6774，多文档端到端 Strict Accuracy 为 0.6333。服务健康检查不覆盖全部依赖；生成依赖有效的 DashScope 额度。以上是冻结实验结果，不应视为生产环境 SLA。

## 迁移审计

`RETAILRAG_FINAL_PROJECT_MIGRATION_REPORT.md` 记录目标验证和运行边界；`RETAILRAG_CLEANUP_MANIFEST.json` 逐文件记录复制、归档及排除动作。源项目在迁移中未删除。
