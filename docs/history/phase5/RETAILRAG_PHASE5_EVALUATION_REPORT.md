# RetailRAG PHASE 5 质量优化与评测报告

评测日期：2026-09-23  
最终状态：**PASS WITH DOCUMENTED LIMITATIONS**

本轮在 PHASE 4.1 已跑通链路上完成优化，没有修改原 EduRAG。开始前记录的 Git 基线为 `baaabd71d0272fc46b127a4e3f0b897b17e10e42`；保护备份为 `E:\EduRAG智慧回答项目\RetailRAG_PHASE4_1_BASELINE_20260923_1803.zip`，SHA256 为 `C663ADAB49378A1B1F2AAA2B10A3F832A827BF19B559C30EE6F63FA1044DFB9D`。

## 1. 评测资产

| 评测集 | 数量 | 构成 |
|---|---:|---|
| FAQ Evaluation Set | 280 | exact 52、paraphrase 156、hard negative 52、non-FAQ 20 |
| 完整 RAG Evaluation Set | 190 | FAQ 30、单文档 35、多文档 35、关键词 20、语义改写 20、版本冲突 20、无答案 20、多轮 10 |
| Retrieval 消融子集 | 130 | 单文档、多文档、关键词、语义改写、版本冲突 |
| Evidence Gate 校准集 | 45 | answerable 25、unanswerable 20 |
| Generation Evaluation | 32 | 来自 7 种问题类型的代表样本 |
| 多轮 Query Rewrite | 10 组 | 每组包含历史与短追问 |

完整 RAG 集已校验：总数和分类数量正确、问题唯一、没有直接复制文档标题、所有 reference document 均存在。源文件为 `RETAILRAG_FAQ_EVALUATION_SET.json` 和 `RETAILRAG_RAG_EVALUATION_SET.json`。

## 2. FAQ 基线

基线为 Jieba + BM25 + Softmax 0.85，在同一 280 条独立集上运行：

| 指标 | 基线 |
|---|---:|
| Exact FAQ Accuracy | 1.000000 |
| Paraphrase Recall | 0.391026 |
| Precision | 0.795775 |
| Recall | 0.543269 |
| F1 | 0.645714 |
| False Accept Rate | 0.333333 |
| False Reject Rate | 0.432692 |

主要原因有两点：中文词面变化使 BM25 对语义改写召回不足；Softmax 0.85 是候选间相对权重，不是“答案正确概率”，即使所有 FAQ 都不相关，某个词面重合项也可能得到很高的相对分数。这正是 PHASE 4.1 错误截获的通用原因。

原始明细见 `RETAILRAG_FAQ_BASELINE_EVALUATION.json`。

## 3. FAQ 候选方案与最终选择

所有方案复用现有 BGE-M3，没有新增模型：

| 方案 | 最佳阈值 | Paraphrase Recall | Precision | Recall | F1 | FAR |
|---|---:|---:|---:|---:|---:|---:|
| BM25 Only | 0.49 | 0.596154 | 0.677570 | 0.697115 | 0.687204 | 0.541667 |
| Dense Only | **0.84** | **0.756410** | **0.965909** | **0.817308** | **0.885417** | **0.013889** |
| BM25 + Dense，词面权重 0.20 | 0.88 | 0.647436 | 0.974522 | 0.735577 | 0.838356 | 0.013889 |
| BM25 初筛 + Dense 二判 | 0.84 | 0.576923 | 0.965986 | 0.682692 | 0.800000 | 0.013889 |

最终采用 **BGE-M3 Dense Only + 0.84**。选择规则是在 Exact Accuracy ≥ 98%、FAR ≤ 2% 的可行方案中优先最大化 F1 和改写召回。

这仍是轻量 FAQ 路径：启动时对 52 条标准问题生成内存向量，查询时只做一次 BGE-M3 query embedding 和 52 行点积；不进入 Milvus、不运行 Reranker、不调用生成模型。FAQ 命中答案仍从 MySQL 获取，Redis 缓存键加入 `dense-bgem3-v1` 算法版本，避免旧 BM25 误答缓存污染。

### 优化前后

| 指标 | BM25 0.85 | Dense 0.84 | 变化 |
|---|---:|---:|---:|
| Exact Accuracy | 1.000000 | 1.000000 | 0 |
| Paraphrase Recall | 0.391026 | 0.756410 | +0.365384 |
| Precision | 0.795775 | 0.965909 | +0.170134 |
| Recall | 0.543269 | 0.817308 | +0.274039 |
| F1 | 0.645714 | 0.885417 | +0.239703 |
| False Accept Rate | 0.333333 | 0.013889 | -0.319444 |
| False Reject Rate | 0.432692 | 0.158654 | -0.274038 |

PHASE 4.1 回归问题“金卡会员的保级条件是什么？”在 FAQ 层的 Dense 分数为 0.681368，低于 0.84，已正确回退 RAG；“法国的首都是什么？”为 0.357048，也正确拒绝。修复没有针对问题文本 hardcode。

残余 FAQ 个案：38 条 paraphrase 未正确命中；1 条 hard negative（“价签与 POS 价格不一致时更换的价签用什么字体？”）仍被相邻 FAQ 接受。该项计入限制，没有隐藏。

详见 `RETAILRAG_FAQ_METHOD_COMPARISON.json`、`RETAILRAG_FAQ_THRESHOLD_SWEEP.csv` 和 `RETAILRAG_FAQ_OPTIMIZED_RESULTS.csv`。

## 4. 文档版本与时效性

文档和 chunk 已加入：`document_name`、`category`、`version`、`effective_date`、`expire_date`、`status`、`source`、`version_group`。支持 `active`、`expired`、`superseded`。

本轮构建了同一 `version_group=member_gold_points_rate` 的冲突样本：

- 1.0 / 2025：每 1 元累计 1 积分，`superseded`；
- 2.0 / 2026：每 1 元累计 1.5 积分，`active`。

“现在金卡积分怎么算？”测试结果：active 路径首条为 2.0，排除 1.0；审计用 unfiltered 路径能同时看到两个版本。线上 WebSocket 回答包含当前 1.5 倍与 2026 生效信息，不包含旧口径。

当前集合为 40 篇文档、40 个父块、82 个子块；80 个 active 子块、2 个 superseded 子块。重复入库后可见实体和唯一 ID 均保持 82，逻辑幂等。

详见 `RETAILRAG_VERSION_CONFLICT_RESULTS.json`。

## 5. 代码级可靠拒答

拒答不再只依赖 Prompt。internal 查询完成检索后，代码会检查：

1. 是否存在有效上下文；
2. 文档是否为 active；
3. `source` 和 `document_name` 是否完整；
4. Top Reranker score 是否达到实测阈值。

真实分数分布：answerable 最低 0.183327、中位数 0.982977；unanswerable 最高 0.017679、中位数 0.000416。阈值扫描选定 **0.020635**。

| 指标 | 结果 |
|---|---:|
| Answerable Accuracy（未误拒） | 1.000000 |
| Refusal Precision | 1.000000 |
| Refusal Recall | 1.000000 |
| Refusal F1 | 1.000000 |
| False Accept Unanswerable | 0 / 20 |
| False Refuse Answerable | 0 / 25 |

低证据时直接返回“当前知识库信息不足”，`llm_called=false`，不会让 LLM 猜答案。在线用例“Nova Retail 今年营业收入多少？”也通过。

详见 `RETAILRAG_EVIDENCE_GATE_EVALUATION.json`。

## 6. Retrieval 消融

评测模式独立存在，不改变线上 Hybrid + Reranker、K=3、M=2 链路。消融以 candidate/return K=5 跑 130 条：

| 模式 | Any Recall@1 | Any Recall@3 | Any Recall@5 | All Recall@3 | All Recall@5 | MRR | Hit Rate@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sparse Only | 0.815385 | 0.953846 | 0.969231 | 0.907692 | 0.938462 | 0.884615 | 0.969231 |
| Dense Only | 0.900000 | 0.976923 | 0.992308 | 0.915385 | 0.953846 | 0.940256 | 0.992308 |
| Hybrid | 0.892308 | 0.992308 | 0.992308 | 0.923077 | 0.953846 | 0.939744 | 0.992308 |
| Hybrid + Reranker | **0.923077** | **0.992308** | **0.992308** | **0.953846** | **0.953846** | **0.956410** | **0.992308** |

“Any”表示命中任一 reference；“All”表示覆盖全部必要 reference。

### Reranker 增益

相对 Hybrid，Reranker 带来：

- Recall@1：+0.030769；
- MRR：+0.016666；
- All Recall@3：+0.030769；
- 线上 M=2 Context Precision：0.565385 → 0.588462，+0.023077；
- Recall@3/5 与 Context Precision@5 不变，因为 Reranker 只重排同一组候选。

结论：Reranker 的真实价值是把已有正确候选前移，对线上只取 2 个父块的场景有小幅但可测的收益，因此保留。

详见 `RETAILRAG_RETRIEVAL_ABLATION.json` 和 `RETAILRAG_RETRIEVAL_ABLATION_RESULTS.csv`。

## 7. Generation Evaluation

32 条样本全部通过当前真实 `RAGSystem` 现场检索和生成，再由现有 Qwen 接口作为 grounded judge；没有用预写 context 或 answer 替代 Retriever 输出。

| 指标 | 结果 |
|---|---:|
| Faithfulness | 1.000000 |
| Answer Relevance | 1.000000 |
| Context Precision | 1.000000 |
| Context Recall | 0.937500 |
| 无答案样本拒答率 | 1.000000 |

直接 RAGAS 0.3 路径本轮未使用；采用用户允许的“其他生成指标”方案，以现有已认证 Qwen endpoint 做四维 grounded judge，避免为评测另接模型。限制是生成模型与裁判属于同一模型家族，前三项满分可能偏乐观，因此不应把这组分数当成独立人工验收。

详见 `RETAILRAG_GENERATION_EVALUATION.json`，其中保存每条现场答案、检索文档、证据门决定和裁判理由。

## 8. 多轮 Query Rewrite

实现方式是用最近历史与当前短追问生成一条独立检索问题；不会把整个历史直接拼进 Milvus query。原问题和改写问题均进入 trace。

| 指标 | 无 Rewrite | 有 Rewrite | 增益 |
|---|---:|---:|---:|
| Hit@1 | 0.300000 | 0.700000 | +0.400000 |
| Hit@3 | 0.500000 | 1.000000 | +0.500000 |
| MRR | 0.403333 | 0.833333 | +0.430000 |

10 组全部完成，收益明显，线上默认启用。详见 `RETAILRAG_QUERY_REWRITE_EVALUATION.json`。

## 9. 展示与占位清理

`your_phone_here` 已替换为“门店值班经理或内部服务台”。重新扫描正式 `README.md` 与 `static/index.html`：`your_`、`TODO`、`Edu`、`教育`、`Itcast`、`传智`、`student`、`course` 无不必要可见命中；`example` 只在 README 中作为真实文件名 `.env.example` 和 `config.example.ini` 出现。

详见 `RETAILRAG_PLACEHOLDER_SCAN.json`。

## 10. 在线回归与工程检查

最终在线回归 8/8：health、正式页面、FAQ 原句、FAQ 改写、PHASE 4.1 FAQ 边界回归、非 FAQ 边界、代码级拒答、active 版本回答全部通过。

此外：

- `python -m compileall`：PASS；
- `git diff --check`：PASS，仅有 Git 的 LF/CRLF 提示；
- 五个 RetailRAG Docker 服务最终均为 Up；
- Docker Desktop 曾因虚拟磁盘挂载 I/O error 导致 Milvus/etcd 退出，重启 Docker Desktop 后恢复，之后所有在线评测通过；该项已恢复，不列为最终 BLOCKED；
- FastAPI 正式页面改为绝对静态路径，消除依赖启动目录的问题。

详见 `RETAILRAG_PHASE5_ONLINE_ACCEPTANCE.json`。

## 11. 所有 FAIL、BLOCKED 与限制

### FAIL

最终必选验收项没有未恢复的 FAIL。保留以下样本级失败，不粉饰：

- FAQ：38 条 paraphrase 失败，1 条 hard negative 被接受；
- Retrieval：130 条中 Any Recall@5 为 129/130；All Recall@5 为 124/130；
- Generation Context Recall 为 0.9375，不是满分；
- Reranker 对 Recall@3/5 和 Context Precision@5 没有提升，其收益集中在 Top-1/MRR/M=2。

### BLOCKED

最终没有 BLOCKED。直接 RAGAS 没有作为本轮裁判框架使用，但 Generation Evaluation 已用允许的 grounded LLM-as-judge 路径真实完成；这属于方法选择与已披露限制，不是伪造结果或未完成评测。

### 需要以后重校准的条件

- FAQ 标准问题或业务分布明显变化时，重跑 280 条集并重新选 0.84；
- 知识库规模、领域或 Reranker 模型变化时，重跑 45 条 evidence set，不应永久固定 0.020635；
- 简历或对外材料应同时注明评测集规模，不能只写满分指标。

## 12. 最终推荐线上配置

```text
FAQ_RETRIEVAL_MODE=dense
FAQ_MATCH_THRESHOLD=0.84
FAQ_ALGORITHM_VERSION=dense-bgem3-v1

正常 RAG：Hybrid + Reranker
RETRIEVAL_K=3
CANDIDATE_M=2
EVIDENCE_RERANKER_THRESHOLD=0.020635
QUERY_REWRITE_ENABLED=true

版本过滤：status=active + effective_date/expire_date + version_group 最新版本
低证据：代码直接拒答，不调用生成模型
```

简历用汇总指标见 `RETAILRAG_FINAL_METRICS.json`。所有详细 JSON/CSV 均保留原始逐条结果，可复核本报告中的数字。
