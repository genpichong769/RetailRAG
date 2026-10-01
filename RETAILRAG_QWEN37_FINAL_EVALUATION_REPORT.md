# RetailRAG Final Full Evaluation V1

## FINAL RESULT

**FINAL MODEL**  
qwen3.7-plus

**QUALITY JUDGEMENT**  
GOOD_WITH_LIMITATIONS

## DATA SCALE

| Dataset | Count |
|---|---:|
| FAQ | 200 |
| Documents | 80 |
| FAQ Eval | 480 |
| Retrieval Eval | 235 |
| Generation Eval | 80 |
| Full RAG Eval | 350 |
| Multi-turn | 30 |

## FAQ

| Metric | Result |
|---|---:|
| Precision | 1.000000 |
| Recall | 0.760000 |
| F1 | 0.863636 |
| FAR | 0.000000 |
| Complex Wrong Intercept | 0 |

## RETRIEVAL

| Metric | Hybrid + Reranker |
|---|---:|
| Recall@1 | 0.701064 |
| Recall@3 | 0.901064 |
| Recall@5 | 0.905674 |
| MRR | 0.877447 |
| All Recall@3 | 0.677419 |

## RERANKER ABLATION

| Mode | Top1 Accuracy | Recall@1 | Recall@3 | MRR |
|---|---:|---:|---:|---:|
| Hybrid | 0.787234 | 0.677660 | 0.874113 | 0.853830 |
| Hybrid + Reranker | 0.821277 | 0.701064 | 0.901064 | 0.877447 |

Improved / Unchanged / Degraded: **24 / 197 / 14**.

## DYNAMIC M ABLATION

| Mode | Any@3 | All@3 | Context Precision | Context Recall | Avg Contexts | Avg Chars | Token Proxy | Multi-doc All@3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed M=2 | 0.893617 | 0.761702 | 0.523404 | 0.828369 | 1.965957 | 1141.280851 | 634.038298 | 0.435484 |
| Dynamic M | 0.893617 | 0.778723 | 0.507801 | 0.837943 | 2.072340 | 1202.225532 | 667.902128 | 0.500000 |

## QUERY REWRITE

| Metric | Before | After |
|---|---:|---:|
| Hit@1 | 0.300000 | 0.766667 |
| Hit@3 | 0.400000 | 1.000000 |
| MRR | 0.350000 | 0.872222 |

Rewrite Success / Error: **1.000000 / 0.000000**. Improved / Unchanged / Degraded: **20 / 8 / 2**.

## EVIDENCE GATE

| Metric | Result |
|---|---:|
| Precision | 0.957447 |
| Recall | 1.000000 |
| F1 | 0.978261 |
| False Answer Rate | 0.000000 |
| False Refusal Rate | 0.036364 |
| Answerable Accuracy | 0.963636 |

## GENERATION / RAGAS

| Metric | Result |
|---|---:|
| Execution | 80/80 |
| Faithfulness | 0.981250 |
| Answer Relevance | 0.986250 |
| Context Precision | 0.571250 |
| Context Recall | 0.771875 |

### Generation 80 by category

| Category | Count | Faithfulness | Answer Relevance | Context Precision | Context Recall |
|---|---:|---:|---:|---:|---:|
| hard_confusion | 10 | 1.000000 | 1.000000 | 0.547000 | 0.860000 |
| keyword_exact | 5 | 1.000000 | 1.000000 | 0.760000 | 1.000000 |
| multi_document | 15 | 0.933333 | 0.993333 | 0.940000 | 0.943333 |
| multiturn | 10 | 1.000000 | 0.900000 | 0.600000 | 0.950000 |
| semantic_paraphrase | 5 | 1.000000 | 1.000000 | 0.400000 | 0.800000 |
| single_document | 10 | 0.950000 | 1.000000 | 0.583000 | 0.900000 |
| unanswerable | 15 | 1.000000 | 1.000000 | 0.120000 | 0.266667 |
| version_conflict | 10 | 1.000000 | 1.000000 | 0.670000 | 0.750000 |

## END-TO-END 350

| Metric | Result |
|---|---:|
| Execution | 350/350 |
| Correct | 229 |
| Partial | 43 |
| Incorrect | 9 |
| Correct Refusal | 48 |
| Wrong Refusal | 21 |
| Execution Error | 0 |
| Strict Accuracy | 0.791429 |
| Partial Rate | 0.122857 |
| Incorrect Rate | 0.025714 |
| Wrong Refusal Rate | 0.060000 |
| Answerable Accuracy | 0.750820 |
| Unanswerable Refusal Accuracy | 1.000000 |

### End-to-end 350 by category

| Category | Count | Correct | Partial | Incorrect | Correct Refusal | Wrong Refusal | Strict Accuracy | Partial Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| faq | 40 | 31 | 4 | 1 | 0 | 4 | 0.775000 | 0.100000 |
| hard_confusion | 20 | 13 | 2 | 0 | 1 | 4 | 0.700000 | 0.100000 |
| keyword_exact | 30 | 26 | 2 | 0 | 0 | 2 | 0.866667 | 0.066667 |
| multi_document | 60 | 37 | 19 | 2 | 1 | 1 | 0.633333 | 0.316667 |
| multiturn | 30 | 24 | 3 | 1 | 0 | 2 | 0.800000 | 0.100000 |
| semantic_paraphrase | 40 | 28 | 3 | 1 | 0 | 8 | 0.700000 | 0.075000 |
| single_document | 55 | 44 | 7 | 3 | 1 | 0 | 0.818182 | 0.127273 |
| unanswerable | 45 | 0 | 0 | 0 | 45 | 0 | 1.000000 | 0.000000 |
| version_conflict | 30 | 26 | 3 | 1 | 0 | 0 | 0.866667 | 0.100000 |

## 主要失败阶段

1. FAQ_ROUTING: 20
2. EVIDENCE_GATE: 19
3. GENERATION: 15

## LATENCY

单位：毫秒。P99 由完整逐题观测值计算。

| Component | Count | Mean | Median | P95 | P99 |
|---|---:|---:|---:|---:|---:|
| faq | 350 | 766.334660 | 788.241000 | 1602.048150 | 1775.879490 |
| query_rewrite | 285 | 1334.581747 | 0.003000 | 11718.817000 | 20325.097360 |
| query_classifier | 285 | 104.529109 | 93.462000 | 189.671800 | 234.524200 |
| strategy_selector | 285 | 6898.753270 | 4716.580000 | 18734.721000 | 33286.787960 |
| embedding | 283 | 454.970053 | 423.159000 | 603.128900 | 1217.320880 |
| milvus | 283 | 5.832597 | 5.462000 | 7.278700 | 16.028580 |
| reranker | 283 | 4308.185682 | 4030.405000 | 6057.307900 | 11352.978020 |
| evidence_gate | 285 | 0.535782 | 0.494000 | 0.814600 | 1.014800 |
| llm_first_token | 222 | 14304.402185 | 12963.990000 | 25412.148550 | 35091.062410 |
| llm_total | 222 | 16973.329982 | 15268.104000 | 31992.127700 | 39330.196560 |
| full_rag | 350 | 26685.525714 | 24130.000000 | 59988.350000 | 76049.110000 |

## QWEN3.7-PLUS TOKEN USAGE

| Evaluation | Prompt Tokens | Completion Tokens | Total Tokens | Average Total Tokens / Case |
|---|---:|---:|---:|---:|
| Generation 80 | 164847 | 179926 | 344773 | 4309.662500 |
| Full RAG 350 production | 225088 | 281482 | 506570 | 1447.342857 |
| Full RAG 350 correctness judge | 377033 | 356099 | 733132 | 2094.662857 |
| Query Rewrite 30 | 3666 | 17469 | 21135 | 704.500000 |

### Average production model calls by strategy

| Strategy | Average calls / case |
|---|---:|
| FAQ Direct | 0.000000 |
| Backtracking | 3.038961 |
| Direct | 1.795812 |
| HyDE | 2.750000 |
| SubQuery | 3.000000 |
| Multi-turn | 2.966667 |
| General | 2.000000 |

StrategySelector 未暴露 Usage，因此 Token 总量仅汇总 API 实际返回字段，没有估算缺失值。

## ENGINEERING

| Check | Result |
|---|---|
| MySQL | PASS |
| Redis | PASS |
| Milvus | PASS |
| MinIO | PASS |
| etcd | PASS |
| FastAPI | PASS |
| WebSocket | PASS |
| Browser E2E | PASS |

Known engineering limitation: one async acceptance client sent keepalive pings during a 95.23-second synchronous retrieval and timed out; the server completed the request, and the 120/180-second WebSocket suites passed. A single Uvicorn worker can delay health and ping responses while CPU-bound retrieval is running.

## HISTORICAL_MODEL_EXPERIMENTS

- qwen3-max results are retained as historical evidence only.
- qwen3.6-plus Generation 80/80 and incomplete Full RAG 133/350 are retained separately.
- qwen3.6-flash incomplete results are retained separately.
- No historical or mixed-model row is included in the qwen3.7-plus final metrics.

## Final assessment questions

1. **RetailRAG 当前实际效果怎么样？** 端到端严格准确率为 0.791429，质量结论为 **GOOD_WITH_LIMITATIONS**。Generation 的 Faithfulness 为 0.981250，但完整效果仍受检索覆盖、上下文完整性和拒答决策共同影响。
2. **哪类问题效果最好？** `unanswerable`，Strict Accuracy 为 1.000000。
3. **哪类问题效果最差？** `multi_document`，Strict Accuracy 为 0.633333。
4. **FAQ 是否稳定？** 精确问法稳定：Exact Accuracy=1.000000、FAR=0.000000。整体还不能称为完全稳定：paraphrase recall 只有 0.640000，且 350 端到端中有 20 条失败归因于 FAQ_ROUTING，主要是非纯 FAQ 问题被 FAQ_DIRECT 截获后缺少检索证据。
5. **Dense + Sparse + Reranker 是否有效？** 有效，但增益温和。相对 Hybrid，Recall@3 从 0.874113 提升到 0.901064，MRR 从 0.853830 提升到 0.877447；Improved/Unchanged/Degraded 为 24/197/14。
6. **多文档完整召回是否仍是主要问题？** Dynamic M 的多文档 All@3 为 0.500000，低于单文档表现，仍是关键瓶颈。
7. **Query Rewrite 是否值得保留？** 值得。Hit@3 从 0.400000 提升到 1.000000，MRR 从 0.350000 提升到 0.872222，错误率为 0.000000。
8. **Evidence Gate 是否过度拒答？** 控制集上的 False Refusal Rate 为 0.036364，并不高；但 350 端到端出现 21 条 Wrong Refusal，其中 19 条归因于 EVIDENCE_GATE，说明真实分布上存在明显过度拒答，需要下一轮优先复核。
9. **qwen3.7-plus Generator 是否仍是瓶颈？** 是显著瓶颈，但不是第一瓶颈。GENERATION 归因 15 条，少于 FAQ_ROUTING 的 20 条和 EVIDENCE_GATE 的 19 条；同时 Generation 80 的 Faithfulness/Answer Relevance 达到 0.981250/0.986250，说明上游路由与拒答对端到端结果影响更大。
10. **Context Precision / Recall 为什么处于当前水平？** Context Precision=0.571250 反映候选中仍含旁支证据；Context Recall=0.771875 主要受 K=3 候选覆盖与多文档完整召回限制。Dynamic M 增加复杂题上下文，但不能恢复未进入候选池的文档。
11. **投实习时哪些指标适合写简历？** 可以写固定 200 FAQ、80 文档、480/235/80/350/30 评测规模；FAQ F1/FAR；Hybrid+Reranker Recall@3 与 MRR；Query Rewrite 前后增益；Evidence Gate Precision/Recall/F1；Generation 四项指标；以及 350 条严格准确率。应同时注明统一模型与固定 Benchmark。
12. **如果只能再优化一轮，优先改哪里？** 优先修 FAQ_ROUTING：它贡献 20 条失败，是最大单一阶段。目标应是减少带上下文、组合条件、多文档或多轮问题被 FAQ_DIRECT 过早截获，同时守住 Exact Accuracy=1.000000 和 FAR=0.000000；随后再处理 19 条 EVIDENCE_GATE 失败。

## Method and limitations

- Generator、Generation Judge 与 350 正确性 Judge 均为 qwen3.7-plus，同模型裁判可能偏乐观。
- Benchmark、Ground Truth、Prompt、阈值与业务逻辑在正式运行期间均未修改。
- Token Usage 只汇总 API 实际返回的字段；未返回部分标为 `NOT_AVAILABLE`，没有估算。
