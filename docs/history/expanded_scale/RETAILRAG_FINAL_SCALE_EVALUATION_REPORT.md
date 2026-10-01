# RetailRAG 最终数据扩充与复测报告

评测日期：2026-09-29。结论：**完成复测，存在真实退化，不建议作为质量验收通过版本正式冻结。**

本轮保持核心架构、模型、阈值及线上逻辑不变，只扩充数据和评测工具。原 Itcast_qa_system 未修改。新增虚拟制度用于 Nova Retail 演示，不代表真实企业政策。

## 1. 基线保护与复现

PHASE 5 基线 Git HEAD：`baaabd71d0272fc46b127a4e3f0b897b17e10e42`（HEAD 未改变，但原工作区有大量未提交内容，不能只靠该提交复原）。

完整工作状态保护目录：`E:\EduRAG智慧回答项目\RetailRAG_PHASE5_BACKUP_20260929`。保存 Git status、工作区/暂存区二进制补丁、Git 历史 bundle、源文件副本及 SHA256 清单；原报告、指标、FAQ、40 篇文档和全部旧评测均单独保留。原始要求提到38篇，实际基线为38篇业务文档加2篇版本样本，共40篇。工作区广泛复制中的 MySQL socket 属性复制报错不影响要求的数据源备份；该目录不是数据库一致性恢复备份。

核心代码变化检查：`[]`。唯一修改的原有辅助脚本为 `scripts/init_faq.py`，将写死的52条导入断言改为按CSV计数；业务逻辑不变。

数据固定证据为 `RETAILRAG_SCALE_DATA_AUDIT.json` 的逐文件哈希。新增数据运行前固定，运行后未为指标删除、重标或改写样本。复现入口：`scripts/build_scale_data.py`、`scripts/ingest_knowledge_base.py`、`scripts/init_faq.py`、`scripts/evaluate_scale.py`、`scripts/evaluate_scale_full.py`、`scripts/evaluate_scale_online.py`、`scripts/audit_scale_runtime.py`、`scripts/report_scale.py`。生成结果与重排分数有检查点；重排缓存只复用同模型相同问题/上下文的实测分数，不缓存期望标签。

## 2. 最终数据规模

| 资产 | PHASE 5 | 扩充后 |
|---|---|---|
| FAQ | 52 | 200 |
| 知识文档 | 40 | 80 |
| QueryClassifier总集 | 1000 | 1000 |
| Classifier train/dev/test | 800/100/100 | 800/100/100 |
| 独立人工分类测试 | 50 | 50 |
| FAQ Eval | 280 | 480 |
| RAG Eval | 190 | 350 |
| Retrieval Eval | 130 | 235 |
| 多轮对话 | 10 | 30 |
| 无答案 | 20 | 45 |
| 部分可答独立标记 | 未独立统计 | 5 |
| 生成裁判 | 32 | 80 |
| 全RAG现场执行 | 旧15条线上回归 | 350 |

FAQ分类：{"会员制度": 28, "促销活动": 32, "售后政策": 39, "门店运营": 35, "设备操作": 20, "商品知识": 20, "企业制度": 15, "内部通知": 11}。

文档分类：{"after_sales_data": 12, "equipment_data": 9, "membership_data": 10, "notice_data": 10, "policy_data": 6, "product_data": 8, "promotion_data": 11, "store_operation_data": 14}。

RAG分布：{"single_document": 55, "keyword_exact": 30, "semantic_paraphrase": 40, "multi_document": 60, "faq": 40, "version_conflict": 30, "unanswerable": 45, "multiturn": 30, "hard_confusion": 20}。所有旧190条RAG和280条FAQ评测保留。

## 3. 独立性与难度

FAQ中只有单条规则可回答的固定高频业务决策；改写不作为新的正式FAQ。新增文档含目的、规则、执行、异常与关联资料，包含完整 source/category/version/effective_date/expire_date/status/version_group 元数据。

480条FAQ评测：Exact100、Paraphrase200、Hard Negative90、Non-FAQ90。Exact是明确重复标准问题的查找对照，因此不能把Exact100%称为泛化能力；新增非Exact、RAG题均经FAQ原句和文档原句检查。旧样本原样保存，文字检查只能证明无直接复制，不能证明统计独立或人工标注质量。知识与新增标签来自同一虚拟业务编写过程，尚无外部专家盲评。

混淆主题覆盖银卡/金卡、秒杀/会员日、固定套装/自由凑单、食品/普通品、跨店/线上售后、开店/闭店现金、调拨出库/入库、小票/价签打印机、探头显示/实测温差。跨文档题包含需要3至4篇材料的规则组合，不能靠单篇直接回答完整结论。

## 4. FAQ最终指标

| 指标 | PHASE 5 | 扩充480条 | 扩充库上的旧280条 |
|---|---|---|---|
| precision | 0.965909 | 0.950000 | 0.965909 |
| recall | 0.817308 | 0.760000 | 0.817308 |
| f1 | 0.885417 | 0.844444 | 0.885417 |
| paraphrase_recall | 0.756410 | 0.640000 | 0.756410 |
| false_accept_rate | 0.013889 | 0.038889 | 0.013889 |
| false_reject_rate | 0.158654 | 0.223333 | 0.158654 |

Exact Accuracy=1.000000，Hard Negative Accuracy=0.922222。错误记录79条。新增44条Paraphrase Recall=0.227273。

固定Dense阈值0.84。Precision把错答另一FAQ也计为FP；Recall要求FAQ ID正确；FAR只以负例为分母；FRR为正例被拒绝比例，因此错接受其他FAQ不计FRR，但计入Recall损失。旧集复测保持原指标，整体下降主要来自新增自然表达和多条件硬负例；不能通过更多Exact掩盖。

## 5. Retrieval消融

| 模式 | Any@1 | Any@3 | Any@5 | MRR | Hit Rate@5 | All@3 | All@5 |
|---|---|---|---|---|---|---|---|
| sparse_only | 0.693617 | 0.889362 | 0.923404 | 0.790851 | 0.923404 | 0.800000 | 0.863830 |
| dense_only | 0.748936 | 0.910638 | 0.940426 | 0.825957 | 0.940426 | 0.804255 | 0.868085 |
| hybrid | 0.787234 | 0.927660 | 0.940426 | 0.853830 | 0.940426 | 0.817021 | 0.868085 |
| hybrid_reranker | 0.821277 | 0.936170 | 0.940426 | 0.877447 | 0.940426 | 0.859574 | 0.868085 |

每种模式235条。为与旧消融对齐，诊断candidate K=5；不是更改线上K。Any表示至少命中一篇必要文档，All表示命中全部必要文档。原PHASE 5的Recall使用Any语义。另存真正按必要文档比例计算的document_recall_at_k，防止混淆召回定义。

## 6. 多文档完整召回与线上配置

| 模式 | 多文档样本数 | Any@3 | All@3 | All@5 | 文档比例Recall@3 |
|---|---|---|---|---|---|
| sparse_only | 62 | 0.967742 | 0.629032 | 0.758065 | 0.807796 |
| dense_only | 62 | 0.951613 | 0.548387 | 0.709677 | 0.756720 |
| hybrid | 62 | 0.967742 | 0.548387 | 0.709677 | 0.764785 |
| hybrid_reranker | 62 | 0.967742 | 0.677419 | 0.709677 | 0.834677 |

实际K=3、M=2路径：Any@2=0.893617，All@2=0.761702，Context Precision@2=0.523404。需要3或4篇必要文档的问题无法在2个最终父块内完成全覆盖；这是固定配置下必须披露的边界，不能用Any-hit代替完整召回。

## 7. Reranker价值

| 指标 | Hybrid | Hybrid + Reranker | 增益 |
|---|---|---|---|
| any_at_1 | 0.787234 | 0.821277 | 0.034043 |
| any_at_3 | 0.927660 | 0.936170 | 0.008510 |
| mrr | 0.853830 | 0.877447 | 0.023617 |
| context_precision_at_2 | 0.514894 | 0.544681 | 0.029787 |

Top1 accuracy与Any Recall@1在该相关文档判定下等价，为0.821277。重排只改变既有候选顺序，不增加候选；@5集合指标相等是预期性质。Context Precision按人工参考文档集合计算，与生成裁判的语义Context Precision分开。

## 8. Version Test

30条版本题；线上返回文档的状态/日期合法率=1.000000；必要文档All@2=0.866667。同时核验同库中真实存在superseded、expired和未来生效active记录，详见 `RETAILRAG_SCALE_VERSION_RESULTS.json`。

新增自提6小时旧规/2小时新规、秋季92折旧稿/9折有效稿、夏季秒杀过期特例、六月生日券延期过期通知、十二月未来通知和调拨废止口径。注意：当前实现按执行日期过滤，并不自动按问题所述历史/未来日期检索；过滤安全通过不等于版本语义回答全部正确。旧春季通知缺显式日期metadata仍原样保留，应将其时效风险列入后续治理，不能在本轮偷偷修正。

## 9. Refusal Test

| 指标 | PHASE 5证据门 | 扩充固定证据门 | 扩充端到端回答裁判 |
|---|---|---|---|
| answerable_accuracy | 1.000000 | 0.963636 | 0.909091 |
| refusal_precision | 1.000000 | 0.945946 | 0.936170 |
| refusal_recall | 1.000000 | 0.777778 | 0.977778 |
| false_refusal_rate | 未统计 | 0.036364 | 0.054545 |
| false_answer_rate | 未统计 | 0.222222 | 0.022222 |

证据门：55条可回答+45条无答案，固定阈值0.020635；另5条部分可答不混入二元分母。证据门Answerable Accuracy表示未误拒；端到端Answerable Accuracy由实际答案与参考判断，含语义正确性，因此定义不同。端到端拒答率基于真实输出，经同Qwen家族独立调用裁判，部分可答处理率=0.800000。看起来像内部制度但不存在、诱导编造、实时余额/库存/温度及部分可答均保留，见完整明细。

## 10. Query Rewrite

| 指标 | 旧Without | 旧With | 扩充Without | 扩充With |
|---|---|---|---|---|
| Hit@3 | 0.500000 | 1.000000 | 0.400000 | 1.000000 |
| MRR | 0.403333 | 0.833333 | 0.350000 | 0.944444 |

30组，使用当前RAGSystem的rewrite函数，比较与PHASE 5一致的诊断K=5重排。语义Rewrite Error Rate=0.000000，原本Hit@3成功改写后失败的比例=0.000000。两者分别表示语义错误与检索退化，不混为一项；语义错误由同家族裁判判定，存在偏差。

## 11. Generation Evaluation

| 指标 | PHASE 5（32条） | 扩充（80条） |
|---|---|---|
| faithfulness | 1.000000 | 0.985000 |
| answer_relevance | 1.000000 | 0.965625 |
| context_precision | 1.000000 | 0.787500 |
| context_recall | 0.937500 | 0.774000 |

80条全由当前真实RAGSystem检索和生成，保留上下文、分类、策略、改写、引用、证据门及答案。单文档10、多文档15、关键词5、语义5、版本10、无答案15、多轮10、困难混淆10；困难混淆含5条部分可答。选择规则事先固定，优先新增样本，不按结果挑选。成功裁判80/80。

生成与裁判都使用 `qwen3-max`（Qwen同家族），可能偏乐观，不能声称人工独立验收。新裁判提示明确按实际上下文覆盖评分，比旧提示更严格；两期生成分数的变化同时含样本难度和裁判提示差异，不能全归因于系统退化。未使用原生RAGAS，不将这些分数冒称为RAGAS标准跑分。

## 12. 完整RAG执行

全部350条已通过当前RAGSystem生成（含复用本轮80条真实生成记录，不复用旧PHASE 5答案）。返回答案350条，执行错误0条。

| 诊断项 | 结果 |
|---|---|
| execution_errors | 0 |
| answers_returned | 350 |
| required_doc_any_hit | 0.809836 |
| all_required_covered | 0.675410 |
| answerable_code_false_refusal_rate | 0.042623 |
| unanswerable_code_refusal_rate | 0.733333 |
| citation_presence | 0.973510 |

该路径直接评测RAGSystem；正式HTTP前置FAQ可能截获复杂问题，因此另有FAQ及HTTP/Browser边界测试，不能把直接RAG的成功替代整合路径成功。未被80条生成裁判覆盖的其余输出只做自动诊断，不声称已逐条独立语义判分。

## 13. 入库与metadata

文档80、父块80、子块203。首次和重复入库后的可见实体均203、唯一ID均203，当前Milvus物理统计 `{'row_count': 203}`；刚upsert后物理统计曾分别为285和406，压缩后收敛，不能把短暂版本行计为独立知识实体。

metadata字段完整性=True，source/category一致性=True。未drop集合，复用稳定ID进行upsert；重复执行没有增加可见重复数据。MySQL200条答案全量对照CSV一致；初始化只清理RetailRAG FAQ命名空间缓存。

## 14. 在线与Browser E2E

MySQL、Redis、Milvus、MinIO、etcd、FastAPI均完成真实连接/健康检查。HTTP/WebSocket原始断言11/14，Browser检查6/8。

真实失败：金卡生日月100元积分组合题和现金/设备/库存闭店综合题被单条FAQ提前截获，返回不完整答案。API多轮断言因答案没有出现字面7/七而失败，但实际回答正确处理5天普通商品退货并引用来源；原始失败标记不改，复核意见另存manual_assertion_audit。Browser的FAQ、RAG、general、多轮、版本、拒答与引用展示均已实际操作并记录。

## 15. PHASE 5 vs Expanded Data

| 指标 | 旧 | 扩充 |
|---|---|---|
| FAQ F1 | 0.885417 | 0.844444 |
| FAQ FAR | 0.013889 | 0.038889 |
| Retrieval Any Recall@3 | 0.992308 | 0.936170 |
| Retrieval MRR | 0.956410 | 0.877447 |
| Retrieval All@3 | 0.953846 | 0.859574 |
| Gate Refusal Precision | 1.000000 | 0.945946 |
| Gate Refusal Recall | 1.000000 | 0.777778 |

不同难度与规模的数据集不构成严格受控因果实验。FAQ旧集回放和Retrieval各模式cohorts提供更接近同题对照的视角；其他总体对比只用于泛化压力判断。

## 16. 全部失败与当前限制

共输出456条失败/不完整诊断记录，涉及218个不同问题；同一问题可在不同检索模式、生成或在线检查中重复出现。完整列表：[RETAILRAG_SCALE_ALL_FAILURES.md](RETAILRAG_SCALE_ALL_FAILURES.md)，包含原始details的机器可读版本：[RETAILRAG_SCALE_ALL_FAILURES.json](RETAILRAG_SCALE_ALL_FAILURES.json)。不删除失败样本、不仅摘录成功案例。

主要限制：合成虚拟业务数据和同源标注；新FAQ口语改写覆盖仍不足；相似规则与多条件问题可能被FAQ截断；M=2限制多证据覆盖；相关性高不代表能回答不存在的具体数据；时间过滤采用执行日；同Qwen家族生成及裁判；新增业务规则仍需人工业务专家审阅。

## 17. 最终线上配置

```json
{
  "PARENT_CHUNK_SIZE": 1200,
  "CHILD_CHUNK_SIZE": 300,
  "CHUNK_OVERLAP": 50,
  "RETRIEVAL_K": 3,
  "CANDIDATE_M": 2,
  "FAQ_MATCH_THRESHOLD": 0.84,
  "EVIDENCE_RERANKER_THRESHOLD": 0.020635,
  "QUERY_REWRITE_ENABLED": true,
  "FAQ_RETRIEVAL_MODE": "dense",
  "FAQ_ALGORITHM_VERSION": "dense-bgem3-v1",
  "LLM_MODEL": "qwen3-max"
}
```

Dense/Sparse权重保持1.0:0.7；BGE-M3和BGE-reranker-large未更换；未新增Agent、LangGraph、MCP、权限系统或微服务。

## 18. 参考线与冻结判断

| 参考线 | 实测 | 达到 |
|---|---|---|
| Recall@3 > 95% | 0.936170 | False |
| MRR > 90% | 0.877447 | False |
| FAQ F1 > .85 | 0.844444 | False |
| FAQ FAR < 3% | 0.038889 | False |
| Gate拒答Precision >95% | 0.945946 | False |
| Gate拒答Recall >95% | 0.777778 | False |

这些只是用户指定的观察参考线，不用于调参或重写测试集。本轮保存可复现评测快照，但不将系统标为质量验收PASS。

## 19. 六项最终回答

1. **FAQ是否足以支持简历项目？** 200条不同业务决策已足够展示FAQ分流、缓存与边界评测；简历必须同时注明真实F1/FAR与改写失败，不只写数量或Exact满分。
2. **知识库规模是否足够？** 80篇覆盖8类业务、真实版本干扰与跨文档条件，对演示项目足够；不是生产成熟度证明。
3. **是否明显退化？** FAQ泛化和硬负例边界已退化，最终检索与拒答结果见上表；旧FAQ同题指标保持不代表新增条件稳定。
4. **最容易失败什么？** 多条件问题被FAQ单规则截获、相似制度的条件区分、超过两个必要文档的完整证据召回、带内部术语却索求不存在数据的问题。
5. **是否继续扩大数据？** 暂不建议继续堆数量。先人工复核标签与冲突规则，按真实失败类别评审下一轮改进，再收集独立问题；本轮没有实施这些架构或阈值修改。
6. **能否正式冻结？** 可冻结本轮数据、实验和演示快照，**不建议作为最终质量验收通过版本冻结**。线上可运行，但已复现不完整回答，需保留限制并另行决定修复范围。
