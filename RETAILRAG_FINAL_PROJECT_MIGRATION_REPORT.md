# RetailRAG 最终项目迁移报告

## SOURCE / TARGET

- SOURCE: E:\EduRAG智慧回答项目\RetailRAG
- TARGET: E:\RetailRAG：连锁零售企业知识问答库
- 完成状态：TARGET VERIFIED
- 源目录文件未移动或删除；源目录仍保留。

## 审计摘要与 COPY_STATUS

源目录共 950 个文件、8,507,387,008 字节。逐文件清单见 RETAILRAG_CLEANUP_MANIFEST.json。复制 220 个文件，另复制 1 个仅本机使用且被 Git 忽略的 .env；归档 5 个有保留价值的历史文件；排除 724 个文件、8,071,203,291 字节。初始目标目录不存在，因此没有覆盖用户原有目标文件。

| 类别 | 源文件数 | 处理 |
|---|---:|---|
| 核心代码与测试脚本 | 84 | 保留当前运行链与初始化、评测、Smoke 工具 |
| FAQ 数据 | 1 | 200 条冻结 CSV |
| 知识文档 | 80 | 八个业务域，包含版本冲突样本 |
| 分类数据与模型 | 13 | 1000 条分类数据及 8 个正式模型文件 |
| 冻结评测输入 | 8 | FAQ、检索、RAG、生成选择及审计配置 |
| qwen3.7-plus 最终结果 | 19 | 全部原样复制 |
| 历史实验 | 115 | 精选 5 个归档，其余排除 |
| 缓存、日志与运行数据 | 448 | 不进入 Git 项目 |

目标 Git 交付内容为 231 个文件、35 个目录；另有被忽略的本机 .env、Docker 运行数据和 Smoke 日志。分类与逐文件动作均可在 JSON 清单中核查。

## CORE_CODE / FAQ_DATA / KNOWLEDGE_DATA / CLASSIFIER

保留 app.py、new_main.py、base/、mysql_qa/、rag_qa/ 当前运行依赖，以及 scripts/ 中初始化、入库、正式评测和 Smoke 脚本。未改变 FAQ、检索、分块、重排、证据门、提示词或 Ground Truth 算法。迁移修正了 scripts/init_mysql.py 遗留的 52 条硬编码校验，改为与冻结 CSV 一致的 200 条。

FAQ CSV 实测 200 条。知识文档实测 80 篇，覆盖 membership、promotion、after_sales、store_operation、product、equipment、policy、notice。版本冲突所需的旧版本文档仍在 80 篇之内。分类器数据实测 1000 行；正式模型 rag_qa/models/retail-query-classifier/ 的 8 个文件已保留，其中 409,100,240 字节权重配置 Git LFS。

BGE-M3 与 BGE Reranker 是可重新获取的上游模型，不放在 Git 项目中。本机模型缓存位于 E:\RetailRAG_models；它由源模型文件建立硬链接，删除源目录不会删除这些缓存文件。目标 .env 指向该独立缓存。新机器可以用模型 ID 下载，或设置 BGE_M3_PATH、BGE_RERANKER_PATH。

## EVALUATION_DATA / FINAL_REPORTS / HASH_VALIDATION

冻结计数：FAQ Evaluation 480、Retrieval Evaluation 235、RAG Evaluation 350、Multi-turn 30、Generation Evaluation 80。RETAILRAG_SCALE_DATA_AUDIT.json 所列 84 个文件 SHA-256 在目标目录全部一致：FINAL_BENCHMARK_HASH=PASS。FAQ、文档、分类、评测输入及正式结果共 121 个迁移资产与源文件逐一 SHA-256 相同。

用户指定的 12 个 qwen3.7-plus 正式交付物均存在；项目另外保留了 7 个同批正式附属结果。最终模型 qwen3.7-plus，冻结状态 GOOD_WITH_LIMITATIONS。正式指标未重新生成或改写；根目录 README 引用原评测文件。主要指标为 FAQ F1 0.8636、Retrieval Recall@3 0.9011、Full RAG Strict Accuracy 0.7914，执行 350/350、错误 0。已知限制为多文档 All Recall@3 0.6774、多文档 Strict Accuracy 0.6333。

## DOCKER / FRONTEND / PATH_MIGRATION

新增根目录 docker-compose.yml；在目标目录执行 docker compose up -d 成功。MySQL、Redis、Milvus、MinIO、etcd 五个容器目前都在运行，全部 bind mount 指向目标目录，旧源目录挂载数为 0。目标初始化脚本重建 MySQL 200 条 FAQ、Redis 缓存状态、Milvus retailrag 数据库及 80 篇文档；Milvus 可见 203 个唯一子块，压缩完成。原 Docker 持久化文件未复制，也未删除。

当前前端 static/index.html 返回 HTTP 200；保留 static/src/App.jsx 作为前端源码。85 个目标 Python 文件通过 AST 检查，核心依赖导入与配置读取通过。正式 Python/配置/前端代码中，旧 EduRAG 绝对路径、Itcast_qa_system 和旧 Phase 5 兄弟备份引用计数为 0。历史报告、清理清单和分类器训练元数据中的旧路径仅为历史文字或审计字段。配置的项目路径由文件位置推导，本机模型缓存通过 .env 指定。

## SECRET_AUDIT / Git

源目录发现根目录 .env 和 docker/mysql/.env 两个本机凭据风险文件。只将根目录 .env 复制为目标本机文件并由 .gitignore 排除；docker/mysql/.env 未复制。目标 .env.example 仅含占位值。对 230 个候选 Git 文件扫描实际 MySQL、MinIO 密码及当前 DashScope Key，命中 0；未在报告或清单中输出 Secret 值。

目标已初始化独立 Git 仓库，231 个交付文件均已暂存；.env、缓存、日志、Docker 数据卷和上游模型缓存均被忽略。正式分类器权重使用 Git LFS。尝试创建初始提交时发现本机未配置 Git author name/email，因此未伪造用户身份，提交未创建。上传到 GitHub 前需配置本仓库 Git 身份，再提交与推送。

## REMOVED_FROM_TARGET / NOT_COPIED / ARCHIVED_HISTORY

目标曾复制的旧 README 说明已重写为最终版，rag_qa/data、rag_qa/models 和分类数据说明也已更新。没有从源目录删除文件。未复制的 724 个文件主要为 .git 历史对象、Docker 运行数据、BGE/语义切分模型缓存、Python 缓存、日志、教程 Notebook、旧实验 JSON、阶段性 checkpoint、旧入口与教育遗留。逐项原因在清理清单中。

docs/history/ 仅归档 Phase 5、Expanded Scale、Phase 6 关键报告及 qwen3.6-plus 实验摘要/一次部分结果，共 5 个文件。

## SMOKE_TEST / FASTAPI / WEBSOCKET

迁移后使用目标目录及其自有 Docker 数据卷完成最小在线验证：FAQ 原问题、FAQ paraphrase、单文档 RAG、多文档 RAG、无答案拒答、多轮 Query Rewrite、FastAPI 首页/健康/来源接口和 WebSocket start/token/end 流程。单文档结果返回业务规则，无答案问题返回知识不足拒答。一次独立的 qwen3.7-plus 最小生成请求返回预期值；未重跑 480/235/80/350 全量评测。

FastAPI 当前由目标目录启动，在 http://127.0.0.1:8003/ 提供服务，启动 PID 为 19488。原 8003 进程先经 PID 与工作目录识别，再停止；没有盲目杀进程。

## FINAL_STATUS

TARGET VERIFIED。新目录的代码、冻结数据、模型配置、Docker 卷与当前服务均不再依赖源 RetailRAG 路径。源目录从当前运行依赖角度可以在用户确认后删除；本次按要求停在验证状态，没有自动删除。目标目录可作为 GitHub 项目根目录，正式分类器需通过 Git LFS 上传。
