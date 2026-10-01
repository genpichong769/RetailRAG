# Nova Retail 冻结知识库

本目录保留 80 篇虚构的零售业务 Markdown 文档，覆盖 membership、promotion、after_sales、store_operation、product、equipment、policy、notice 八个领域。其中包含版本冲突与历史版本样本，属于正式评测依赖；不能仅按最新版本删除。

文档与 `RETAILRAG_SCALE_DATA_AUDIT.json` 中的 SHA-256 基线一致。请勿对冻结文档运行旧的生成脚本。新环境先启动 Milvus，再执行 `python scripts/init_milvus.py` 和 `python scripts/ingest_knowledge_base.py --replace`。

这些内容是演示数据，不代表真实企业制度或承诺。
