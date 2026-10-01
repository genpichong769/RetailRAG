# 模型准备

正式自训练分类器 `retail-query-classifier/` 已随项目保留，其 `model.safetensors` 使用 Git LFS。克隆时运行 `git lfs pull`。

BGE-M3 与 BGE Reranker 是上游模型，不随项目复制。默认配置在本地目录不存在时使用 `BAAI/bge-m3` 和 `BAAI/bge-reranker-large` 模型 ID 下载到标准 Hugging Face 缓存。也可在 `.env` 设置 `BGE_M3_PATH` 和 `BGE_RERANKER_PATH` 指向机器上的本地模型目录。`nlp_bert_document-segmentation_chinese-base` 仅供可选语义切分实验使用。
