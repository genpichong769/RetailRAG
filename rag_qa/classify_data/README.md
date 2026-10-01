# 查询分类训练数据

`generate_training_data.py` 生成 1000 条平衡的二分类样本，标签为 `general` 与 `internal`。脚本使用固定随机种子 `20260923`，按主题分组拆分为 800/100/100，确保同一主题的模板化改写不会跨越训练、验证和测试集。

当前正式零售分类器位于 `rag_qa/models/retail-query-classifier/`，对应的冻结数据为 `retail_query_classifier_1000.jsonl`，并提供 800/100/100 的训练、验证和测试拆分。在线分类使用该模型，同时用规则保护明显的内部查询。模型权重通过 Git LFS 管理。
