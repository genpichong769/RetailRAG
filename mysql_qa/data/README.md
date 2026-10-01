# RetailRAG FAQ 数据

`retail_faq.csv` 是冻结的 200 条演示 FAQ，字段固定为：`知识类别`、`问题`、`答案`。文件哈希记录在根目录 `RETAILRAG_SCALE_DATA_AUDIT.json`。

执行 `python -m mysql_qa.init_faq` 会显式创建表、替换 FAQ 数据并清理当前 RetailRAG 命名空间下的 Redis 缓存。项目不会在服务启动时自动初始化 MySQL。
