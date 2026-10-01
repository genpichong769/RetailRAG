# 这个脚本讲义的代码架构图没有体现，需要进行补充
import hashlib
import os
from datetime import date, datetime

import yaml
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader  # NLTK
from langchain.text_splitter import MarkdownTextSplitter
# ========================================
# import sys
# # 获取当前文件所在目录的绝对路径
# current_dir = os.path.dirname(os.path.abspath(__file__))
# # print(f'current_dir--》{current_dir}')
# # 获取core文件所在的目录的绝对路径
# rag_qa_path = os.path.dirname(current_dir)
# # print(f'rag_qa_path--》{rag_qa_path}')
# sys.path.insert(0, rag_qa_path)
# # 获取根目录文件所在的绝对位置
# project_root = os.path.dirname(rag_qa_path)
# sys.path.insert(0, project_root)
# ========================================
from rag_qa.document_loaders import OCRPDFLoader, OCRDOCLoader, OCRPPTLoader, OCRIMGLoader
from rag_qa.text_splitters import ChineseRecursiveTextSplitter
from base import logger, Config

conf = Config()

VALID_DOCUMENT_STATUSES = {"active", "expired", "superseded"}


def _normalise_date(value, fallback):
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value or "").strip()
    return text or fallback


def _load_version_metadata(file_path, source, document_name):
    """Read optional YAML front matter and return deterministic version metadata."""
    metadata = {
        "version": "1.0",
        "effective_date": "2000-01-01",
        "expire_date": "9999-12-31",
        "status": "active",
        "version_group": document_name,
    }
    if os.path.splitext(file_path)[1].lower() == ".md":
        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                text = handle.read()
            if text.startswith("---"):
                parts = text.split("---", 2)
                if len(parts) == 3:
                    parsed = yaml.safe_load(parts[1]) or {}
                    if isinstance(parsed, dict):
                        metadata.update({key: parsed[key] for key in metadata if key in parsed})
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            logger.warning("读取版本元数据失败，使用默认值 (%s): %s", file_path, exc)

    metadata["version"] = str(metadata.get("version") or "1.0")
    metadata["effective_date"] = _normalise_date(
        metadata.get("effective_date"), "2000-01-01"
    )
    metadata["expire_date"] = _normalise_date(metadata.get("expire_date"), "9999-12-31")
    metadata["version_group"] = str(metadata.get("version_group") or document_name)
    status = str(metadata.get("status") or "active").lower()
    if status not in VALID_DOCUMENT_STATUSES:
        logger.warning("文档 %s 的 status=%s 无效，按 expired 处理", file_path, status)
        status = "expired"
    metadata["status"] = status
    metadata["source"] = source
    metadata["category"] = source
    metadata["document_name"] = document_name
    return metadata
# 定义支持的文件类型及其对应的加载器字典
document_loaders = {
    # 文本文件使用 TextLoader
    ".txt": TextLoader,
    # PDF 文件使用 OCRPDFLoader
    ".pdf": OCRPDFLoader,
    # Word 文件使用 OCRDOCLoader
    ".docx": OCRDOCLoader,
    # PPT 文件使用 OCRPPTLoader
    ".ppt": OCRPPTLoader,
    # PPTX 文件使用 OCRPPTLoader
    ".pptx": OCRPPTLoader,
    # JPG 文件使用 OCRIMGLoader
    ".jpg": OCRIMGLoader,
    # PNG 文件使用 OCRIMGLoader
    ".png": OCRIMGLoader,
    # Markdown 文件使用 UnstructuredMarkdownLoader
    ".md": UnstructuredMarkdownLoader
}


# 定义函数，从指定文件夹加载多种类型文件并添加元数据
def load_documents_from_directory(directory_path):
    # 初始化空列表，用于存储加载的文档
    documents = []
    # 获取支持的文件扩展名集合
    supported_extensions = document_loaders.keys()
    # print(f'supported_extensions--》{supported_extensions}')
    # 从目录名提取零售知识类别（如 "membership_data" -> "membership"）
    # print(f'1---》{os.path.basename(directory_path)}')
    source = os.path.basename(directory_path).replace("_data", "")
    # print(f'source-->{source}')
    # 遍历指定目录及其子目录
    for root, _, files in os.walk(directory_path):
        # print(f'root---》{root}')
        # print(f'files---》{files}')
        # 遍历当前目录下的所有文件
        for file in files:
            # 构造文件的完整路径
            file_path = os.path.join(root, file)
            # print(f'file_path--》{file_path}')
            # print(os.path.splitext(file_path))
            # 获取文件扩展名并转换为小写
            file_extension = os.path.splitext(file_path)[1].lower()
            # print(f'file_extension--》{file_extension}')
            # 检查文件类型是否在支持的扩展名列表中
            if file_extension in supported_extensions:
                # 使用 try-except 捕获加载过程中的异常
                try:
                    # 根据文件扩展名获取对应的加载器类
                    loader_class = document_loaders[file_extension]
                    # 实例化加载器对象，传入文件路径
                    if file_extension == ".txt":
                        loader = loader_class(file_path, encoding="utf-8")
                    else:
                        loader = loader_class(file_path)
                    # 调用加载器加载文档内容，返回文档列表
                    loaded_docs = loader.load()
                    # print(f'loaded_docs--》{loaded_docs}')
                    # print(f'loaded_docs--》{len(loaded_docs)}')
                    for doc in loaded_docs:
                        document_name = os.path.splitext(file)[0]
                        doc.metadata.update(
                            _load_version_metadata(file_path, source, document_name)
                        )
                        # 为文档添加文件路径元数据
                        doc.metadata["file_path"] = file_path
                        # 为文档添加当前时间戳元数据
                        doc.metadata["timestamp"] = datetime.now().isoformat()
                    # print(f'loaded_docs111--》{loaded_docs}')
                    documents.extend(loaded_docs)
                    # 记录成功加载文件的日志
                    logger.info(f"成功加载文件: {file_path}")
                except Exception as e:
                    logger.error(f"加载文件 {file_path} 失败: {str(e)}")
            # 如果文件类型不在支持列表中
            else:
                # 记录警告日志，提示不支持的文件类型
                logger.warning(f"不支持的文件类型: {file_path}")
    # 返回加载的所有文档列表
    return documents


# 定义函数，处理文档并进行分层切分，返回子块结果
def process_documents(directory_path, parent_chunk_size=conf.PARENT_CHUNK_SIZE,
                      child_chunk_size=conf.CHILD_CHUNK_SIZE,
                      chunk_overlap=conf.CHUNK_OVERLAP):
    # 从指定目录加载所有文档
    documents = load_documents_from_directory(directory_path)
    # 记录加载的文档总数日志
    logger.info(f"加载的文档数量: {len(documents)}")

    # 初始化父块和子块分词器（通用）
    parent_splitter = ChineseRecursiveTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap)
    child_splitter = ChineseRecursiveTextSplitter(chunk_size=child_chunk_size, chunk_overlap=chunk_overlap)
    # 初始化 Markdown 专用分词器
    markdown_parent_splitter = MarkdownTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap)
    markdown_child_splitter = MarkdownTextSplitter(chunk_size=child_chunk_size, chunk_overlap=chunk_overlap)

    # 初始化空列表，用于存储所有子块
    child_chunks = []
    # 遍历每个原始文档，带上索引 i
    for i, doc in enumerate(documents):
        # print(f'doc--》{doc}')
        # print(doc.metadata.get("file_path", ''))
        # # 获取文件的扩展名
        # print(os.path.splitext(doc.metadata.get("file_path", '')))
        file_extension = os.path.splitext(doc.metadata.get("file_path", ''))[1].lower()
        # print(f'file_extension--》{file_extension}')
        # 选择分词器
        is_markdown = (file_extension == '.md')
        # print(f'is_markdown--》{is_markdown}')
        parent_splitter_to_use = markdown_parent_splitter if is_markdown else parent_splitter
        # print(f'parent_splitter_to_use-->{parent_splitter_to_use}')
        child_splitter_to_use = markdown_child_splitter if is_markdown else child_splitter
        logger.info(
            f"处理文档: {doc.metadata['file_path']}, 使用切分器: {'Markdown' if is_markdown else 'ChineseRecursive'}")
        # 使用父块切分器将文档切分为父块
        parent_docs = parent_splitter_to_use.split_documents([doc])
        # print(f'parent_docs--》{parent_docs}')
        # print(f'parent_docs--》{len(parent_docs)}')
        # 遍历每个父块，带上索引 j
        for j, parent_doc in enumerate(parent_docs):
            # 稳定 ID 不依赖目录枚举顺序，新增文档不会改变旧块 ID。
            document_identity = "|".join(
                [
                    str(doc.metadata.get("source", "unknown")),
                    str(doc.metadata.get("document_name", "unknown")),
                    str(doc.metadata.get("version", "1.0")),
                ]
            )
            document_hash = hashlib.sha1(document_identity.encode("utf-8")).hexdigest()[:16]
            parent_id = f"doc_{document_hash}_parent_{j}"
            # # 将父块 ID 添加到元数据
            # parent_doc.metadata["parent_id"] = parent_id
            # # 将父块内容存储到元数据
            # parent_doc.metadata["parent_content"] = parent_doc.page_content

            # 使用子块分词器将父块切分为子块
            sub_chunks = child_splitter_to_use.split_documents([parent_doc])
            # print(f'sub_chunks--》{sub_chunks}')
            # print(f'sub_chunks--》{len(sub_chunks)}')
            # 遍历每个子块，为子块主要添加对应的父块文档
            for k, sub_chunk in enumerate(sub_chunks):
                # print(f'原始的sub_chunk--》{sub_chunk}')
                # 为子块添加父块的ID
                sub_chunk.metadata["parent_id"] = parent_id
                # 为子块添加对应的父块文档（元数据）
                sub_chunk.metadata["parent_content"] = parent_doc.page_content
                # 为子块生成一个唯一的ID,格式为 "parent_id_child_k"
                sub_chunk.metadata["id"] = f"{parent_id}_child_{k}"
                # print(f'修改后的sub_chunk--》{sub_chunk}')
                # 将子块添加到子块列表中
                child_chunks.append(sub_chunk)

    # 记录子块总数日志
    logger.info(f"子块数量: {len(child_chunks)}")
    # 返回所有子块列表
    return child_chunks


if __name__ == '__main__':
    directory_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "membership_data")
    # documents = load_documents_from_directory(directory_path)
    # print(documents)
    child_chunks = process_documents(directory_path)
    print(f'child_chunks--》{child_chunks[0]}')
