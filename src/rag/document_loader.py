"""文档加载与分块"""

import logging
import os
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import CHUNK_OVERLAP, CHUNK_SIZE, KNOWLEDGE_BASE_DIR

logger = logging.getLogger(__name__)


def load_markdown_file(file_path: str) -> list[dict[str, Any]]:
    """加载单个Markdown文件，返回文档块列表"""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    if not content.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", "。", "！", "？", "；", " "],
        length_function=len,
    )

    chunks = splitter.split_text(content)
    base_name = os.path.basename(file_path)

    return [
        {
            "text": chunk,
            "metadata": {
                "source": base_name,
                "doc_type": "text",
                "chunk_index": i,
            },
        }
        for i, chunk in enumerate(chunks)
    ]


def load_all_documents(directory: str | None = None) -> list[dict[str, Any]]:
    """加载目录下所有Markdown文档"""
    doc_dir = directory or KNOWLEDGE_BASE_DIR
    all_chunks: list[dict[str, Any]] = []

    if not os.path.exists(doc_dir):
        logger.warning(f"知识库目录不存在: {doc_dir}")
        return all_chunks

    for filename in os.listdir(doc_dir):
        if filename.endswith((".md", ".txt")):
            file_path = os.path.join(doc_dir, filename)
            chunks = load_markdown_file(file_path)
            all_chunks.extend(chunks)
            logger.info(f"加载 {filename}: {len(chunks)} 个分块")

    logger.info(f"共加载 {len(all_chunks)} 个文档分块")
    return all_chunks


def add_image_description(
    image_url: str,
    description: str,
    source: str = "image",
) -> dict[str, Any]:
    """将图片描述构建为可入库的文档"""
    return {
        "text": description,
        "metadata": {
            "source": source,
            "doc_type": "image_description",
            "image_url": image_url,
        },
    }
