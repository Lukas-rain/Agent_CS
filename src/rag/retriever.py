"""RAG检索逻辑"""

import logging
from typing import Any

from config.settings import RETRIEVAL_TOP_K
from src.rag.chroma_store import get_chroma_store

logger = logging.getLogger(__name__)


def retrieve_documents(
    query: str,
    top_k: int | None = None,
    doc_type_filter: str | None = None,
) -> list[dict[str, Any]]:
    """检索与query相关的文档

    Returns:
        list of {"text": str, "metadata": dict, "distance": float}
    """
    store = get_chroma_store()
    k = top_k or RETRIEVAL_TOP_K

    where_filter = None
    if doc_type_filter:
        where_filter = {"doc_type": doc_type_filter}

    results = store.query(
        query_texts=[query],
        n_results=k,
        where=where_filter,
    )

    documents = []
    if results["documents"] and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            distance = results["distances"][0][i] if results["distances"] else 0.0
            documents.append({
                "text": doc,
                "metadata": metadata,
                "distance": distance,
            })

    return documents


def format_retrieved_context(documents: list[dict[str, Any]]) -> str:
    """将检索结果格式化为上下文字符串"""
    if not documents:
        return "（未找到相关知识库内容）"

    parts = []
    for i, doc in enumerate(documents, 1):
        source = doc["metadata"].get("source", "未知来源")
        doc_type = doc["metadata"].get("doc_type", "text")
        text = doc["text"]
        distance = doc.get("distance", 0)

        if doc_type == "image_description":
            image_url = doc["metadata"].get("image_url", "")
            parts.append(f"[来源{i}: 图片描述 (来源:{source}, 相似度:{1 - distance:.2f})]\n{text}\n图片链接: {image_url}")
        else:
            parts.append(f"[来源{i}: {source}, 相似度:{1 - distance:.2f}]\n{text}")

    return "\n\n".join(parts)


def filter_by_relevance(
    documents: list[dict[str, Any]],
    min_similarity: float = 0.3,
) -> list[dict[str, Any]]:
    """过滤低相关性的检索结果 (cosine距离越小越相似)"""
    filtered = [doc for doc in documents if doc.get("distance", 1.0) < (1 - min_similarity)]
    if not filtered and documents:
        # 至少返回最相关的一条
        filtered = [min(documents, key=lambda d: d.get("distance", 1.0))]
    return filtered
