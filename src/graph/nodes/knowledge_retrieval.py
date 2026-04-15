"""节点3: 知识检索 - Chroma检索 + 相关性过滤"""

import logging
from typing import Any

from config.prompts import QUERY_GENERATION_PROMPT
from src.graph.state import AgentState
from src.llm.zhipu_client import get_zhipu_client
from src.rag.retriever import filter_by_relevance, format_retrieved_context, retrieve_documents

logger = logging.getLogger(__name__)


async def knowledge_retrieval(state: AgentState) -> dict[str, Any]:
    """从Chroma向量库检索相关知识"""
    client = get_zhipu_client()

    user_input = state.get("user_input", "")
    intent = state.get("intent", "")
    image_description = state.get("image_description", "")

    # 生成优化的检索query
    query_prompt = QUERY_GENERATION_PROMPT.format(
        user_input=user_input,
        intent=intent,
        image_description=image_description or "无",
    )
    retrieval_query = client.chat(
        prompt=query_prompt,
        system="你是一个查询优化器，请生成简洁准确的检索查询。",
        temperature=0.3,
        max_tokens=100,
    )

    logger.info(f"检索query: {retrieval_query}")

    # 检索文档
    raw_docs = retrieve_documents(retrieval_query)

    # 过滤低相关性结果
    filtered_docs = filter_by_relevance(raw_docs, min_similarity=0.2)

    # 格式化上下文
    retrieved_context = format_retrieved_context(filtered_docs)

    logger.info(f"检索到 {len(filtered_docs)} 条相关文档 (原始: {len(raw_docs)})")

    return {
        "retrieval_query": retrieval_query,
        "retrieved_documents": filtered_docs,
        "retrieved_context": retrieved_context,
    }
