"""节点5: 回复生成 - 格式化多模态回复(含引用来源)"""

import logging
from typing import Any

from config.prompts import RESPONSE_GENERATION_PROMPT
from src.graph.state import AgentState
from src.llm.zhipu_client import get_zhipu_client

logger = logging.getLogger(__name__)


async def response_generation(state: AgentState) -> dict[str, Any]:
    """基于推理结果生成最终回复"""
    client = get_zhipu_client()

    reasoning_text = state.get("reasoning", "")
    retrieved_context = state.get("retrieved_context", "（无检索结果）")
    retrieved_documents = state.get("retrieved_documents", [])

    # 构建带来源标注的上下文
    sources_text = ""
    if retrieved_documents:
        sources = set()
        for doc in retrieved_documents:
            source = doc.get("metadata", {}).get("source", "")
            if source:
                sources.add(source)
        if sources:
            sources_text = "参考来源: " + ", ".join(sources)

    context_with_sources = retrieved_context
    if sources_text:
        context_with_sources += f"\n\n{sources_text}"

    prompt = RESPONSE_GENERATION_PROMPT.format(
        reasoning=reasoning_text,
        retrieved_context_with_sources=context_with_sources,
    )

    response = client.chat(
        prompt=prompt,
        system="你是一个专业、友好的客服智能体。仅基于提供的信息回答，不要编造事实。",
        temperature=0.6,
        max_tokens=1024,
    )

    # 如果有检索来源，追加引用
    if sources_text and "来源" not in response:
        response += f"\n\n{sources_text}"

    logger.info(f"回复生成完成，长度: {len(response)}")

    return {
        "draft_response": response,
        "final_response": response,
    }
