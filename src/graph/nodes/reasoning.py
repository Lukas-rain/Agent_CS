"""节点4: 推理+回复生成 (合并为单次LLM调用，节省延迟)"""

import logging
from typing import Any

from config.prompts import REASONING_PROMPT
from src.graph.state import AgentState
from src.memory.conversation_memory import format_history, get_recent_messages
from src.llm.zhipu_client import get_zhipu_client

logger = logging.getLogger(__name__)

CHITCHAT_PROMPT = """你是一个友好的客服助手。请简短、自然地回复。

用户说: {user_input}
请简短回复(1-2句话):"""

QA_PROMPT = """你是一个专业、友好的客服智能体。请根据以下信息直接回答用户问题。

## 用户问题
{user_input}

## 图片描述
{image_description}

## 知识库内容
{retrieved_context}

## 要求
1. 仅使用上面提供的知识库内容回答，不要编造事实
2. 回答简洁专业，突出重点
3. 如果知识库没有相关信息，请诚实告知并建议联系人工客服
4. 如有引用，末尾标注来源
5. 不要输出推理过程，直接给出最终回复

请直接回复:"""


async def reasoning(state: AgentState) -> dict[str, Any]:
    """推理+回复合并: 减少LLM调用次数"""
    client = get_zhipu_client()

    user_input = state.get("user_input", "")
    image_description = state.get("image_description", "")
    intent = state.get("intent", "")
    retrieved_context = state.get("retrieved_context", "（无检索结果）")
    conversation_summary = state.get("conversation_summary", "")
    history = state.get("conversation_history", [])
    retrieved_documents = state.get("retrieved_documents", [])

    # ── 闲聊: 1次快速调用，直接结束 ──
    if intent == "chitchat":
        prompt = CHITCHAT_PROMPT.format(user_input=user_input)
        reply = client.chat(prompt=prompt, temperature=0.7, max_tokens=150)
        logger.info("闲聊快速回复")
        return {
            "reasoning": "[闲聊]",
            "draft_response": reply,
            "final_response": reply,
            "hallucination_score": 0.0,
            "should_retry": False,
        }

    # ── 正式问答: 1次调用直接生成回复（不再分reasoning+generation两步）──
    prompt = QA_PROMPT.format(
        user_input=user_input,
        image_description=image_description or "无",
        retrieved_context=retrieved_context,
    )

    reply = client.chat(
        prompt=prompt,
        system="你是专业客服，仅基于提供的信息回答，简洁直接，不输出推理过程。",
        temperature=0.4,
        max_tokens=800,
    )

    # 追加来源标注
    if retrieved_documents:
        sources = set(
            doc.get("metadata", {}).get("source", "")
            for doc in retrieved_documents
            if doc.get("metadata", {}).get("source")
        )
        if sources and "来源" not in reply:
            reply += f"\n\n参考来源: {', '.join(sources)}"

    logger.info(f"回复完成，长度: {len(reply)}")

    return {
        "reasoning": "[直接生成回复]",
        "draft_response": reply,
        "final_response": reply,
    }
