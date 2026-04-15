"""对话记忆与总结管理"""

import logging
from typing import Any

from config.prompts import MEMORY_SUMMARY_PROMPT
from config.settings import MEMORY_SUMMARY_THRESHOLD, RECENT_MEMORY_WINDOW
from src.llm.zhipu_client import get_zhipu_client

logger = logging.getLogger(__name__)


def format_history(messages: list[dict[str, str]]) -> str:
    """将对话历史格式化为字符串"""
    if not messages:
        return ""
    lines = []
    for msg in messages:
        role = "用户" if msg["role"] == "user" else "客服"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def get_recent_messages(
    history: list[dict[str, str]],
    window: int | None = None,
) -> list[dict[str, str]]:
    """获取最近N条消息"""
    w = window or RECENT_MEMORY_WINDOW
    return history[-w:]


def should_summarize(message_count: int, threshold: int | None = None) -> bool:
    """判断是否需要触发总结"""
    th = threshold or MEMORY_SUMMARY_THRESHOLD
    return message_count > th and message_count % th == 0


async def summarize_history(
    history: list[dict[str, str]],
    existing_summary: str = "",
) -> str:
    """使用GLM-4-flash总结对话历史"""
    client = get_zhipu_client()

    history_text = format_history(history)
    prompt_input = existing_summary + "\n\n" + history_text if existing_summary else history_text

    prompt = MEMORY_SUMMARY_PROMPT.format(conversation_history=prompt_input)
    summary = client.chat(
        prompt=prompt,
        system="你是一个对话摘要助手，请简洁地总结对话关键信息。",
        temperature=0.3,
        max_tokens=512,
    )
    logger.info(f"对话历史已总结，原文{len(history)}条消息")
    return summary


def build_llm_messages(
    summary: str,
    recent: list[dict[str, str]],
    current_input: str,
    system_prompt: str = "你是一个专业的客服智能体。",
) -> list[dict[str, str]]:
    """构建发送给LLM的完整消息列表: 系统提示 + 摘要 + 最近对话 + 当前输入"""
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    # 如果有摘要，作为系统上下文注入
    if summary:
        messages.append({
            "role": "system",
            "content": f"对话历史摘要:\n{summary}",
        })

    # 最近对话
    messages.extend(recent)

    # 当前输入
    messages.append({"role": "user", "content": current_input})

    return messages
