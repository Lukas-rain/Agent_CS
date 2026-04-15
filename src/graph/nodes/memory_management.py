"""节点7: 记忆管理 - 追加历史，触发总结"""

import logging
from typing import Any

from config.settings import MAX_RETRY_COUNT
from src.graph.state import AgentState
from src.memory.conversation_memory import should_summarize, summarize_history

logger = logging.getLogger(__name__)


async def memory_management(state: AgentState) -> dict[str, Any]:
    """管理对话记忆: 追加当前轮次，必要时触发总结"""
    user_input = state.get("user_input", "")
    final_response = state.get("final_response", "")
    retry_count = state.get("retry_count", 0)

    # 重试耗尽时附加免责声明
    if retry_count >= MAX_RETRY_COUNT:
        final_response += (
            "\n\n---\n⚠️ **温馨提示**: 以上回复未经知识库完全验证，"
            "建议您联系人工客服确认详细信息。"
        )

    # 追加当前对话到历史
    new_messages = [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": final_response},
    ]

    # 计算消息数
    current_count = state.get("message_count", 0) + 2
    existing_summary = state.get("conversation_summary", "")
    history = state.get("conversation_history", [])

    # 触发总结
    summary = existing_summary
    if should_summarize(current_count):
        all_history = history + new_messages
        try:
            summary = await summarize_history(all_history, existing_summary)
        except Exception as e:
            logger.warning(f"总结失败: {e}")
            summary = existing_summary

    logger.info(f"记忆更新完成 - 消息数:{current_count}, 总结长度:{len(summary)}")

    return {
        "conversation_history": new_messages,
        "final_response": final_response,
        "conversation_summary": summary,
        "message_count": current_count,
    }
