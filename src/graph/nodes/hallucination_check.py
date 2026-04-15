"""节点6: 幻觉检测 - 交叉验证回复与源文档"""

import json
import logging
from typing import Any

from config.prompts import HALLUCINATION_CHECK_PROMPT
from config.settings import HALLUCINATION_THRESHOLD, MAX_RETRY_COUNT
from src.graph.state import AgentState
from src.llm.zhipu_client import get_zhipu_client

logger = logging.getLogger(__name__)


async def hallucination_check(state: AgentState) -> dict[str, Any]:
    """检测回复中的幻觉，评分0-1"""
    client = get_zhipu_client()

    response = state.get("final_response", "")
    retrieved_documents = state.get("retrieved_documents", [])
    user_input = state.get("user_input", "")

    # 如果没有检索到文档，跳过幻觉检测（纯闲聊或图片查询场景）
    if not retrieved_documents:
        logger.info("无检索文档，跳过幻觉检测")
        return {
            "hallucination_score": 0.0,
            "hallucination_reason": "无检索文档，跳过检测",
            "should_retry": False,
        }

    # 格式化源文档
    source_docs = "\n\n".join(
        f"[文档{i + 1}]: {doc['text']}" for i, doc in enumerate(retrieved_documents)
    )

    prompt = HALLUCINATION_CHECK_PROMPT.format(
        response=response,
        source_documents=source_docs,
        user_input=user_input,
    )

    result = client.chat_json(
        prompt=prompt,
        system="你是一个回复质量审核员，请严格按照JSON格式输出评估结果。",
        temperature=0.1,
    )

    # 解析结果
    score = result.get("score", 0.5)
    reason = result.get("reason", "未知原因")

    # 安全处理score类型
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = 0.5

    score = max(0.0, min(1.0, score))

    retry_count = state.get("retry_count", 0)
    should_retry = score > HALLUCINATION_THRESHOLD and retry_count < MAX_RETRY_COUNT

    logger.info(f"幻觉评分: {score:.2f} (阈值:{HALLUCINATION_THRESHOLD}, 重试:{should_retry}, 次数:{retry_count})")
    if should_retry:
        logger.info(f"幻觉检测原因: {reason}")

    return {
        "hallucination_score": score,
        "hallucination_reason": reason,
        "should_retry": should_retry,
        "retry_count": retry_count + (1 if should_retry else 0),
    }


def route_after_hallucination_check(state: AgentState) -> str:
    """条件路由: 幻觉检测通过→memory_management，未通过→reasoning重试"""
    if state.get("should_retry", False):
        retry_count = state.get("retry_count", 0)
        if retry_count >= MAX_RETRY_COUNT:
            # 重试耗尽，附加免责声明
            response = state.get("final_response", "")
            disclaimer = (
                "\n\n---\n⚠️ **温馨提示**: 以上回复未经知识库完全验证，"
                "建议您联系人工客服确认详细信息。"
            )
            # 直接返回修改后的字段在routing函数中不行，需要通过node处理
            # 这里用metadata传递
            return "memory_management"
        return "reasoning"
    return "memory_management"
