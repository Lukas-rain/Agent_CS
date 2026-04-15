"""节点2: 意图分类 - GLM-4-flash分类用户意图"""

import logging
from typing import Any

from config.prompts import INTENT_CLASSIFICATION_PROMPT
from src.graph.state import AgentState
from src.llm.zhipu_client import get_zhipu_client

logger = logging.getLogger(__name__)

VALID_INTENTS = {"faq", "complaint", "tech_support", "image_query", "chitchat"}
# 需要知识检索的意图
RETRIEVAL_INTENTS = {"faq", "complaint", "tech_support"}


async def intent_classification(state: AgentState) -> dict[str, Any]:
    """使用GLM-4-flash分类用户意图"""
    client = get_zhipu_client()

    user_input = state.get("preprocessed_text", "")
    image_description = state.get("image_description", "")

    prompt = INTENT_CLASSIFICATION_PROMPT.format(
        user_input=user_input,
        image_description=image_description or "无",
    )

    result = client.chat(
        prompt=prompt,
        system="你是一个意图分类器，只输出类别名称。",
        temperature=0.1,
        max_tokens=20,
    )

    # 解析意图
    intent = result.strip().lower()
    if intent not in VALID_INTENTS:
        # 尝试模糊匹配
        for valid in VALID_INTENTS:
            if valid in intent:
                intent = valid
                break
        else:
            intent = "chitchat"
            logger.warning(f"无法识别意图 '{result}'，默认为chitchat")

    # 有图片且有实质内容的查询标记为image_query
    if image_description and intent not in {"chitchat"}:
        intent = "image_query"

    needs_retrieval = intent in RETRIEVAL_INTENTS

    logger.info(f"意图分类: {intent} (需要检索: {needs_retrieval})")

    return {
        "intent": intent,
        "intent_confidence": 1.0,
        "needs_retrieval": needs_retrieval,
    }
