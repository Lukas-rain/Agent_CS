"""节点1: 输入预处理 - 解析图文输入，GLM-4V生成图片描述"""

import logging
from typing import Any

from src.graph.state import AgentState
from src.llm.zhipu_client import get_zhipu_client
from config.prompts import IMAGE_DESCRIPTION_PROMPT

logger = logging.getLogger(__name__)


async def input_preprocessing(state: AgentState) -> dict[str, Any]:
    """预处理用户输入: 文本清洗 + 图片描述生成"""
    user_input = state.get("user_input", "").strip()
    images = state.get("images", [])

    # 文本预处理
    preprocessed_text = user_input

    # 图片描述生成
    image_description = ""
    if images:
        client = get_zhipu_client()
        descriptions = []
        for i, img in enumerate(images):
            try:
                desc = client.analyze_image(img, prompt=IMAGE_DESCRIPTION_PROMPT)
                descriptions.append(f"[图片{i + 1}]: {desc}")
            except Exception as e:
                logger.warning(f"图片{i + 1}分析失败: {e}")
                descriptions.append(f"[图片{i + 1}]: (分析失败)")
        image_description = "\n".join(descriptions)

    # 拼接完整输入文本(含图片描述)
    if image_description:
        preprocessed_text = f"{preprocessed_text}\n\n图片内容:\n{image_description}"

    logger.info(f"预处理完成 - 文本长度:{len(preprocessed_text)}, 图片数:{len(images)}")

    return {
        "preprocessed_text": preprocessed_text,
        "image_description": image_description,
    }
