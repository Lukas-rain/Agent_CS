"""智谱AI GLM-4V/GLM-4-flash 封装"""

import base64
import json
import logging
from typing import Any

from zhipuai import ZhipuAI

from config.settings import ZHIPU_API_KEY, ZHIPU_TEXT_MODEL, ZHIPU_VISION_MODEL

logger = logging.getLogger(__name__)


class ZhipuClient:
    """智谱AI客户端，封装文本和视觉模型调用"""

    def __init__(self):
        self.client = ZhipuAI(api_key=ZHIPU_API_KEY)
        self.text_model = ZHIPU_TEXT_MODEL
        self.vision_model = ZHIPU_VISION_MODEL

    # ── 文本对话 ──
    def chat(
        self,
        prompt: str,
        system: str = "你是一个专业的客服智能体。",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """调用GLM-4-flash文本模型"""
        try:
            response = self.client.chat.completions.create(
                model=self.text_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"文本模型调用失败: {e}")
            raise

    def chat_json(
        self,
        prompt: str,
        system: str = "你是一个JSON生成器，请严格按照要求输出JSON格式。",
        temperature: float = 0.3,
    ) -> dict[str, Any]:
        """调用文本模型并解析JSON输出"""
        raw = self.chat(prompt=prompt, system=system, temperature=temperature)
        # 尝试提取JSON块
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            logger.warning(f"JSON解析失败，原始输出: {raw}")
            return {"error": "json_parse_failed", "raw": raw}

    def chat_with_history(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """带历史消息的对话"""
        try:
            response = self.client.chat.completions.create(
                model=self.text_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"历史对话调用失败: {e}")
            raise

    # ── 视觉理解 ──
    def analyze_image(
        self,
        image_path: str,
        prompt: str = "请描述这张图片的内容。",
        temperature: float = 0.5,
    ) -> str:
        """调用GLM-4V视觉模型分析图片"""
        try:
            image_url = self._prepare_image_input(image_path)
            response = self.client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
                temperature=temperature,
                max_tokens=1024,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"视觉模型调用失败: {e}")
            return f"[图片分析失败: {e}]"

    # ── 辅助方法 ──
    @staticmethod
    def _prepare_image_input(image_path: str) -> str:
        """准备图片输入: 如果是URL直接用，否则转base64"""
        if image_path.startswith(("http://", "https://")):
            return image_path
        # 读取本地文件并转base64
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        # 推断MIME类型
        ext = image_path.lower().split(".")[-1]
        mime_map = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp"}
        mime = mime_map.get(ext, "jpeg")
        return f"data:image/{mime};base64,{image_data}"


# 全局单例
_zhipu_client: ZhipuClient | None = None


def get_zhipu_client() -> ZhipuClient:
    global _zhipu_client
    if _zhipu_client is None:
        _zhipu_client = ZhipuClient()
    return _zhipu_client
