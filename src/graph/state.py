"""AgentState TypedDict - LangGraph状态定义"""

import operator
from typing import Annotated, Any, TypedDict


class AgentState(TypedDict):
    # ── 输入 ──
    user_input: str                          # 用户文本输入
    images: list[str]                        # 用户上传的图片路径/base64列表
    session_id: str                          # 会话ID

    # ── 预处理 ──
    image_description: str                   # GLM-4V生成的图片描述
    preprocessed_text: str                   # 预处理后的文本

    # ── 意图分类 ──
    intent: str                              # faq/complaint/tech_support/image_query/chitchat
    intent_confidence: float                 # 意图分类置信度

    # ── 知识检索 ──
    retrieval_query: str                     # 检索用的查询语句
    retrieved_documents: list[dict[str, Any]]  # 检索到的文档列表
    retrieved_context: str                   # 格式化后的检索上下文

    # ── 推理 ──
    reasoning: str                           # 思维链推理过程

    # ── 回复 ──
    draft_response: str                      # 草稿回复
    final_response: str                      # 最终回复

    # ── 幻觉检测 ──
    hallucination_score: float               # 幻觉评分 0-1
    hallucination_reason: str                # 评分理由
    retry_count: int                         # 当前重试次数

    # ── 记忆 ──
    conversation_history: Annotated[list[dict[str, str]], operator.add]  # 累加对话记录
    conversation_summary: str                # 对话总结
    message_count: int                       # 当前消息计数

    # ── 控制 ──
    needs_retrieval: bool                    # 是否需要知识检索
    should_retry: bool                       # 是否需要重试
    metadata: dict[str, Any]                 # 其他元数据
