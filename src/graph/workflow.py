"""LangGraph StateGraph 构建 - 图组装 (优化: reasoning合并回复生成)"""

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.graph.nodes.hallucination_check import (
    hallucination_check,
    route_after_hallucination_check,
)
from src.graph.nodes.input_preprocessing import input_preprocessing
from src.graph.nodes.intent_classification import intent_classification
from src.graph.nodes.knowledge_retrieval import knowledge_retrieval
from src.graph.nodes.memory_management import memory_management
from src.graph.nodes.reasoning import reasoning
from src.graph.state import AgentState

logger = logging.getLogger(__name__)


def route_after_intent(state: AgentState) -> str:
    """意图分类后路由"""
    if state.get("needs_retrieval", False):
        return "knowledge_retrieval"
    return "reasoning"


def route_after_reasoning(state: AgentState) -> str:
    """推理后路由: 闲聊→memory, 问答→hallucination_check"""
    if state.get("intent") == "chitchat":
        return "memory_management"
    return "hallucination_check"


def build_agent_graph() -> StateGraph:
    """构建Agent图 (reasoning已包含回复生成，省掉response_generation节点)"""
    graph = StateGraph(AgentState)

    # ── 注册节点 ──
    graph.add_node("input_preprocessing", input_preprocessing)
    graph.add_node("intent_classification", intent_classification)
    graph.add_node("knowledge_retrieval", knowledge_retrieval)
    graph.add_node("reasoning", reasoning)
    graph.add_node("hallucination_check", hallucination_check)
    graph.add_node("memory_management", memory_management)

    # ── 入口 ──
    graph.set_entry_point("input_preprocessing")

    # ── 固定边 ──
    graph.add_edge("input_preprocessing", "intent_classification")
    graph.add_edge("knowledge_retrieval", "reasoning")
    graph.add_edge("hallucination_check", "memory_management")
    graph.add_edge("memory_management", END)

    # ── 条件边: 意图→检索或推理 ──
    graph.add_conditional_edges(
        "intent_classification",
        route_after_intent,
        {
            "knowledge_retrieval": "knowledge_retrieval",
            "reasoning": "reasoning",
        },
    )

    # ── 条件边: 推理→闲聊直接结束 / 问答走幻觉检测 ──
    graph.add_conditional_edges(
        "reasoning",
        route_after_reasoning,
        {
            "memory_management": "memory_management",
            "hallucination_check": "hallucination_check",
        },
    )

    # ── 条件边: 幻觉检测→通过或重试 ──
    graph.add_conditional_edges(
        "hallucination_check",
        route_after_hallucination_check,
        {
            "reasoning": "reasoning",
            "memory_management": "memory_management",
        },
    )

    return graph


def compile_agent():
    graph = build_agent_graph()
    compiled = graph.compile()
    logger.info("Agent图编译完成")
    return compiled


def get_initial_state(
    user_input: str,
    images: list[str] | None = None,
    session_id: str = "default",
) -> dict[str, Any]:
    return {
        "user_input": user_input,
        "images": images or [],
        "session_id": session_id,
        "image_description": "",
        "preprocessed_text": "",
        "intent": "",
        "intent_confidence": 0.0,
        "retrieval_query": "",
        "retrieved_documents": [],
        "retrieved_context": "",
        "reasoning": "",
        "draft_response": "",
        "final_response": "",
        "hallucination_score": 0.0,
        "hallucination_reason": "",
        "retry_count": 0,
        "conversation_history": [],
        "conversation_summary": "",
        "message_count": 0,
        "needs_retrieval": False,
        "should_retry": False,
        "metadata": {},
    }
