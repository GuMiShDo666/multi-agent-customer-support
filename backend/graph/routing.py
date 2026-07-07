"""
条件路由函数 — 独立模块，不依赖 langgraph，便于单元测试。
"""

from ..config import MAX_RETRIES, QA_THRESHOLD
from .state import SupportState


def route_to_agent(state: SupportState) -> str:
    """
    智能路由：基于意图分类结果 + 生效优先级选择Agent。

    相比原CrewAI版本的关键词匹配，路由依据是LLM结构化分类的输出，
    且综合了情感分析（angry客户直接升级）。
    """
    priority = state.get("predicted_priority", "medium")
    intent = state.get("intent", "general")
    sentiment = state.get("sentiment", "neutral")

    if priority in ("high", "urgent") or sentiment == "angry" or intent == "complaint":
        return "escalation_agent"
    if intent == "technical":
        return "technical_agent"
    return "support_agent"


def route_after_qa(state: SupportState) -> str:
    """
    QA后路由：不达标且未超过重试上限 → 回到原生成Agent重试；否则输出。

    体现LangGraph的循环控制能力（CrewAI的Sequential Process做不到）。
    """
    score = state.get("qa_score", 10.0)
    retries = state.get("retry_count", 0)

    if score < QA_THRESHOLD and retries <= MAX_RETRIES:
        # 回到生成本次回复的Agent重试
        return state.get("handled_by", "support_agent")
    return "finalize"
