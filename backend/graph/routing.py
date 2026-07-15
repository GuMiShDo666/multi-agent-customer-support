"""
条件路由函数 — 独立模块，不依赖 langgraph，便于单元测试。
"""

from ..config import MAX_RETRIES, QA_THRESHOLD
from .state import SupportState


def route_to_agent(state: SupportState) -> str:
    """根据意图、优先级和情感选择处理Agent。"""
    priority = state.get("predicted_priority", "medium")
    intent = state.get("intent", "general")
    sentiment = state.get("sentiment", "neutral")

    if priority in ("high", "urgent") or sentiment == "angry" or intent == "complaint":
        return "escalation_agent"
    if intent == "technical":
        return "technical_agent"
    return "support_agent"


def route_after_qa(state: SupportState) -> str:
    """QA未达标且未超过上限时回到原Agent，否则结束。"""
    score = state.get("qa_score", 10.0)
    retries = state.get("retry_count", 0)

    if score < QA_THRESHOLD and retries <= MAX_RETRIES:
        return state.get("handled_by", "support_agent")
    return "finalize"
