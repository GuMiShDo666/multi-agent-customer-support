"""
SupportState — 贯穿整个 LangGraph 工作流的共享状态定义。

LangGraph 的核心思想：每个节点接收当前状态，返回状态的增量更新（partial dict），
框架负责合并。相比 CrewAI 的隐式黑盒编排，状态显式、可追踪、可调试。
"""

from typing import Annotated, Any, Dict, List, Optional, TypedDict


def append_trace(existing: List[str], new: List[str]) -> List[str]:
    """trace 字段的 reducer：追加而不是覆盖，用于记录完整执行轨迹。"""
    return existing + new


class SupportState(TypedDict, total=False):
    # ===== 输入 =====
    customer_id: str
    message: str                          # 客户当前消息
    priority: str                         # low / medium / high / urgent
    conversation_history: List[Dict[str, Any]]  # 历史对话上下文
    ticket_info: Optional[Dict[str, Any]]        # 关联工单信息

    # ===== 意图分类节点输出 =====
    intent: str            # general / technical / billing / complaint
    intent_confidence: float
    sentiment: str         # positive / neutral / negative / angry
    predicted_priority: str  # LLM根据内容预测的优先级（可能高于用户声明的）

    # ===== RAG检索节点输出 =====
    retrieved_docs: List[Dict[str, str]]  # 检索到的知识库文档

    # ===== Agent节点输出 =====
    draft_response: str    # 当前候选回复
    handled_by: str        # 本次生成回复的agent名
    escalation_summary: str  # escalation agent 的协调摘要

    # ===== QA质检节点输出 =====
    qa_score: float        # 0-10 质量分
    qa_feedback: str       # 质检意见（重试时反馈给生成节点）
    retry_count: int       # 已重试次数

    # ===== 最终输出 =====
    final_response: str
    agents_used: List[str]

    # ===== 可观测性：完整执行轨迹（追加合并） =====
    trace: Annotated[List[str], append_trace]
