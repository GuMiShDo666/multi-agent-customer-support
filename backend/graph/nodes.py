"""
LangGraph 工作流节点实现。

节点清单：
  intake                 — 输入预处理
  classify_intent        — LLM意图分类 + 情感分析 + 优先级预测（结构化输出）
  retrieve_knowledge     — RAG知识库检索
  support_agent          — 一线客服Agent
  technical_agent        — 技术专家Agent
  escalation_agent       — 升级协调Agent（综合技术分析给出最终答复）
  qa_check               — QA质检打分，不达标触发重试循环
  finalize               — 输出整理

每个节点都是纯函数：接收 SupportState，返回增量更新 dict（由LangGraph合并）。
"""

import json
from typing import Any, Dict, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError

from ..config import QA_THRESHOLD, RAG_TOP_K
from ..llm import create_llm
from ..rag import get_retriever
from .state import SupportState

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


class IntentClassification(BaseModel):
    intent: Literal["general", "technical", "billing", "complaint"]
    confidence: float = Field(ge=0.0, le=1.0)
    sentiment: Literal["positive", "neutral", "negative", "angry"]
    predicted_priority: Literal["low", "medium", "high", "urgent"]


class QAEvaluation(BaseModel):
    score: float = Field(ge=0.0, le=10.0)
    feedback: str = Field(min_length=1, max_length=2000)


def _parse_json(text: Any) -> Dict[str, Any]:
    """提取第一个完整JSON对象，容忍代码块或前后说明文字。"""
    if not isinstance(text, str):
        return {}
    start = text.find("{")
    if start < 0:
        return {}
    try:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        pass
    return {}


def _format_history(state: SupportState, max_turns: int = 6) -> str:
    """格式化最近的对话历史作为上下文。"""
    history = state.get("conversation_history", [])[-max_turns:]
    if not history:
        return "（无历史对话）"
    lines = []
    for msg in history:
        role = "客户" if msg.get("role") == "user" else f"客服({msg.get('agent', 'AI')})"
        lines.append(f"{role}: {msg.get('content', '')}")
    return "\n".join(lines)


def _format_docs(state: SupportState) -> str:
    """格式化检索到的知识库文档。"""
    docs = state.get("retrieved_docs", [])
    if not docs:
        return "（未检索到相关知识库内容）"
    return "\n\n".join(
        f"【参考{i+1}】{d['title']}\n{d['content']}" for i, d in enumerate(docs)
    )


# ---------------------------------------------------------------------------
# 节点实现
# ---------------------------------------------------------------------------


def intake(state: SupportState) -> Dict[str, Any]:
    """输入预处理：初始化计数器和轨迹。"""
    return {
        "retry_count": 0,
        "agents_used": [],
        "trace": [f"[intake] 收到客户 {state.get('customer_id', 'unknown')} 的消息"],
    }


def classify_intent(state: SupportState) -> Dict[str, Any]:
    """
    意图分类节点 — 用LLM做结构化分类（替代原项目的关键词匹配）。

    输出：意图类别、置信度、情感、预测优先级。
    这是智能路由的依据，是相对原CrewAI版本的核心改进之一。
    """
    llm = create_llm(temperature=0.0)  # 分类任务用确定性输出

    prompt = f"""你是客服系统的意图分类器。分析以下客户消息，输出JSON（不要输出其他内容）：

客户消息：{state['message']}

历史对话：
{_format_history(state)}

输出格式：
{{
  "intent": "general|technical|billing|complaint",
  "confidence": 0.0到1.0,
  "sentiment": "positive|neutral|negative|angry",
  "predicted_priority": "low|medium|high|urgent"
}}

判断规则：
- technical: 报错、崩溃、安装、配置、API、集成等技术问题
- billing: 账单、退款、订阅、价格问题
- complaint: 投诉、强烈不满
- general: 其他一般咨询
- 客户情绪angry或提及"投诉/退款/紧急"时 predicted_priority 至少为 high"""

    result = llm.invoke([HumanMessage(content=prompt)])
    order = ["low", "medium", "high", "urgent"]
    declared = state.get("priority", "medium")
    if declared not in order:
        declared = "medium"

    try:
        classification = IntentClassification.model_validate(
            _parse_json(result.content)
        )
        intent = classification.intent
        confidence = classification.confidence
        sentiment = classification.sentiment
        predicted = classification.predicted_priority
        classification_note = ""
    except ValidationError:
        # 分类器不可用时不应把潜在紧急问题降级为普通咨询。
        intent = "general"
        confidence = 0.0
        sentiment = "neutral"
        predicted = max(declared, "high", key=order.index)
        classification_note = ", 结构化输出无效，已按高优先级升级"

    # 取用户声明优先级与LLM预测优先级中较高者
    effective = max(declared, predicted, key=lambda p: order.index(p) if p in order else 1)

    return {
        "intent": intent,
        "intent_confidence": confidence,
        "sentiment": sentiment,
        "predicted_priority": effective,
        "trace": [
            f"[classify_intent] 意图={intent}, 情感={sentiment}, "
            f"声明优先级={declared} → 生效优先级={effective}{classification_note}"
        ],
    }


def retrieve_knowledge(state: SupportState) -> Dict[str, Any]:
    """RAG检索节点：从知识库检索与客户问题相关的文档。"""
    retriever = get_retriever()
    docs = retriever.search(state["message"], top_k=RAG_TOP_K)
    return {
        "retrieved_docs": docs,
        "trace": [f"[retrieve_knowledge] 检索到 {len(docs)} 条知识库文档"],
    }


# ----- 三个业务Agent节点 -----


def support_agent(state: SupportState) -> Dict[str, Any]:
    """一线客服Agent：处理一般咨询。"""
    llm = create_llm(temperature=0.7)

    qa_feedback = state.get("qa_feedback", "")
    feedback_part = (
        f"\n\n注意：上一版回复未通过质检，质检意见如下，请针对性改进：\n{qa_feedback}"
        if qa_feedback and state.get("retry_count", 0) > 0
        else ""
    )

    messages = [
        SystemMessage(
            content="""你是一名友好、专业的一线客服代表。
- 语气亲切、有同理心，先回应客户的情绪再解决问题
- 基于知识库参考内容回答，不编造信息
- 如果知识库没有相关内容，如实告知并给出通用建议
- 回复用中文，简洁清晰，必要时分步骤说明"""
        ),
        HumanMessage(
            content=f"""客户消息：{state['message']}

客户情绪：{state.get('sentiment', 'neutral')}

历史对话：
{_format_history(state)}

知识库参考：
{_format_docs(state)}{feedback_part}

请给出回复："""
        ),
    ]

    result = llm.invoke(messages)
    return {
        "draft_response": result.content,
        "handled_by": "support_agent",
        "agents_used": state.get("agents_used", []) + ["support_agent"],
        "trace": ["[support_agent] 已生成候选回复"],
    }


def technical_agent(state: SupportState) -> Dict[str, Any]:
    """技术专家Agent：处理技术类问题。"""
    llm = create_llm(temperature=0.3)  # 技术回答用较低温度保证准确性

    qa_feedback = state.get("qa_feedback", "")
    feedback_part = (
        f"\n\n注意：上一版回复未通过质检，质检意见：\n{qa_feedback}"
        if qa_feedback and state.get("retry_count", 0) > 0
        else ""
    )

    messages = [
        SystemMessage(
            content="""你是资深技术支持专家。
- 精准诊断问题根因，给出分步骤的排查/解决方案
- 基于知识库参考内容回答，涉及命令或配置时给出具体示例
- 无法远程确认的信息，明确列出需要客户提供什么
- 回复用中文，结构化（问题诊断 → 解决步骤 → 验证方法）"""
        ),
        HumanMessage(
            content=f"""客户技术问题：{state['message']}

历史对话：
{_format_history(state)}

知识库参考：
{_format_docs(state)}{feedback_part}

请给出技术支持回复："""
        ),
    ]

    result = llm.invoke(messages)
    return {
        "draft_response": result.content,
        "handled_by": "technical_agent",
        "agents_used": state.get("agents_used", []) + ["technical_agent"],
        "trace": ["[technical_agent] 已生成技术回复"],
    }


def escalation_agent(state: SupportState) -> Dict[str, Any]:
    """
    升级协调Agent：处理高优先级问题。

    多Agent协同体现：先调用技术Agent的分析能力生成内部诊断，
    再以升级经理身份综合诊断结果、安抚客户、给出解决方案和跟进承诺。
    """
    llm = create_llm(temperature=0.5)

    # 第一步：内部技术诊断（Agent间协作）
    diag_llm = create_llm(temperature=0.2)
    diag_result = diag_llm.invoke(
        [
            SystemMessage(content="你是技术专家，为升级工单做内部诊断。输出简洁的问题分析和建议方案（内部使用，不面向客户）。"),
            HumanMessage(
                content=f"问题：{state['message']}\n\n知识库参考：\n{_format_docs(state)}"
            ),
        ]
    )
    diagnosis = diag_result.content

    qa_feedback = state.get("qa_feedback", "")
    feedback_part = (
        f"\n\n注意：上一版回复未通过质检，质检意见：\n{qa_feedback}"
        if qa_feedback and state.get("retry_count", 0) > 0
        else ""
    )

    # 第二步：升级经理综合答复
    messages = [
        SystemMessage(
            content="""你是客服升级经理，负责处理高优先级/紧急工单。
- 首先真诚致歉并表明重视（客户已经历升级流程，情绪可能不佳）
- 综合内部技术诊断，给出明确的解决方案和时间承诺
- 提供后续跟进渠道
- 回复用中文，专业且有担当"""
        ),
        HumanMessage(
            content=f"""高优先级客户问题：{state['message']}

客户情绪：{state.get('sentiment', 'neutral')}

内部技术诊断（不要直接透露这是内部文档）：
{diagnosis}

历史对话：
{_format_history(state)}{feedback_part}

请给出升级答复："""
        ),
    ]

    result = llm.invoke(messages)
    return {
        "draft_response": result.content,
        "escalation_summary": diagnosis,
        "handled_by": "escalation_agent",
        "agents_used": state.get("agents_used", [])
        + ["technical_agent(内部诊断)", "escalation_agent"],
        "trace": ["[escalation_agent] 完成内部诊断+升级答复（2次LLM协作）"],
    }


def qa_check(state: SupportState) -> Dict[str, Any]:
    """
    QA质检节点：对候选回复打分（0-10），不达标则给出改进意见触发重试。

    这是LangGraph循环能力的体现：qa_check → (不达标) → 回到生成节点重试。
    """
    llm = create_llm(temperature=0.0)

    prompt = f"""你是客服质检专员。评估以下客服回复的质量，输出JSON（不要输出其他内容）：

客户问题：{state['message']}
客户情绪：{state.get('sentiment', 'neutral')}

客服回复：
{state.get('draft_response', '')}

知识库参考（检验回复是否与知识库一致，是否编造信息）：
{_format_docs(state)}

评分维度（每项满分2.5，总分10）：
1. 准确性：是否基于知识库、无编造
2. 完整性：是否完整回答了客户问题
3. 同理心：是否回应了客户情绪
4. 可执行性：客户能否按回复操作

输出格式：
{{"score": 0.0到10.0, "feedback": "具体改进意见（如果满分则写'通过'）"}}"""

    result = llm.invoke([HumanMessage(content=prompt)])
    try:
        evaluation = QAEvaluation.model_validate(_parse_json(result.content))
        score = evaluation.score
        feedback = evaluation.feedback
    except ValidationError:
        score = 0.0
        feedback = "质检结构化输出解析失败，按不通过处理"
    passed = score >= QA_THRESHOLD

    return {
        "qa_score": score,
        "qa_feedback": feedback,
        "retry_count": state.get("retry_count", 0) + (0 if passed else 1),
        "agents_used": state.get("agents_used", []) + ["qa_agent"],
        "trace": [
            f"[qa_check] 质检得分={score:.1f} (阈值{QA_THRESHOLD}) "
            f"{'✓通过' if passed else '✗不通过，准备重试'}"
        ],
    }


def finalize(state: SupportState) -> Dict[str, Any]:
    """输出整理节点。"""
    return {
        "final_response": state.get("draft_response", "抱歉，系统暂时无法生成回复。"),
        "trace": [
            f"[finalize] 完成。处理Agent={state.get('handled_by')}, "
            f"QA得分={state.get('qa_score', 0):.1f}, 重试={state.get('retry_count', 0)}次"
        ],
    }
