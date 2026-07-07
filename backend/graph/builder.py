"""
状态图构建 — 定义节点连接和条件路由。

工作流拓扑：

    intake → classify_intent → retrieve_knowledge → [智能路由]
                                                       ├─ support_agent ──┐
                                                       ├─ technical_agent ─┤
                                                       └─ escalation_agent ┘
                                                                │
                                                            qa_check
                                                           ╱        ╲
                                              (不达标且未超限)      (通过/超限)
                                                 回到原Agent重试 → finalize → END
"""

from langgraph.graph import END, StateGraph

from . import nodes
from .routing import route_after_qa, route_to_agent
from .state import SupportState


def build_support_graph():
    """构建并编译客服多Agent状态图。"""
    graph = StateGraph(SupportState)

    # 注册节点
    graph.add_node("intake", nodes.intake)
    graph.add_node("classify_intent", nodes.classify_intent)
    graph.add_node("retrieve_knowledge", nodes.retrieve_knowledge)
    graph.add_node("support_agent", nodes.support_agent)
    graph.add_node("technical_agent", nodes.technical_agent)
    graph.add_node("escalation_agent", nodes.escalation_agent)
    graph.add_node("qa_check", nodes.qa_check)
    graph.add_node("finalize", nodes.finalize)

    # 主干边
    graph.set_entry_point("intake")
    graph.add_edge("intake", "classify_intent")
    graph.add_edge("classify_intent", "retrieve_knowledge")

    # 智能路由（条件分支）
    graph.add_conditional_edges(
        "retrieve_knowledge",
        route_to_agent,
        {
            "support_agent": "support_agent",
            "technical_agent": "technical_agent",
            "escalation_agent": "escalation_agent",
        },
    )

    # 所有Agent → QA质检
    graph.add_edge("support_agent", "qa_check")
    graph.add_edge("technical_agent", "qa_check")
    graph.add_edge("escalation_agent", "qa_check")

    # QA后路由（重试循环 或 输出）
    graph.add_conditional_edges(
        "qa_check",
        route_after_qa,
        {
            "support_agent": "support_agent",
            "technical_agent": "technical_agent",
            "escalation_agent": "escalation_agent",
            "finalize": "finalize",
        },
    )

    graph.add_edge("finalize", END)

    return graph.compile()
