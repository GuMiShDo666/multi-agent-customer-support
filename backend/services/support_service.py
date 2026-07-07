"""
支持服务 — 将 LangGraph 状态图与数据库持久化层集成。

这是原项目 SupportService 的 LangGraph 版：
原来调用 CrewManager.process_inquiry()，现在调用编译好的状态图 graph.invoke()。
"""

from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from ..graph import build_support_graph
from ..models.conversation import MessageCreate
from ..models.ticket import TicketStatus, TicketUpdate
from .conversation_service import ConversationService
from .ticket_service import TicketService


class SupportService:
    """基于 LangGraph 多Agent工作流的客服服务。"""

    def __init__(self):
        # 图只编译一次，可重复调用（无状态，状态由每次invoke的输入携带）
        self.graph = build_support_graph()

    def handle_customer_message(
        self,
        db: Session,
        customer_id: str,
        message: str,
        conversation_id: Optional[int] = None,
        ticket_id: Optional[int] = None,
        priority: str = "medium",
    ) -> Dict[str, Any]:
        """
        处理客户消息：持久化 → 执行状态图 → 持久化回复 → 返回结果+执行轨迹。
        """
        # 1. 获取或创建对话
        if conversation_id:
            conversation = ConversationService.get_conversation(db, conversation_id)
        elif ticket_id:
            conversation = ConversationService.get_conversation_by_ticket(db, ticket_id)
            if not conversation:
                conversation = ConversationService.create_conversation(
                    db, customer_id, ticket_id
                )
        else:
            conversation = ConversationService.create_conversation(
                db, customer_id, ticket_id
            )

        # 2. 保存客户消息
        ConversationService.add_message(
            db,
            MessageCreate(
                conversation_id=conversation.id, role="user", content=message
            ),
        )

        # 3. 构建上下文
        history = self._build_history(db, conversation.id)
        ticket_info = self._build_ticket_info(db, ticket_id)

        # 4. 执行 LangGraph 状态图
        result = self.graph.invoke(
            {
                "customer_id": customer_id,
                "message": message,
                "priority": priority,
                "conversation_history": history,
                "ticket_info": ticket_info,
            }
        )

        # 5. 保存Agent回复（含执行元数据）
        ConversationService.add_message(
            db,
            MessageCreate(
                conversation_id=conversation.id,
                role="assistant",
                content=result["final_response"],
                agent_name=result.get("handled_by"),
                intent=result.get("intent"),
                sentiment=result.get("sentiment"),
                qa_score=result.get("qa_score"),
            ),
        )

        # 6. 更新工单状态
        if ticket_id:
            ticket = TicketService.get_ticket(db, ticket_id)
            if ticket:
                new_status = None
                if result.get("handled_by") == "escalation_agent":
                    new_status = TicketStatus.ESCALATED
                elif ticket.status == TicketStatus.OPEN:
                    new_status = TicketStatus.IN_PROGRESS
                if new_status:
                    TicketService.update_ticket(
                        db,
                        ticket_id,
                        TicketUpdate(
                            status=new_status,
                            assigned_agent=result.get("handled_by"),
                        ),
                    )

        # 7. 返回结果 + 完整可观测性数据
        return {
            "conversation_id": conversation.id,
            "response": result["final_response"],
            "agent": result.get("handled_by"),
            "agents_used": result.get("agents_used", []),
            "metadata": {
                "intent": result.get("intent"),
                "intent_confidence": result.get("intent_confidence"),
                "sentiment": result.get("sentiment"),
                "effective_priority": result.get("predicted_priority"),
                "qa_score": result.get("qa_score"),
                "retry_count": result.get("retry_count"),
                "retrieved_docs": [
                    d["title"] for d in result.get("retrieved_docs", [])
                ],
            },
            "trace": result.get("trace", []),
        }

    def _build_history(self, db: Session, conversation_id: int) -> list:
        """构建对话历史（排除刚保存的当前消息）。"""
        messages = ConversationService.get_messages(db, conversation_id)
        return [
            {"role": m.role, "content": m.content, "agent": m.agent_name}
            for m in messages[:-1]
        ]

    def _build_ticket_info(
        self, db: Session, ticket_id: Optional[int]
    ) -> Optional[Dict[str, Any]]:
        if not ticket_id:
            return None
        ticket = TicketService.get_ticket(db, ticket_id)
        if not ticket:
            return None
        return {
            "id": ticket.id,
            "subject": ticket.subject,
            "status": ticket.status.value,
            "priority": ticket.priority.value,
        }
