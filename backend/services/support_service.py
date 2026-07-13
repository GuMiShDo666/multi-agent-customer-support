"""将客服状态图与对话、工单持久化集成。"""

from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from ..models.conversation import MessageCreate
from ..models.ticket import TicketStatus, TicketUpdate
from .conversation_service import ConversationService
from .ticket_service import TicketService


class SupportResourceNotFound(Exception):
    """请求引用的客服资源不存在。"""


class SupportAccessDenied(Exception):
    """客户无权访问请求引用的客服资源。"""


class SupportService:
    """基于 LangGraph 多Agent工作流的客服服务。"""

    def __init__(self):
        # 延迟导入重量级 LangGraph 依赖，API 启动和健康检查无需等待状态图加载。
        from ..graph import build_support_graph

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
        try:
            ticket = None
            resolved_ticket_id = ticket_id

            if ticket_id:
                ticket = TicketService.get_ticket(db, ticket_id)
                if not ticket:
                    raise SupportResourceNotFound("Ticket not found")
                if ticket.customer_id != customer_id:
                    raise SupportAccessDenied(
                        "Ticket does not belong to this customer"
                    )

            # 获取对话前必须验证客户和工单归属。
            if conversation_id:
                conversation = ConversationService.get_conversation(
                    db, conversation_id
                )
                if not conversation:
                    raise SupportResourceNotFound("Conversation not found")
                if conversation.customer_id != customer_id:
                    raise SupportAccessDenied(
                        "Conversation does not belong to this customer"
                    )
                if ticket_id and conversation.ticket_id != ticket_id:
                    raise SupportAccessDenied(
                        "Conversation is not associated with the supplied ticket"
                    )
                if not ticket_id and conversation.ticket_id:
                    resolved_ticket_id = conversation.ticket_id
                    ticket = TicketService.get_ticket(db, resolved_ticket_id)
                    if not ticket:
                        raise SupportResourceNotFound("Associated ticket not found")
                    if ticket.customer_id != customer_id:
                        raise SupportAccessDenied(
                            "Associated ticket does not belong to this customer"
                        )
            elif ticket_id:
                conversation = ConversationService.get_conversation_by_ticket(
                    db, ticket_id
                )
                if conversation and conversation.customer_id != customer_id:
                    raise SupportAccessDenied(
                        "Conversation does not belong to this customer"
                    )
                if not conversation:
                    conversation = ConversationService.create_conversation(
                        db, customer_id, ticket_id, commit=False
                    )
            else:
                conversation = ConversationService.create_conversation(
                    db, customer_id, commit=False
                )

            # 工单优先级不能被请求中的较低优先级覆盖
            if ticket:
                order = ["low", "medium", "high", "urgent"]
                priority = max(
                    priority,
                    ticket.priority.value,
                    key=lambda value: order.index(value),
                )

            # 工作流完成前不提交消息，失败时统一回滚。
            ConversationService.add_message(
                db,
                MessageCreate(
                    conversation_id=conversation.id, role="user", content=message
                ),
                commit=False,
            )

            history = self._build_history(db, conversation.id)
            ticket_info = self._build_ticket_info(db, resolved_ticket_id)

            result = self.graph.invoke(
                {
                    "customer_id": customer_id,
                    "message": message,
                    "priority": priority,
                    "conversation_history": history,
                    "ticket_info": ticket_info,
                }
            )

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
                commit=False,
            )

            if resolved_ticket_id and ticket:
                new_status = None
                if result.get("handled_by") == "escalation_agent":
                    new_status = TicketStatus.ESCALATED
                elif ticket.status == TicketStatus.OPEN:
                    new_status = TicketStatus.IN_PROGRESS
                if new_status:
                    TicketService.update_ticket(
                        db,
                        resolved_ticket_id,
                        TicketUpdate(
                            status=new_status,
                            assigned_agent=result.get("handled_by"),
                        ),
                        commit=False,
                    )

            # 用户消息、Agent回复和工单状态必须原子提交。
            db.commit()
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
        except Exception:
            db.rollback()
            raise

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
