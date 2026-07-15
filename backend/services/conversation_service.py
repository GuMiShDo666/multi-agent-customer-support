"""对话和消息的持久化服务。"""

from datetime import datetime
from typing import List, Optional

from sqlalchemy.orm import Session

from ..models.conversation import (
    Conversation,
    ConversationDB,
    Message,
    MessageCreate,
    MessageDB,
)


class ConversationService:
    """对话管理服务。"""

    @staticmethod
    def create_conversation(
        db: Session,
        customer_id: str,
        ticket_id: Optional[int] = None,
        *,
        commit: bool = True,
    ) -> Conversation:
        db_conv = ConversationDB(customer_id=customer_id, ticket_id=ticket_id)
        db.add(db_conv)
        if commit:
            db.commit()
        else:
            db.flush()
        db.refresh(db_conv)
        return Conversation.model_validate(db_conv)

    @staticmethod
    def get_conversation(db: Session, conversation_id: int) -> Optional[Conversation]:
        db_conv = (
            db.query(ConversationDB)
            .filter(ConversationDB.id == conversation_id)
            .first()
        )
        if not db_conv:
            return None
        conv = Conversation.model_validate(db_conv)
        conv.messages = [Message.model_validate(m) for m in db_conv.messages]
        return conv

    @staticmethod
    def get_conversation_by_ticket(
        db: Session, ticket_id: int
    ) -> Optional[Conversation]:
        db_conv = (
            db.query(ConversationDB)
            .filter(ConversationDB.ticket_id == ticket_id)
            .first()
        )
        if not db_conv:
            return None
        conv = Conversation.model_validate(db_conv)
        conv.messages = [Message.model_validate(m) for m in db_conv.messages]
        return conv

    @staticmethod
    def add_message(
        db: Session, message_data: MessageCreate, *, commit: bool = True
    ) -> Message:
        db_message = MessageDB(
            conversation_id=message_data.conversation_id,
            role=message_data.role,
            content=message_data.content,
            agent_name=message_data.agent_name,
            intent=message_data.intent,
            sentiment=message_data.sentiment,
            qa_score=message_data.qa_score,
        )
        db.add(db_message)

        db_conv = (
            db.query(ConversationDB)
            .filter(ConversationDB.id == message_data.conversation_id)
            .first()
        )
        if db_conv:
            db_conv.updated_at = datetime.utcnow()

        if commit:
            db.commit()
        else:
            db.flush()
        db.refresh(db_message)
        return Message.model_validate(db_message)

    @staticmethod
    def get_messages(db: Session, conversation_id: int) -> List[Message]:
        db_messages = (
            db.query(MessageDB)
            .filter(MessageDB.conversation_id == conversation_id)
            .order_by(MessageDB.created_at)
            .all()
        )
        return [Message.model_validate(m) for m in db_messages]
