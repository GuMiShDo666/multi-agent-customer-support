"""
工单管理服务（沿用原项目设计）。
"""

from datetime import datetime
from typing import List, Optional

from sqlalchemy.orm import Session

from ..models.ticket import Ticket, TicketCreate, TicketDB, TicketStatus, TicketUpdate


class TicketService:
    """工单管理服务。"""

    @staticmethod
    def create_ticket(
        db: Session, ticket_data: TicketCreate, *, commit: bool = True
    ) -> Ticket:
        db_ticket = TicketDB(
            customer_id=ticket_data.customer_id,
            subject=ticket_data.subject,
            description=ticket_data.description,
            priority=ticket_data.priority,
            status=TicketStatus.OPEN,
        )
        db.add(db_ticket)
        if commit:
            db.commit()
        else:
            db.flush()
        db.refresh(db_ticket)
        return Ticket.model_validate(db_ticket)

    @staticmethod
    def get_ticket(db: Session, ticket_id: int) -> Optional[Ticket]:
        db_ticket = db.query(TicketDB).filter(TicketDB.id == ticket_id).first()
        return Ticket.model_validate(db_ticket) if db_ticket else None

    @staticmethod
    def get_tickets_by_customer(db: Session, customer_id: str) -> List[Ticket]:
        db_tickets = (
            db.query(TicketDB).filter(TicketDB.customer_id == customer_id).all()
        )
        return [Ticket.model_validate(t) for t in db_tickets]

    @staticmethod
    def get_all_tickets(db: Session, skip: int = 0, limit: int = 100) -> List[Ticket]:
        db_tickets = db.query(TicketDB).offset(skip).limit(limit).all()
        return [Ticket.model_validate(t) for t in db_tickets]

    @staticmethod
    def update_ticket(
        db: Session,
        ticket_id: int,
        ticket_update: TicketUpdate,
        *,
        commit: bool = True,
    ) -> Optional[Ticket]:
        db_ticket = db.query(TicketDB).filter(TicketDB.id == ticket_id).first()
        if not db_ticket:
            return None

        if ticket_update.status:
            db_ticket.status = ticket_update.status
            if ticket_update.status == TicketStatus.RESOLVED:
                db_ticket.resolved_at = datetime.utcnow()

        if ticket_update.priority:
            db_ticket.priority = ticket_update.priority

        if ticket_update.assigned_agent:
            db_ticket.assigned_agent = ticket_update.assigned_agent

        if ticket_update.resolution:
            db_ticket.resolution = ticket_update.resolution

        db_ticket.updated_at = datetime.utcnow()
        if commit:
            db.commit()
        else:
            db.flush()
        db.refresh(db_ticket)
        return Ticket.model_validate(db_ticket)

    @staticmethod
    def delete_ticket(db: Session, ticket_id: int) -> bool:
        db_ticket = db.query(TicketDB).filter(TicketDB.id == ticket_id).first()
        if not db_ticket:
            return False
        db.delete(db_ticket)
        db.commit()
        return True
