"""Repository for message operations."""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError

from .models import Message
from .connection import get_session_manager


class MessageRepository:
    """Repository for message database operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def save_message(
        self,
        message_uid: str,
        user_id: str,
        message_text: str,
        timestamp: datetime,
        intent: Optional[str] = None,
    ) -> Message:
        """
        Save user message to database.

        Args:
            message_uid: Unique message identifier
            user_id: User identifier
            message_text: Original message text
            timestamp: Message timestamp
            intent: Optional intent classification (learn/solve)

        Returns:
            Message: Created message record

        Raises:
            ValueError: If message_uid already exists
        """
        message = Message(
            message_uid=message_uid,
            user_id=user_id,
            message_text=message_text,
            timestamp=timestamp,
            intent=intent,
        )

        try:
            self.session.add(message)
            await self.session.commit()
            await self.session.refresh(message)
            return message
        except IntegrityError as e:
            await self.session.rollback()
            raise ValueError(f"Message with UID '{message_uid}' already exists") from e

    async def get_message_by_uid(self, message_uid: str) -> Optional[Message]:
        """
        Retrieve original message by UID.

        Args:
            message_uid: Unique message identifier

        Returns:
            Optional[Message]: Message record or None if not found
        """
        stmt = select(Message).where(Message.message_uid == message_uid)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_user_messages(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Message]:
        """
        Get message history for a user (ordered by timestamp desc).

        Args:
            user_id: User identifier
            limit: Maximum number of messages to return
            offset: Number of messages to skip

        Returns:
            List[Message]: List of user messages
        """
        stmt = (
            select(Message)
            .where(Message.user_id == user_id)
            .order_by(desc(Message.timestamp))
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def count_user_messages(self, user_id: str) -> int:
        """
        Count total messages for a user.

        Args:
            user_id: User identifier

        Returns:
            int: Total message count
        """
        from sqlalchemy import func

        stmt = select(func.count(Message.id)).where(Message.user_id == user_id)
        result = await self.session.execute(stmt)
        return result.scalar() or 0


# Convenience function for one-off operations
async def save_message_quick(
    message_uid: str,
    user_id: str,
    message_text: str,
    timestamp: datetime,
    intent: Optional[str] = None,
) -> Message:
    """
    Quick save message function with automatic session management.

    For use in routers and other places where you just need to save quickly.
    """
    manager = await get_session_manager()
    async for session in manager.get_session():
        repo = MessageRepository(session)
        return await repo.save_message(message_uid, user_id, message_text, timestamp, intent)
