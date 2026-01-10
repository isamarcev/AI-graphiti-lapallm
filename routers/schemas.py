"""
Pydantic schemas for FastAPI request/response models.
"""

from pydantic import BaseModel, Field
from typing import Optional


class TextRequest(BaseModel):
    """Request model for POST /text endpoint."""

    text: str = Field(..., description="Повідомлення користувача до агента")
    uid: str = Field(..., description="ID цього повідомлення (UUID v4)")


class TextResponse(BaseModel):
    """Response model for POST /text endpoint."""

    response: str = Field(..., description="Фінальна відповідь агента на завдання")
    references: list[str] = Field(
        default_factory=list,
        description="Масив UUID повідомлень, на які спирався агент"
    )
    reasoning: Optional[str] = Field(
        None,
        description="(Опційно) Опис процесу прийняття рішення"
    )