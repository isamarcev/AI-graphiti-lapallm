"""
Router for /text endpoint.
"""

from fastapi import APIRouter, HTTPException
from routers.schemas import TextRequest, TextResponse

router = APIRouter()


@router.post("/text", response_model=TextResponse)
async def process_text(request: TextRequest) -> TextResponse:
    """
    Обробка текстового повідомлення від користувача.

    Args:
        request: TextRequest з полями text та uid

    Returns:
        TextResponse з відповіддю агента, посиланнями та reasoning
    """
    # TODO: Implement agent logic here
    # For now, return a placeholder response

    return TextResponse(
        response=f"Отримано повідомлення: {request.text}",
        references=[request.uid],
        reasoning="Placeholder reasoning - логіка агента буде імплементована пізніше"
    )