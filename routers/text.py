"""
Router для /text endpoint з knowledge-centered agent.
"""

from fastapi import APIRouter, HTTPException
from datetime import datetime
import logging

from routers.schemas import TextRequest, TextResponse
from agent.graph import get_agent_app
from agent.state import create_initial_state
from db.message_repository import save_message_quick

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/text", response_model=TextResponse)
async def process_text(request: TextRequest) -> TextResponse:
    """
    Обробка повідомлення через knowledge-centered agent.
    
    Flow:
    1. Create initial state з request
    2. Invoke agent graph (classify → teach/solve branch)
    3. Return response з references
    
    Args:
        request: TextRequest з полями text, uid, user_id
    
    Returns:
        TextResponse з response, references, reasoning
    
    Raises:
        HTTPException: If agent processing fails
    """
    time_start = datetime.now()
    logger.info(f"Processing text request: uid={request.uid}, user={request.user_id}")
    logger.info(f"Message: {request.text[:100]}...")
    
    try:
        # Get agent instance
        agent = get_agent_app()
        
        # Save message to database
        timestamp = datetime.now()

        # Create initial state
        initial_state = create_initial_state(
            message_uid=request.uid,
            message_text=request.text,
            user_id=request.user_id,
            timestamp=timestamp,
        )

        logger.debug("Invoking agent graph...")
        
        # Invoke graph
        result = await agent.ainvoke(
            initial_state,
            config={"configurable": {"thread_id": request.uid}}
        )
        
        logger.info(f"Agent completed: intent={result.get('intent')}")
        
        # Extract response
        response_text = result.get("response", "")
        references = result.get("sources", [])
        reasoning = result.get("reasoning")
        
        # Validate response
        if not response_text:
            logger.error("Agent returned empty response")
            raise ValueError("Agent returned empty response")
        
        logger.info(f"Returning response with {len(references)} references")
        time_end = datetime.now()
        logger.info(f"Processing time: {time_end - time_start}")
        result = TextResponse(
            response=response_text,
            references=references,
            reasoning=""
        )
        logger.info(f"Returning response: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Agent processing error: {str(e)}"
        )
