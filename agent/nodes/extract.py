"""
Node 2: Fact Extraction
Витягує structured facts (triplets) з teaching message.
"""

import logging
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from agent.state import AgentState
from clients.llm_client import get_llm_client
from langsmith import traceable

logger = logging.getLogger(__name__)


class ExtractedFact(BaseModel):
    """Single extracted fact in triplet format"""
    subject: str = Field(description="Subject entity")
    relation: str = Field(description="Relation/predicate between entities")
    object: str = Field(description="Object entity")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in this fact")


class FactExtractionResult(BaseModel):
    """Result of fact extraction containing list of facts"""
    facts: List[ExtractedFact] = Field(default_factory=list)


@traceable(name="extract_facts")
async def extract_facts_node(state: AgentState) -> Dict[str, Any]:
    """
    Витягує knowledge triplets з teaching message.
    
    Domain-agnostic extraction - працює з будь-яким контентом.
    Epistemic dimension: confidence для кожного fact показує
    certainty агента про витягнутий факт.
    
    Implements knowledge quality assessment через:
    - Structured extraction (subject-relation-object)
    - Confidence scores для epistemic awareness
    
    Args:
        state: Current agent state with message_text
    
    Returns:
        State update with extracted_facts list
    """
    logger.info("=== Extract Facts Node ===")
    logger.info(f"Message: {state['message_text'][:100]}...")
    
    llm = get_llm_client()
    
    prompt = f"""Витягни усі факти з повідомлення у форматі triplets: (subject, relation, object).

Повідомлення: {state['message_text']}

Правила:
- Subject і Object - сутності (люди, місця, об'єкти, концепти)
- Relation - зв'язок між ними (дієслово або відношення)
- Confidence - твоя впевненість 0.0-1.0 (1.0 = впевнений, 0.5 = можливо)

Приклади:
"Київ - столиця України" -> {{subject: "Київ", relation: "є_столицею", object: "України", confidence: 1.0}}
"Мабуть Олег любить каву" -> {{subject: "Олег", relation: "любить", object: "кава", confidence: 0.7}}
"Сьогодні я купив хліб" -> {{subject: "я", relation: "купив", object: "хліб", confidence: 1.0}}

Витягни ВСІ факти з повідомлення. Не вигадуй - тільки те що є в тексті.
Якщо факт виражений невпевнено (мабуть, можливо, здається) - зменш confidence."""
    
    try:
        result = await llm.generate_async(
            messages=[{"role": "user", "content": prompt}],
            response_format=FactExtractionResult,
            temperature=0.1
        )
        
        # Convert Pydantic models to dicts
        facts = [
            {
                "subject": f.subject,
                "relation": f.relation,
                "object": f.object,
                "confidence": f.confidence
            }
            for f in result.facts
        ]
        
        logger.info(f"Extracted {len(facts)} facts")
        for fact in facts:
            logger.info(
                f"  - {fact['subject']} {fact['relation']} {fact['object']} "
                f"(confidence: {fact['confidence']:.2f})"
            )
        
        return {"extracted_facts": facts}
        
    except Exception as e:
        logger.error(f"Error extracting facts: {e}", exc_info=True)
        # Return empty facts list on error
        return {"extracted_facts": []}
