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
    """Single extracted fact in triplet format with epistemic dimensions"""
    subject: str = Field(description="Subject entity")
    relation: str = Field(description="Relation/predicate between entities")
    object: str = Field(description="Object entity")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in this fact (0.0-1.0)")
    temporal: bool = Field(default=False, description="True if event in time, False if permanent attribute")
    modality: str = Field(default="assertive", description="assertive | uncertain | negated")


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
    
    prompt = f"""🚫 **TABULA RASA РЕЖИМ**:
Ти починаєш з НУЛЬОВИМИ знаннями про предметну область. НЕ використовуй свої pretrained знання.
Витягуй ТІЛЬКИ те, що ЯВНО написано в повідомленні. Якщо чогось немає - не здогадуйся.

Витягни усі факти з повідомлення у форматі knowledge triplets з epistemic dimensions.

**ЕПІСТЕМІЧНИЙ ФРЕЙМВОРК:**
- **Confidence** = наскільки впевнений що це факт (НЕ твоя здогадка!)
- **Temporal** = чи факт прив'язаний до часу (event) чи постійний (attribute)
- **Modality** = assertive (стверджує) vs uncertain (можливо) vs negated (НЕ)

**ПОВІДОМЛЕННЯ:** {state['message_text']}

**ПРАВИЛА ЕКСТРАКЦІЇ:**

1. **Subject** - головна сутність (особа, місце, концепт, процедура, структура)
2. **Relation** - зв'язок (дієслово, відношення, стан, властивість)
3. **Object** - друга сутність, значення, або опис
4. **Confidence** (0.0-1.0):
   - 1.0 = чіткий факт "Київ є столицею"
   - 0.8 = ймовірно "Здається я любив"
   - 0.5 = невпевнено "Мабуть"
   - 0.3 = здогадка "Можливо"
5. **Temporal** - true = подія в часі (купив, зустрів), false = постійне (є, любить)
6. **Modality** - assertive/uncertain/negated

**FEW-SHOT EXAMPLES (універсальні типи знань):**

Input: "Київ - столиця України"
Output: {{subject: "Київ", relation: "є_столицею", object: "України", confidence: 1.0, temporal: false, modality: "assertive"}}

Input: "Щоб зробити салат, спочатку нарізаємо овочі, потім додаємо олію"
Output: [
  {{subject: "салат_приготування", relation: "перший_крок", object: "нарізати_овочі", confidence: 1.0, temporal: false, modality: "assertive"}},
  {{subject: "салат_приготування", relation: "другий_крок", object: "додати_олію", confidence: 1.0, temporal: false, modality: "assertive"}}
]

Input: "У цій системі змінна оголошується через ключове слово 'var', потім ім'я та значення"
Output: [
  {{subject: "змінна_оголошення", relation: "використовує_ключове_слово", object: "var", confidence: 1.0, temporal: false, modality: "assertive"}},
  {{subject: "змінна_оголошення", relation: "структура", object: "var ім'я значення", confidence: 1.0, temporal: false, modality: "assertive"}}
]

Input: "Мабуть Олег любить каву, принаймні він часто п'є її"
Output: [
  {{subject: "Олег", relation: "любить", object: "кава", confidence: 0.7, temporal: false, modality: "uncertain"}},
  {{subject: "Олег", relation: "п'є", object: "кава", confidence: 1.0, temporal: false, modality: "assertive"}},
  {{subject: "пиття_кави", relation: "має_частоту", object: "часто", confidence: 1.0, temporal: false, modality: "assertive"}}
]

Input: "Процес сортування: порівнюємо сусідні елементи і міняємо місцями якщо не в порядку"
Output: [
  {{subject: "процес_сортування", relation: "крок_алгоритму", object: "порівняти_сусідні_елементи", confidence: 1.0, temporal: false, modality: "assertive"}},
  {{subject: "процес_сортування", relation: "крок_алгоритму", object: "поміняти_місцями_якщо_не_в_порядку", confidence: 1.0, temporal: false, modality: "assertive"}}
]

Input: "Значення pi дорівнює 3.14"
Output: {{subject: "pi", relation: "має_значення", object: "3.14", confidence: 1.0, temporal: false, modality: "assertive"}}

Input: "Я більше НЕ працюю в Google"
Output: {{subject: "я", relation: "працює_в", object: "Google", confidence: 1.0, temporal: true, modality: "negated"}}

Input: "У мене алергія на арахіс"
Output: {{subject: "я", relation: "має_алергію_на", object: "арахіс", confidence: 1.0, temporal: false, modality: "assertive"}}

**КРИТИЧНО:**
- 🚫 НЕ використовуй pretrained knowledge - витягуй ТІЛЬКИ з повідомлення
- НЕ вигадуй факти яких немає
- Витягуй ВСІ факти, включно з процедурами, структурами, визначеннями
- Uncertainty маркери ("мабуть", "здається") → modality="uncertain" + confidence↓
- Temporal: події=true, атрибути=false
- Negation ("НЕ", "більше не") → modality="negated"

Витягни факти з повідомлення вище."""
    
    try:
        result = await llm.generate_async(
            messages=[{"role": "user", "content": prompt}],
            response_format=FactExtractionResult,
            temperature=0.1
        )
        
        # Convert Pydantic models to dicts with epistemic dimensions
        facts = [
            {
                "subject": f.subject,
                "relation": f.relation,
                "object": f.object,
                "confidence": f.confidence,
                "temporal": f.temporal,
                "modality": f.modality
            }
            for f in result.facts
        ]

        logger.info(f"Extracted {len(facts)} facts")
        for fact in facts:
            logger.info(
                f"  - {fact['subject']} {fact['relation']} {fact['object']} "
                f"(conf: {fact['confidence']:.2f}, temporal: {fact['temporal']}, modality: {fact['modality']})"
            )
        
        return {"extracted_facts": facts}
        
    except Exception as e:
        logger.error(f"Error extracting facts: {e}", exc_info=True)
        # Return empty facts list on error
        return {"extracted_facts": []}
