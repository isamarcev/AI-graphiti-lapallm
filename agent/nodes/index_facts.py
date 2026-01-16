import logging
import json
from typing import Any, Dict, List

from pydantic import BaseModel, ValidationError

from agent.state import AgentState
from clients.llm_client import get_llm_client
from langsmith import traceable

logger = logging.getLogger(__name__)


class FactExtraction(BaseModel):
    fact: str
    description: str
    examples: List[str]


@traceable(name="index_fact")
async def index_facts_node(state: AgentState) -> Dict[str, Any]:
    """
    Node: Extract concise fact, description, and examples for each memory update.

    Processes state["memory_updates"] (list of strings) and returns indexed_facts: List[FactExtraction].
    """

    logger.info("=== Index Fact Node ===")

    llm = get_llm_client()

    system_prompt = (
        "Твоя мета - розкласти надану інформацію на короткий факт, його розгорнутий опис та приклади. Використовуй лише інформацію з тексту повідомлення. Повертай ЛИШЕ валідний JSON, без прози."
    )

    def build_user_prompt(message_text: str) -> str:
        return (
            "Проаналізуй повідомлення і поверни JSON об'єкт, що точно відповідає цій схемі:\n"
            "{\n"
            "  \"fact\": string,               // факт коротко\n"
            "  \"description\": string,        // розгорнуте пояснення\n"
            "  \"examples\": [string, ...]     // список прикладів, залиш порожнім якщо немає\n"
            "}\n"
            "Правила:\n"
            "- Виводь лише JSON, без markdown, без code fences.\n"
            "- Використовуй лише інформацію з тексту повідомлення. Ні в якому разі не додавй нічого від себе.\n"
            "- Будь дуже уважний з прикладами.Кожен приклад має бути повним\n"
            "- Examples має бути масивом; якщо прикладів немає, залиш його порожнім.\n"
            "Повідомлення:\n"
            f"{message_text}"
        )

    indexed_facts = []
    memory_updates = state.get("memory_updates") or []

    for update_text in memory_updates:
        try:
            raw = await llm.generate_async(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": build_user_prompt(update_text)},
                ],
                temperature=0.2,
            )

            if isinstance(raw, str):
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("LLM response was not valid JSON; returning defaults")
                    parsed = {}
            elif isinstance(raw, dict):
                parsed = raw
            else:
                parsed = {}

            try:
                result = FactExtraction(**parsed)
            except ValidationError:
                logger.warning("Parsed response missing fields; filling defaults")
                result = FactExtraction(
                    fact=parsed.get("fact", ""),
                    description=parsed.get("description", ""),
                    examples=parsed.get("examples", []) or [],
                )

            indexed_facts.append(result.dict())

        except Exception as e:
            logger.error(f"Error extracting fact: {e}", exc_info=True)
            indexed_facts.append({"fact": None, "description": None, "examples": []})

    return {"indexed_facts": indexed_facts}