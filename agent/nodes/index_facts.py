import logging
import json
import asyncio
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
        "- Будь дуже уважний з прикладами. Кожен приклад має бути повним\n"
        "- Examples має бути масивом; якщо прикладів немає, залиш його порожнім.\n"
        "Повідомлення:\n"
        f"{message_text}"
    )


async def process_single_fact(
        update_text: str,
        llm: Any,
        system_prompt: str,
        semaphore: asyncio.Semaphore
) -> Dict[str, Any]:
    """
    Обрабатывает один факт изолированно.
    """
    async with semaphore:  # Ограничиваем количество одновременных запросов
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

            return result.dict()

        except Exception as e:
            logger.error(f"Error extracting fact: {e}", exc_info=True)
            # Return empty for saving
            return {"fact": None, "description": None, "examples": []}


@traceable(name="index_fact")
async def index_facts_node(state: AgentState) -> Dict[str, Any]:
    """
    Node: Extract concise fact, description, and examples for each memory update.
    PARALLEL VERSION
    """
    logger.info("=== Index Fact Node (Parallel) ===")

    llm = get_llm_client()
    system_prompt = (
        "Твоя мета - розкласти надану інформацію на короткий факт, його розгорнутий опис та приклади. "
        "Використовуй лише інформацію з тексту повідомлення. Повертай ЛИШЕ валідний JSON, без прози."
    )

    memory_updates = state.get("memory_updates") or []

    if not memory_updates:
        return {"indexed_facts": []}

    # Семафор на 5-10 одновременных запросов
    semaphore = asyncio.Semaphore(5)

    tasks = []
    for update_text in memory_updates:
        task = process_single_fact(
            update_text=update_text,
            llm=llm,
            system_prompt=system_prompt,
            semaphore=semaphore
        )
        tasks.append(task)

    # Запускаем все задачи одновременно и ждем их завершения
    indexed_facts = await asyncio.gather(*tasks)

    logger.info(f"Processed {len(indexed_facts)} facts in parallel")

    return {"indexed_facts": indexed_facts}