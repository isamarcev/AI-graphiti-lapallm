"""
Query Analyzer Node - аналізує запит користувача та формує стратегію пошуку.

Цей вузол розуміє ЩО потрібно для відповіді та генерує оптимізовані пошукові запити.
"""

import logging
from typing import Dict, Any
from langsmith import traceable

from agent.state import AgentState
from models.schemas import QueryAnalysis
from clients.llm_client import get_llm_client

logger = logging.getLogger(__name__)


def _build_analysis_prompt(message_text: str) -> list[dict[str, str]]:
    """
    Формує промпт для аналізу запиту користувача.

    Args:
        message_text: Запит користувача

    Returns:
        Список повідомлень для LLM
    """
    system_prompt = """Ти експерт з аналізу запитів користувачів для AI-агента.

**ТВОЯ ЗАДАЧА:**
Проаналізуй запит користувача та визнач:
1. Тип запиту (питання, завдання, пояснення тощо)
2. Предметну область
3. Ключові сутності (імена, місця, об'єкти)
4. Інструменти/методи/технології (якщо згадані)
5. ЩО потрібно знати для повної відповіді
6. Оптимальні пошукові запити для збору інформації

**ТИПИ ЗАПИТІВ:**
- factual_question: питання про конкретний факт ("Яка столиця?", "Скільки?", "Хто?")
- task: завдання на виконання ("Знайди", "Порівняй", "Створи список", "Реалізуй", "Зроби")
- explanation: запит на пояснення ("Чому?", "Як працює?", "Розкажи про")
- comparison: порівняння ("Що краще?", "В чому різниця?")
- other: інші типи запитів

**ПРЕДМЕТНІ ОБЛАСТІ (приклади):**
їжа, робота, хобі, місце_проживання, навчання, здоров'я, подорожі, загальне тощо

**ВИДІЛЕННЯ ІНСТРУМЕНТІВ/МЕТОДІВ:**
Шукай паттерни у запиті:
- "за допомогою X" → required_tools_or_methods: ["X"]
- "мовою X" / "на X" → required_tools_or_methods: ["X"]
- "використовуючи X" / "інструментом X" → required_tools_or_methods: ["X"]
- "методом X" / "технологією X" → required_tools_or_methods: ["X"]

Приклади:
- "Приготуй борщ за допомогою мультиварки" → ["мультиварка"]
- "Реалізуй алгоритм мовою X" → ["мова X"]
- "Зроби розрахунок методом Y" → ["метод Y"]

**ФОРМУВАННЯ ІНФОРМАЦІЙНИХ ПОТРЕБ:**
Подумай: які факти потрібні для ПОВНОЇ відповіді?

Для ВСІХ типів запитів включай:
1. Предметну інформацію (що саме треба знати)
2. Інформацію про інструменти/методи (якщо є в required_tools_or_methods)

Приклади:
- "Що любить їсти Марія?" → ["улюблені страви Марії", "харчові вподобання Марії"]
- "Де працює Олег?" → ["місце роботи Олега", "посада Олега", "компанія Олега"]
- "Зроби X за допомогою Y" → ["як виконати X", "як працює Y", "приклади використання Y для X"]

**ГЕНЕРАЦІЯ ПОШУКОВИХ ЗАПИТІВ:**
Створи 1-5 різних запитів для знаходження потрібної інформації.

ВАЖЛИВО: search_queries мають бути РІЗНИМИ від information_needs!
- information_needs = "що треба знати" (чек-лист)
- search_queries = "як шукати" (конкретні фрази для пошуку)

Стратегії формування:
1. Основний запит (скорочений/перефразований)
2. Пошук по сутностях окремо
3. Пошук по інструментах/методах (якщо є)
4. Альтернативні формулювання
5. Комбінації ключових слів

**ВАЖЛИВО:**
- information_needs має включати ВСІ потреби: предметні + технічні (про інструменти)
- search_queries мають бути оптимізовані для ПОШУКУ (короткі, ключові слова)
- НЕ дублюй search_queries та information_needs
- Фокусуйся на КЛЮЧОВІЙ інформації для відповіді

**ПРИКЛАДИ:**

Запит: "Що любить їсти Марія?"
→ query_type: "factual_question"
→ domain: "їжа"
→ key_entities: ["Марія"]
→ information_needs: ["улюблені страви Марії", "харчові вподобання Марії"]
→ search_queries: ["що любить їсти Марія", "улюблена їжа Марії", "Марія їжа вподобання"]

Запит: "Де працює Олег та яка у нього посада?"
→ query_type: "factual_question"
→ domain: "робота"
→ key_entities: ["Олег"]
→ information_needs: ["місце роботи Олега", "посада Олега", "компанія Олега"]
→ search_queries: ["де працює Олег", "посада Олега", "робота Олег компанія"]

Запит: "Розкажи про хобі Ірини"
→ query_type: "explanation"
→ domain: "хобі"
→ key_entities: ["Ірина"]
→ information_needs: ["захоплення Ірини", "хобі Ірини", "чим займається Ірина у вільний час"]
→ search_queries: ["хобі Ірини", "Ірина захоплення", "Ірина вільний час"]

Запит: "Порівняй де живуть Марія та Олег"
→ query_type: "comparison"
→ domain: "місце_проживання"
→ key_entities: ["Марія", "Олег"]
→ information_needs: ["де живе Марія", "де живе Олег", "місто Марії", "місто Олега"]
→ search_queries: ["де живе Марія", "де живе Олег", "місце проживання Марія Олег"]

Запит: "Знайди всі факти про Київ"
→ query_type: "task"
→ domain: "загальне"
→ key_entities: ["Київ"]
→ required_tools_or_methods: []
→ information_needs: ["факти про Київ", "інформація про Київ"]
→ search_queries: ["Київ факти", "Київ інформація", "про Київ"]

Запит: "Приготуй борщ за допомогою мультиварки"
→ query_type: "task"
→ domain: "їжа"
→ key_entities: ["борщ", "мультиварка"]
→ required_tools_or_methods: ["мультиварка"]
→ information_needs: ["рецепт борщу", "як готувати в мультиварці", "час приготування в мультиварці"]
→ search_queries: ["борщ мультиварка", "рецепт борщу", "мультиварка приготування"]

Запит: "Зроби розрахунок методом найменших квадратів"
→ query_type: "task"
→ domain: "загальне"
→ key_entities: ["розрахунок", "метод найменших квадратів"]
→ required_tools_or_methods: ["метод найменших квадратів"]
→ information_needs: ["як працює метод найменших квадратів", "формули методу", "приклади застосування методу"]
→ search_queries: ["метод найменших квадратів", "розрахунок найменші квадрати", "формули найменші квадрати"]"""

    user_prompt = f"""**ЗАПИТ КОРИСТУВАЧА:**
{message_text}

Проаналізуй цей запит та визнач тип, предметну область, ключові сутності, інформаційні потреби та оптимальні пошукові запити."""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


@traceable(name="query_analyzer")
async def query_analyzer_node(state: AgentState) -> Dict[str, Any]:
    """
    Аналізує запит користувача та формує стратегію пошуку контексту.

    Цей вузол:
    1. Визначає тип запиту та предметну область
    2. Виділяє ключові сутності
    3. Формує чек-лист необхідної інформації для повної відповіді
    4. Генерує оптимізовані пошукові запити

    Args:
        state: AgentState з message_text

    Returns:
        State update з query_analysis
    """
    logger.info("=== Query Analyzer Node ===")

    message_text = state.get("message_text", "")

    if not message_text:
        logger.warning("Empty message_text, skipping analysis")
        # Fallback: створюємо базовий аналіз
        return {
            "query_analysis": {
                "query_type": "other",
                "domain": "загальне",
                "key_entities": [],
                "required_tools_or_methods": [],
                "information_needs": ["відповідь на запит"],
                "search_queries": [message_text] if message_text else []
            }
        }

    logger.info(f"Analyzing query: '{message_text}'")

    try:
        # Формуємо промпт
        messages = _build_analysis_prompt(message_text)

        # Викликаємо LLM зі structured output
        llm_client = get_llm_client()
        analysis: QueryAnalysis = await llm_client.generate_async(
            messages=messages,
            temperature=0.2,  # Низька температура для консистентності
            max_tokens=500,
            response_format=QueryAnalysis
        )

        # Логуємо результати аналізу
        logger.info(f"Query type: {analysis.query_type}")
        logger.info(f"Domain: {analysis.domain}")
        logger.info(f"Key entities: {analysis.key_entities}")
        logger.info(f"Required tools/methods: {analysis.required_tools_or_methods}")
        logger.info(f"Information needs ({len(analysis.information_needs)}): {analysis.information_needs}")
        logger.info(f"Search queries ({len(analysis.search_queries)}): {analysis.search_queries}")

        # Конвертуємо Pydantic модель у dict для state
        analysis_dict = {
            "query_type": analysis.query_type,
            "domain": analysis.domain,
            "key_entities": analysis.key_entities,
            "required_tools_or_methods": analysis.required_tools_or_methods,
            "information_needs": analysis.information_needs,
            "search_queries": analysis.search_queries
        }

        return {"query_analysis": analysis_dict}

    except Exception as e:
        logger.error(f"Error during query analysis: {e}", exc_info=True)
        logger.warning("Falling back to basic analysis")

        # Graceful fallback: використовуємо оригінальний запит
        return {
            "query_analysis": {
                "query_type": "other",
                "domain": "загальне",
                "key_entities": [],
                "required_tools_or_methods": [],
                "information_needs": ["відповідь на запит"],
                "search_queries": [message_text]
            }
        }
