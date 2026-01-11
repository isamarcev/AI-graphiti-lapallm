"""
Node 4: Conflict Resolution
Asks user to resolve detected conflicts.
"""

import logging
from typing import Dict, Any

from agent.state import AgentState
from db.neo4j_helpers import get_message_store
from langsmith import traceable

logger = logging.getLogger(__name__)


@traceable(name="resolve_conflict")
async def resolve_conflict_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 4: Form context-aware conflict resolution question.

    Generates smart clarifying questions based on conflict type:
    - direct: choose which is correct
    - temporal: confirm change over time
    - contextual: both can be true in different contexts
    - degree: clarification vs replacement
    - partial: resolve partial contradiction

    This implements dialogue-driven learning from the paper.

    Args:
        state: Current agent state

    Returns:
        State update with response (type-specific question)
    """
    logger.info("=== Resolve Conflict Node ===")

    conflicts = state.get("conflicts", [])
    if not conflicts:
        logger.warning("No conflicts to resolve")
        return {
            "response": "Помилка: конфлікт не знайдено",
            "conflict_resolved": True
        }

    # Take first conflict
    conflict = conflicts[0]
    conflict_type = conflict.get("conflict_type", "direct")
    logger.info(f"Resolving {conflict_type} conflict: {conflict['description']}")

    # EPISTEMIC DIALOGUE - питання залежить від типу конфлікту

    if conflict_type == "direct":
        # Прямий конфлікт - потрібно вибрати
        question = f"""⚠️ **Виявлено пряме протиріччя:**

**Стара інформація** (повідомлення `{conflict['old_msg_uid']}`):
> {conflict['old_content']}

**Нова інформація** (повідомлення `{conflict['new_msg_uid']}`):
> {conflict['new_content']}

Ці твердження взаємовиключні. Яка інформація правильна?

1️⃣ **Нове правильне** - оновити на нову інформацію
2️⃣ **Старе правильне** - залишити як було
3️⃣ **Я помилився** - видалити нову інформацію

Будь ласка, уточніть яка інформація точна."""

    elif conflict_type == "temporal":
        # Часова зміна - підтвердити що змінилось
        question = f"""🕐 **Виявлено зміну в часі:**

**Раніше** (повідомлення `{conflict['old_msg_uid']}`):
> {conflict['old_content']}

**Зараз** (повідомлення `{conflict['new_msg_uid']}`):
> {conflict['new_content']}

Це виглядає як зміна з часом. Чи це правильно?

1️⃣ **Так, змінилось** - зберегти обидва факти з часовими мітками
2️⃣ **Ні, я помилився** - виправити на правильну інформацію
3️⃣ **Старе було неправильним** - видалити старе, залишити нове

Підтвердіть будь ласка."""

    elif conflict_type == "contextual":
        # Контекстний - обидва можуть бути правдою
        question = f"""🔀 **Виявлено контекстну різницю:**

**Перший контекст** (повідомлення `{conflict['old_msg_uid']}`):
> {conflict['old_content']}

**Другий контекст** (повідомлення `{conflict['new_msg_uid']}`):
> {conflict['new_content']}

Обидва твердження можуть бути правдою в різних ситуаціях.

1️⃣ **Обидва правильні** - зберегти обидва з контекстом
2️⃣ **Тільки одне правильне** - вибрати яке саме
3️⃣ **Уточнити контекст** - додати деталі для кожного

Як вам зручніше?"""

    elif conflict_type == "degree":
        # Різниця в ступені - уточнення
        question = f"""📊 **Виявлено уточнення ступеня:**

**Базовий рівень** (повідомлення `{conflict['old_msg_uid']}`):
> {conflict['old_content']}

**Уточнений рівень** (повідомлення `{conflict['new_msg_uid']}`):
> {conflict['new_content']}

Друге твердження уточнює перше. Як поступити?

1️⃣ **Замінити на точніше** - використовувати нову інформацію
2️⃣ **Зберегти обидва** - мати і загальне, і деталізоване
3️⃣ **Уточнити ще більше** - додати більше деталей

Що краще відображає реальність?"""

    else:  # partial or unknown
        # Fallback до загального питання
        question = f"""❓ **Виявлено невідповідність:**

**Існуюча інформація** (повідомлення `{conflict['old_msg_uid']}`):
> {conflict['old_content']}

**Нова інформація** (повідомлення `{conflict['new_msg_uid']}`):
> {conflict['new_content']}

Ці твердження можуть суперечити. Допоможіть розібратись:

1️⃣ Яка інформація точніша?
2️⃣ Обидва правильні але в різних контекстах?
3️⃣ Щось інше? (опишіть)

Будь ласка, уточніть ситуацію."""

    return {
        "response": question,
        "references": [conflict["old_msg_uid"], conflict["new_msg_uid"]],
        "conflict_resolved": False
    }
