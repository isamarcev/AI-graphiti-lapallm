"""
Простий тест з'єднання з LLM клієнтом.
Відправка базового повідомлення без структурованої схеми.
"""

import asyncio
from clients.llm_client import get_llm_client


async def test_connection():
    """Тестова функція для перевірки з'єднання з LLM."""
    print("🔌 Підключення до LLM клієнта...")

    # Отримуємо клієнт
    llm_client = get_llm_client()
    print(f"✅ Клієнт створено")
    print(f"   URL: {llm_client.base_url}")
    print(f"   Модель: {llm_client.model_name}")

    # Простий тестовий запит
    print("\n📤 Відправка тестового повідомлення...")

    try:
        response = await llm_client.generate_async(
            messages=[
                {"role": "system", "content": "Ти корисний асистент."},
                {"role": "user", "content": "Привіт! Як справи?"}
            ],
            temperature=0.0
        )

        print("✅ Відповідь отримано:")
        print(f"\n{response}\n")

    except Exception as e:
        print(f"❌ Помилка при виклику LLM: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(test_connection())