# LangSmith Трейсинг

## Що це дає?

LangSmith показує в UI:
- Кожен виклик LLM (запит/відповідь, токени, час)
- Всі кроки агента (retrieve → generate → save)
- Помилки та винятки
- Граф виконання LangGraph

## Налаштування

1. **Зареєструйся на LangSmith:**
   - https://smith.langchain.com/
   - Отримай API ключ

2. **Додай в `.env`:**
   ```bash
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=your_api_key_here
   LANGCHAIN_PROJECT=graphiti-lapa-demo
   ```

3. **Встанови залежність:**
   ```bash
   pip install langsmith
   ```

4. **Додай в код (початок скрипта/ноутбука):**
   ```python
   from utils.langsmith_setup import setup_langsmith

   # Ініціалізація LangSmith
   setup_langsmith()
   ```

5. **Запусти агента** - треки автоматично з'являться в LangSmith UI

## Перегляд треків

1. Відкрий https://smith.langchain.com/
2. Вибери проект `graphiti-lapa-demo`
3. Побачиш всі виклики агента з деталями:
   - ⏱️ Час виконання кожного кроку
   - 🔍 Вхід/вихід кожної функції
   - 💬 Всі повідомлення до/від LLM
   - ❌ Помилки з повним traceback

## Приклад треку

```
Agent Run
├─ retrieve_memory (0.5s)
│  └─ graphiti.search()
├─ generate_response (2.3s)
│  └─ vllm.chat.completions (2.1s)
│     ├─ Input: [system, user messages]
│     └─ Output: "Привіт, Олександре..."
└─ save_to_memory (1.2s)
   └─ graphiti.add_episode()
      ├─ extract_entities()
      └─ create_embeddings()
```

## Вимкнення

Постав `LANGCHAIN_TRACING_V2=false` в `.env` або видали рядок.