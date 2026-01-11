# Testing Guide - Tabula Rasa Agent

## Швидкий старт

### 1. Запустити сервіси

```bash
# Neo4j
docker-compose up -d

# vLLM (якщо локально) або використовуйте hosted API
# Вже налаштовано в .env
```

### 2. Запустити FastAPI

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8080
```

### 3. Перевірити API

```bash
curl http://localhost:8080/
curl http://localhost:8080/health
```

## Тестові сценарії

### Сценарій 1: TEACH flow (навчання)

```bash
curl -X POST http://localhost:8080/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Київ - столиця України",
    "uid": "test-msg-001",
    "user_id": "test_user"
  }'
```

**Очікуваний результат:**
- intent: "teach"
- extracted_facts: [{"subject": "Київ", "relation": "є_столицею", "object": "України"}]
- response: "✓ Навчання збережено. Вивчено 1 фактів..."
- references: ["test-msg-001"]

### Сценарій 2: SOLVE flow (запит)

```bash
curl -X POST http://localhost:8080/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Яка столиця України?",
    "uid": "test-msg-002",
    "user_id": "test_user"
  }'
```

**Очікуваний результат:**
- intent: "solve"
- retrieved_context: [...facts про Київ...]
- response: "Київ [Джерело 0]"
- references: ["test-msg-001", "test-msg-002"]

### Сценарій 3: Conflict detection

```bash
# Спочатку навчити факт
curl -X POST http://localhost:8080/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Київ - столиця України",
    "uid": "test-msg-003",
    "user_id": "test_user"
  }'

# Потім дати суперечливий факт
curl -X POST http://localhost:8080/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Харків - столиця України",
    "uid": "test-msg-004",
    "user_id": "test_user"
  }'
```

**Очікуваний результат:**
- conflicts: [{old_msg_uid: "test-msg-003", new_msg_uid: "test-msg-004", ...}]
- response: "⚠️ Виявлено протиріччя:..."
- Agent чекає на резолюцію від користувача

### Сценарій 4: ReAct reasoning

```bash
curl -X POST http://localhost:8080/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Розкажи про Київ та його історію",
    "uid": "test-msg-005",
    "user_id": "test_user"
  }'
```

**Очікуваний результат:**
- react_steps: [{thought: "...", action: "search", observation: "..."}, ...]
- reasoning: "Крок 1: ... Крок 2: ..."
- response з references до джерел

## Перевірка в Neo4j

### Переглянути Message nodes

```bash
# Відкрити Neo4j Browser: http://localhost:7474
# Login: neo4j / password123
```

```cypher
// Всі messages
MATCH (m:Message)
RETURN m
ORDER BY m.timestamp DESC
LIMIT 10

// Messages з episodes
MATCH (m:Message)-[:GENERATED_EPISODE]->(e:Episode)
RETURN m.uid, m.text, e.name

// Messages конкретного користувача
MATCH (m:Message {user_id: "test_user"})
RETURN m.uid, m.text, m.timestamp
ORDER BY m.timestamp DESC
```

## Debug logging

Для детального логування:

```bash
# В .env
LOG_LEVEL=DEBUG
DEBUG_MODE=true

# Перезапустити app
uvicorn app:app --reload --log-level debug
```

## LangSmith tracing (опційно)

```bash
# В .env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key
LANGCHAIN_PROJECT=graphiti-lapa-demo

# View traces: https://smith.langchain.com/
```

## Troubleshooting

### Error: "Graphiti not initialized"

```python
# Manually initialize
from clients.graphiti_client import get_graphiti_client
graphiti = await get_graphiti_client()
await graphiti.graphiti.build_indices_and_constraints()
```

### Error: "Neo4j connection refused"

```bash
docker-compose ps
docker-compose logs neo4j
docker-compose restart neo4j
```

### Error: "LLM structured output failed"

- Перевірте що vLLM/hosted API підтримує JSON mode
- Або use OpenAI fallback: `USE_OPENAI_FALLBACK=true`

## Приклад Python тесту

```python
import requests
import uuid

def test_teach_solve_flow():
    base_url = "http://localhost:8080"
    
    # 1. Teach
    teach_response = requests.post(f"{base_url}/text", json={
        "text": "Київ - столиця України",
        "uid": str(uuid.uuid4()),
        "user_id": "test_user"
    })
    assert teach_response.status_code == 200
    assert "збережено" in teach_response.json()["response"]
    
    # 2. Solve
    solve_response = requests.post(f"{base_url}/text", json={
        "text": "Яка столиця України?",
        "uid": str(uuid.uuid4()),
        "user_id": "test_user"
    })
    assert solve_response.status_code == 200
    assert "Київ" in solve_response.json()["response"]
    assert len(solve_response.json()["references"]) > 0
    
    print("✅ Test passed!")

if __name__ == "__main__":
    test_teach_solve_flow()
```

## Metrics to track

1. **Classification accuracy** - чи правильно classify teach vs solve
2. **Conflict detection rate** - скільки conflicts виявлено
3. **Reference completeness** - чи всі відповіді мають references
4. **ReAct iterations** - середня кількість кроків
5. **Response latency** - час обробки запиту

## Next steps

1. Протестувати всі 4 сценарії
2. Перевірити Neo4j Message nodes
3. Подивитись LangSmith traces (якщо enabled)
4. Налаштувати conflict resolution endpoint (майбутнє)
5. Додати metrics/monitoring
