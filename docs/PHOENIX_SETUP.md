# Phoenix Observability Setup

Phoenix - це інструмент для observability та трейсингу LLM застосунків. Він дозволяє моніторити виклики LLM, embeddings, token usage та cost tracking.

## 🚀 Швидкий старт

### 1. Встановлення залежностей

```bash
pip install -r requirements.txt
```

Це встановить:
- `arize-phoenix` - основний Phoenix server та UI
- `arize-phoenix-otel` - OpenTelemetry інтеграція
- `openinference-instrumentation-langchain` - автоматична інструментація LangChain
- `openinference-instrumentation-openai` - автоматична інструментація OpenAI API

### 2. Запуск з Docker Compose

Phoenix автоматично запускається разом з іншими сервісами:

```bash
docker-compose up -d
```

Це запустить:
- **Phoenix UI** на `http://localhost:6006`
- **Agent API** на `http://localhost:3000`
- **Neo4j** на `http://localhost:7474`

### 3. Налаштування через `.env`

Додайте до вашого `.env` файлу:

```bash
# Phoenix Observability
ENABLE_PHOENIX=true
PHOENIX_COLLECTOR_ENDPOINT=http://phoenix:6006
PHOENIX_PROJECT_NAME=graphiti-lapa-agent
```

**Параметри:**
- `ENABLE_PHOENIX` - увімкнути/вимкнути Phoenix (true/false)
- `PHOENIX_COLLECTOR_ENDPOINT` - URL Phoenix collector (в Docker: `http://phoenix:6006`, локально: `http://localhost:6006`)
- `PHOENIX_PROJECT_NAME` - назва проекту для групування трейсів

## 📊 Що відслідковується

### Автоматична інструментація

Phoenix автоматично відслідковує:

1. **LLM виклики** (через OpenAI-compatible API)
   - Input/output messages
   - Model name, temperature, max_tokens
   - Token usage (prompt, completion, total)
   - Latency та timing

2. **LangGraph nodes**
   - Виконання кожної ноди агента
   - State transitions
   - Час виконання

3. **LangChain operations**
   - Chains, agents, tools
   - Retrieval operations

### Cost Tracking

Автоматично обчислюється вартість кожного LLM виклику:

```python
# В llm_client.py додано автоматичний cost tracking
from config.cost_tracking import add_cost_to_span

add_cost_to_span(
    model="lapa",
    prompt_tokens=response.usage.prompt_tokens,
    completion_tokens=response.usage.completion_tokens,
    is_embedding=False
)
```

**Pricing налаштування** в `config/cost_tracking.py`:

```python
TOKEN_PRICES = {
    "lapa": {"input": 0.0, "output": 0.0},  # Безкоштовно для хакатону
    "gpt-4": {"input": 30.0, "output": 60.0},
    # ... інші моделі
}
```

## 🔍 Використання Phoenix UI

### Відкрийте Phoenix UI

Перейдіть на `http://localhost:6006` після запуску Docker Compose.

### Основні можливості:

1. **Traces View**
   - Перегляд всіх LLM викликів
   - Деталізація кожного request/response
   - Token usage та cost breakdown

2. **Projects**
   - Групування трейсів по проектах
   - Фільтрація за часом, моделлю, користувачем

3. **Evaluations**
   - Аналіз якості відповідей
   - Порівняння різних промптів

4. **Cost Analysis**
   - Cumulative cost по моделях
   - Cost breakdown по операціях
   - Trending та прогнозування

## 🛠️ Розширені можливості

### Ручна інструментація

Додайте custom spans для специфічних операцій:

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("custom_operation") as span:
    # Ваш код
    span.set_attribute("custom_attribute", "value")
    result = do_something()
    span.set_attribute("result_count", len(result))
```

### Додавання metadata до spans

```python
from config.phoenix_config import add_phoenix_metadata

add_phoenix_metadata(
    span_name="knowledge_extraction",
    metadata={
        "user_id": "user123",
        "message_type": "query",
        "knowledge_count": 5
    }
)
```

### Оновлення pricing

```python
from config.cost_tracking import update_token_prices

# Оновити ціни для нової моделі
update_token_prices(
    model="new-model",
    input_price=1.0,   # $ за 1M токенів
    output_price=2.0
)
```

## 🐛 Troubleshooting

### Phoenix не показує трейси

1. **Перевірте, чи запущений Phoenix:**
   ```bash
   docker-compose ps phoenix
   ```

2. **Перевірте логи Phoenix:**
   ```bash
   docker-compose logs phoenix
   ```

3. **Перевірте змінні оточення:**
   ```bash
   echo $ENABLE_PHOENIX
   echo $PHOENIX_COLLECTOR_ENDPOINT
   ```

### Вимкнення Phoenix для development

Якщо Phoenix сповільнює development, вимкніть його:

```bash
# В .env
ENABLE_PHOENIX=false
```

Або запустіть тільки необхідні сервіси:

```bash
docker-compose up neo4j agent
```

### Очищення даних Phoenix

Phoenix зберігає дані у volume. Для очищення:

```bash
docker-compose down -v  # Видалить всі volumes
docker-compose up -d     # Перезапустить з чистими даними
```

## 📈 Best Practices

1. **Використовуйте різні project names** для різних оточень:
   ```bash
   # Development
   PHOENIX_PROJECT_NAME=graphiti-lapa-dev
   
   # Production
   PHOENIX_PROJECT_NAME=graphiti-lapa-prod
   ```

2. **Додавайте користувацькі атрибути** для кращої фільтрації:
   ```python
   span.set_attribute("user_id", user_id)
   span.set_attribute("conversation_id", conv_id)
   span.set_attribute("feature", "knowledge_extraction")
   ```

3. **Регулярно переглядайте cost reports** для оптимізації витрат

4. **Використовуйте Phoenix для debugging** складних multi-step flows

## 🔗 Корисні посилання

- [Phoenix Documentation](https://docs.arize.com/phoenix)
- [OpenTelemetry Python](https://opentelemetry.io/docs/instrumentation/python/)
- [OpenInference Spec](https://github.com/Arize-ai/openinference)

## 💡 Приклади використання

### Моніторинг конкретного користувача

1. Відкрийте Phoenix UI
2. Перейдіть у "Traces"
3. Відфільтруйте по `user_id` attribute
4. Перегляньте всі LLM виклики цього користувача

### Аналіз повільних запитів

1. У Phoenix UI відсортуйте трейси по latency
2. Знайдіть найповільніші операції
3. Перегляньте span details для розуміння bottleneck'ів
4. Оптимізуйте промпти або паралелізуйте виклики

### Cost optimization

1. Перегляньте "Cost Analysis" в Phoenix
2. Знайдіть операції з найбільшою вартістю
3. Оптимізуйте довжину промптів або використовуйте кешування
4. Розгляньте використання дешевших моделей для простих задач
