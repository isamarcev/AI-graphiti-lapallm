# Tabula Rasa Agent - Knowledge-Centered Conversational AI

Проєкт AI агента з довготривалою пам'яттю для хакатону **Tabula Rasa: Agent Genesis**.

[English version below](#english-version)

---

## 🎯 Огляд проєкту

**Tabula Rasa Agent** - це агент з нульовими знаннями про предметну область, який навчається виключно від користувача через діалог.

### Ключові технології

- **Lapa LLM** - українська мовна модель на базі Gemma 12B
- **Graphiti** - темпоральний граф знань для зберігання пам'яті агента
- **LangGraph** - фреймворк для побудови агентів зі станом
- **Neo4j** - графова база даних для knowledge + message references
- **FastAPI** - REST API endpoints
- **Uv** - package manager

### Ключові можливості

✨ **Tabula Rasa** - агент починає з нульовими знаннями про домен  
🧠 **Графове зберігання** - зв'язки між сутностями та концептами  
🇺🇦 **Українська мова** - оптимізована модель для української  
🔍 **Гібридний пошук** - семантичний + BM25 + обхід графу  
⏱️ **Темпоральність** - відстеження часу подій  
🔗 **References** - кожна відповідь містить посилання на джерела  
🔄 **Auto-resolve** - автоматичне оновлення конфліктів (нова інформація замінює стару)  

---

## 🏗️ Архітектура

### Bidirectional Knowledge Flow

```
┌─────────────┐
│    USER     │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────┐
│   CLASSIFY INTENT               │
│   (TEACH or SOLVE)              │
└──────┬──────────────────┬───────┘
       │                  │
       │ TEACH            │ SOLVE
       ▼                  ▼
┌──────────────┐   ┌──────────────┐
│ Extract      │   │ Retrieve     │
│ Facts        │   │ Context      │
└──────┬───────┘   └──────┬───────┘
       │                  │
       ▼                  ▼
┌──────────────┐   ┌──────────────┐
│ Check        │   │ ReAct        │
│ Conflicts    │   │ Loop         │
└──────┬───────┘   └──────┬───────┘
       │                  │
       ▼                  ▼
┌──────────────┐   ┌──────────────┐
│ Auto-Resolve │   │ Generate     │
│ (new > old)  │   │ Answer       │
└──────┬───────┘   └──────┬───────┘
       │                  │
       ▼                  │
┌──────────────┐          │
│ Store to     │          │
│ Graphiti     │          │
└──────┬───────┘          │
       │                  │
       ▼                  ▼
    Response          Response
  + confirmation    + references
```

### Agent Nodes

**TEACH Path (агент навчається):**
1. `classify` - визначає intent (TEACH/SOLVE)
2. `extract_facts` - витягує structured facts з повідомлення
3. `check_conflicts` - детектує конфлікти з існуючими знаннями
4. `auto_resolve` - автоматично приймає нову інформацію
5. `generate_confirmation` - демонструє розуміння навченого
6. `store_knowledge` - зберігає в Graphiti + Neo4j

**SOLVE Path (агент відповідає):**
1. `classify` - визначає intent
2. `retrieve_context` - шукає релевантний контекст у пам'яті
3. `react_loop` - ітеративне міркування та пошук
4. `generate_answer` - генерує відповідь з обов'язковими references

---

## 📦 Вимоги

### Системні вимоги
- Docker та Docker Compose
- 16GB+ RAM
- 10GB+ вільного місця на диску

### Зовнішні сервіси
- **vLLM сервер** з Lapa LLM (запущений окремо)
- Опціонально: hosted embeddings API

---

## 🚀 Швидкий старт

### 1. Створити .env файл

```bash
make env
# або
cp env.example .env
```

### 2. Налаштувати .env

**КРИТИЧНО:** Встановіть URL до вашого vLLM сервера:

```bash
# Якщо vLLM запущений локально на host:
VLLM_BASE_URL=http://host.docker.internal:8000/v1

# Якщо vLLM на іншому сервері:
VLLM_BASE_URL=http://your-server-ip:8000/v1

# Інші важливі налаштування:
NEO4J_PASSWORD=password123
LOG_LEVEL=INFO
```

### 3. Запустити через Docker Compose

```bash
# Швидкий старт (все в одній команді)
make quick-start

# Або вручну:
docker-compose up -d --build
```

**Перший запуск займе 5-10 хвилин** (завантаження ML моделей в образ).

### 4. Перевірити

```bash
# Через Makefile
make health

# Або через curl
curl http://localhost:3000/health
```

**Очікувана відповідь:**
```json
{
  "status": "healthy",
  "timestamp": "2026-01-11T..."
}
```

---

## 📡 Використання API

### Endpoints

- **API Root**: http://localhost:3000/
- **Swagger UI**: http://localhost:3000/docs
- **Health**: http://localhost:3000/health
- **Text**: POST http://localhost:3000/text
- **Neo4j UI**: http://localhost:7474 (neo4j/password123)

### Приклади запитів

#### Навчання агента (TEACH)

```bash
curl -X POST http://localhost:3000/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Значення pi дорівнює 3.14",
    "user_id": "test-user"
  }'
```

**Відповідь:**
```json
{
  "response": "Зрозумів! Тепер я знаю що pi = 3.14. Це числова константа.\n\n✓ Навчання збережено.",
  "references": ["msg-001"]
}
```

#### Запит до агента (SOLVE)

```bash
curl -X POST http://localhost:3000/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Яке значення pi?",
    "user_id": "test-user"
  }'
```

**Відповідь:**
```json
{
  "response": "Значення pi дорівнює 3.14 [msg-001].",
  "references": ["msg-001"],
  "reasoning": "Крок 1: Пошук інформації про pi..."
}
```

#### Оновлення інформації (Auto-resolve)

```bash
# Спочатку навчаємо
curl -X POST http://localhost:3000/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Столиця України - Київ", "user_id": "test"}'

# Потім оновлюємо (для тесту)
curl -X POST http://localhost:3000/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Столиця України - Львів", "user_id": "test"}'
```

**Відповідь (автоматичне оновлення):**
```json
{
  "response": "✓ Інформацію оновлено\n\n**Було** (msg-001):\nКиїв є_столицею України\n\n**Тепер** (msg-002):\nЛьвів є_столицею України\n\nЯ оновив свої знання...",
  "references": ["msg-001", "msg-002"]
}
```

---

## 📁 Структура проєкту

```
graphity_lapa/
├── 🐳 Docker
│   ├── Dockerfile                    # Multi-stage build з ML моделями
│   ├── docker-compose.yml            # Agent + Neo4j
│   ├── .dockerignore
│   └── env.example
│
├── 🔧 Configuration
│   ├── config/
│   │   ├── settings.py               # Pydantic settings
│   │   └── logging_config.py
│   └── requirements.txt
│
├── 🤖 Agent
│   ├── agent/
│   │   ├── state.py                  # AgentState TypedDict
│   │   ├── graph.py                  # LangGraph topology
│   │   ├── helpers.py                # Utility functions
│   │   └── nodes/
│   │       ├── classify.py           # Intent classification
│   │       ├── extract.py            # Fact extraction
│   │       ├── conflicts.py          # Conflict detection
│   │       ├── auto_resolve.py       # Auto-accept new info
│   │       ├── resolve.py            # Manual resolution (backup)
│   │       ├── confirm.py            # Confirmation generation
│   │       ├── store.py              # Knowledge storage
│   │       ├── retrieve.py           # Context retrieval
│   │       ├── react.py              # ReAct reasoning
│   │       └── generate.py           # Answer generation
│
├── 🔌 Clients
│   ├── clients/
│   │   ├── llm_client.py             # LLM wrapper (Lapa)
│   │   ├── graphiti_client.py        # Graphiti с custom LLM
│   │   └── hosted_embedder.py        # Hosted embeddings
│
├── 🗄️ Database
│   ├── db/
│   │   └── neo4j_helpers.py          # Message references store
│
├── 🌐 API
│   ├── app.py                        # FastAPI application
│   └── routers/
│       ├── text.py                   # /text endpoint
│       └── schemas.py                # Request/Response models
│
└── 📚 Documentation
    ├── README.md                     # Цей файл
    ├── QUICKSTART.md                 # Швидкий старт (3 команди)
    ├── DOCKER_SETUP.md               # Детальний Docker setup
    ├── TABULA_RASA_IMPLEMENTATION.md # Імплементація
    └── TABULA_RASA_TESTING.md        # Testing guide
```

---

## 🛠️ Корисні команди

### Управління (через Makefile)

```bash
make up          # Запустити
make down        # Зупинити
make restart     # Перезапустити
make logs        # Дивитись логи
make ps          # Статус сервісів
```

### Тестування

```bash
make health         # Перевірка API
make test-teach     # Тест навчання
make test-solve     # Тест запиту
make check-vllm     # Перевірка vLLM
```

### Розробка

```bash
make dev-up         # Setup для розробки
make dev-rebuild    # Rebuild без кешу
make logs-agent     # Логи агента
make shell-agent    # Shell в контейнері
```

### UI

```bash
make docs           # Відкрити API docs
make neo4j-query    # Відкрити Neo4j Browser
```

**Повний список команд:** `make help`

---

## ⚙️ Конфігурація

### Environment Variables (.env)

```bash
# === LLM Configuration ===
VLLM_BASE_URL=http://host.docker.internal:8000/v1
VLLM_MODEL_NAME=lapa
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=2048

# === Neo4j ===
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password123

# === Embeddings ===
USE_HOSTED_EMBEDDINGS=true  # false для local
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# === Graphiti ===
GRAPHITI_SEARCH_LIMIT=10
GRAPHITI_RELEVANCE_THRESHOLD=0.7

# === Logging ===
LOG_LEVEL=INFO  # DEBUG для детального логування

# === LangSmith (optional) ===
LANGSMITH_API_KEY=your-key
LANGSMITH_TRACING_V2=false
```

---

## 🧪 Тестування

### Сценарій 1: Базове навчання

```bash
# 1. Навчити факт
curl -X POST http://localhost:3000/text \
  -H "Content-Type: application/json" \
  -d '{"text": "У системі Alpha команда має формат: дія об'\''єкт параметри", "user_id": "test"}'

# 2. Застосувати навчене
curl -X POST http://localhost:3000/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Створи команду для видалення файлу test.txt", "user_id": "test"}'

# Очікується: використання навченої структури з references
```

### Сценарій 2: Auto-resolve конфліктів

```bash
# 1. Навчити початкове значення
curl -X POST http://localhost:3000/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Значення pi дорівнює 3.14", "user_id": "test"}'

# 2. Оновити значення
curl -X POST http://localhost:3000/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Значення pi дорівнює 4", "user_id": "test"}'

# Очікується: автоматичне оновлення на нове значення
```

### Сценарій 3: Визнання gaps

```bash
# Запитати про щось ненавчене
curl -X POST http://localhost:3000/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Що таке Python?", "user_id": "test"}'

# Очікується: "На жаль, я не маю інформації про Python..."
```

**Детальний testing guide:** [TABULA_RASA_TESTING.md](TABULA_RASA_TESTING.md)

---

## 🔧 Troubleshooting

### Agent не стартує

```bash
make logs-agent  # Дивимось логи
make dev-rebuild  # Rebuild без кешу
```

### vLLM недоступний

```bash
make check-vllm  # Перевірка
# Виправте VLLM_BASE_URL в .env
make restart-agent
```

### Neo4j не стартує

```bash
make logs-neo4j  # Логи
sleep 30 && make ps  # Даємо більше часу (до 60 сек)
```

### Порт зайнятий

У `docker-compose.yml`:
```yaml
agent:
  ports:
    - "8080:3000"  # Використати інший порт
```

**Детальний troubleshooting:** [DOCKER_SETUP.md](docs/DOCKER_SETUP.mdUP.md)

---

## 📚 Документація

- **[QUICKSTART.md](QUICKSTART.md)** - швидкий старт за 3 команди
- **[DOCKER_SETUP.md](docs/DOCKER_SETUP.mdUP.md)** - детальний Docker setup
- **[DOCKER_DEPLOYMENT.md](docs/DOCKER_DEPLOYMENT.mdNT.md)** - production deployment
- **[TABULA_RASA_IMPLEMENTATION.md](TABULA_RASA_IMPLEMENTATION.md)** - імплементація змін
- **[TABULA_RASA_TESTING.md](TABULA_RASA_TESTING.md)** - testing scenarios

---

## 🎓 Корисні посилання

### Документація технологій
- [Graphiti GitHub](https://github.com/getzep/graphiti)
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Lapa LLM на HuggingFace](https://huggingface.co/lapa-llm)

### Хакатон
- [Tabula Rasa: Agent Genesis Task](https://www.notion.so/Tabula-Rasa-Agent-Genesis-Lapathon-Task-2dcb51a2f1a880e6a31ddcb7ecb84e00)

---

## ✨ Особливості Tabula Rasa Agent

### 1. Нульові знання про домен
- ✅ НЕ використовує pretrained knowledge
- ✅ Навчається виключно від користувача
- ✅ Універсальний (працює з будь-яким доменом)

### 2. Автоматичне оновлення конфліктів
- ✅ Нова інформація автоматично замінює стару (pi=4 > pi=3.14)
- ✅ Прозоре повідомлення про оновлення
- ✅ References до обох джерел (старе + нове)

### 3. Обов'язкові references
- ✅ Кожна відповідь містить посилання [msg-XXX]
- ✅ Точне мапування джерел (не індекси!)
- ✅ Верифікація використаних знань

### 4. Демонстрація розуміння
- ✅ Не просто "запам'ятав", а показує розуміння структури
- ✅ Різні підходи для різних типів контенту
- ✅ Визнає gaps у знаннях

### 5. Epistemic awareness
- ✅ Confidence scores для фактів
- ✅ Детекція 5 типів конфліктів
- ✅ Reasoning trace для accountability

---

## 🐛 Known Issues

1. **Reranker модель** - 2.27 GB, завантажується під час Docker build
   - Перший build займає ~5-10 хвилин
   - Модель включена в образ для швидкого старту

2. **Embeddings** - перший запит може бути повільним (~5-10 сек)
   - sentence-transformers ініціалізується при першому використанні
   - Наступні запити швидкі

3. **Neo4j startup** - може займати до 60 секунд
   - Healthcheck чекає поки Neo4j готовий
   - Docker Compose автоматично чекає через `depends_on`

---

## 🤝 Contributing

Цей проєкт створено для хакатону. Для покращень:

1. Fork репозиторій
2. Створіть feature branch
3. Commit зміни
4. Push в branch
5. Створіть Pull Request

---

## 📄 License

MIT License - див. LICENSE файл для деталей

---

## 👥 Authors

- Хакатон проєкт для **Tabula Rasa: Agent Genesis**
- Використані технології: Lapa LLM, Graphiti, LangGraph, Neo4j, FastAPI

---

## 🎉 Подяки

- Команді **Lapa LLM** за українську модель
- **Zep** за фреймворк Graphiti
- **LangChain** за LangGraph
- Організаторам **Lapathon** за мотивацію!

---

# English Version

## 🎯 Project Overview

**Tabula Rasa Agent** is a knowledge-centered conversational AI agent that starts with zero knowledge about any domain and learns exclusively from user interactions.

### Key Features

✨ **Tabula Rasa** - agent starts with zero domain knowledge  
🧠 **Graph Memory** - stores knowledge in temporal knowledge graph  
🇺🇦 **Ukrainian Language** - optimized for Ukrainian using Lapa LLM  
🔍 **Hybrid Search** - semantic + BM25 + graph traversal  
⏱️ **Temporal** - tracks time of events and facts  
🔗 **References** - every response includes source citations  
🔄 **Auto-resolve** - automatically updates conflicting information  

### Technologies

- **Lapa LLM** - Ukrainian language model (Gemma 12B based)
- **Graphiti** - Temporal knowledge graph
- **LangGraph** - Agent orchestration framework
- **Neo4j** - Graph database
- **FastAPI** - REST API

---

## 🚀 Quick Start

### 1. Setup environment

```bash
make env
nano .env  # Set VLLM_BASE_URL
```

### 2. Start with Docker

```bash
make quick-start
```

### 3. Test API

```bash
curl http://localhost:3000/health
```

**Documentation:**
- Swagger UI: http://localhost:3000/docs
- Neo4j Browser: http://localhost:7474

---

## 📡 API Examples

### Teach the agent

```bash
curl -X POST http://localhost:3000/text \
  -H "Content-Type: application/json" \
  -d '{"text": "The value of pi equals 3.14", "user_id": "test"}'
```

### Query the agent

```bash
curl -X POST http://localhost:3000/text \
  -H "Content-Type: application/json" \
  -d '{"text": "What is the value of pi?", "user_id": "test"}'
```

Response includes `references` with source message UIDs.

---

## 📚 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - 3-command quick start
- **[DOCKER_SETUP.md](docs/DOCKER_SETUP.mdUP.md)** - detailed Docker setup
- **[TABULA_RASA_TESTING.md](TABULA_RASA_TESTING.md)** - testing scenarios

---

## 🛠️ Commands

```bash
make help          # Show all commands
make up            # Start services
make logs          # View logs
make test-teach    # Test teaching
make test-solve    # Test querying
```

---

## ⚙️ Configuration

Edit `.env` file:

```bash
VLLM_BASE_URL=http://host.docker.internal:8000/v1
NEO4J_PASSWORD=password123
LOG_LEVEL=INFO
```

---

## 🎯 Key Principles

### Tabula Rasa
- No pretrained domain knowledge
- Learns only from user
- Universal (works with any domain)

### Auto-resolve Conflicts
- New information automatically replaces old
- pi=4 replaces pi=3.14
- Transparent notification with references

### Mandatory References
- Every response cites sources [msg-XXX]
- Exact source mapping (not indices)
- Verification of used knowledge

---

## 📄 License

MIT License

---

**Ready to start!** 🚀

```bash
make quick-start
```
