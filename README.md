# Graphiti + LangGraph + Lapa LLM Demo

Демонстрационный проект AI агента с долговременной памятью для хакатона **Tabula Rasa: Agent Genesis**.

## 🎯 Обзор проекта

Этот проект демонстрирует интеграцию трех ключевых технологий:

- **Lapa LLM** - украинская языковая модель на базе Gemma 12B
- **Graphiti** - временно-ориентированный граф знаний для хранения памяти агента
- **LangGraph** - фреймворк для построения агентов с состоянием

### Ключевые возможности

✨ **Долговременная память** - агент запоминает факты из предыдущих разговоров
🧠 **Графовое хранение** - связи между сущностями и концепциями
🇺🇦 **Украинский язык** - оптимизированная модель для украинского
🔍 **Гибридный поиск** - семантический + BM25 + обход графа
⏱️ **Темпоральность** - отслеживание времени событий

---

## 🏗️ Архитектура

```
User Input → LangGraph Agent → Graphiti Memory → Neo4j
                ↓                     ↑
            Lapa LLM (vLLM)          |
                ↓                     |
            Response ←───────────────┘
```

### Поток обработки:
1. **retrieve_memory** - поиск релевантного контекста в графе
2. **generate_response** - генерация ответа с учетом контекста
3. **save_to_memory** - сохранение нового эпизода в граф

---

## 📦 Требования

### Системные требования
- Python 3.10+
- Docker и Docker Compose
- 16GB+ RAM (для работы с 12B моделью)
- GPU рекомендуется (но не обязателен)

### Сервисы
- Neo4j 5.26+
- vLLM или Ollama для запуска Lapa LLM

---

## 🚀 Быстрый старт

### 1. Клонирование и установка зависимостей

```bash
# Перейдите в директорию проекта
cd /path/to/llm/graphity_lapa

# Создайте виртуальное окружение
python3 -m venv venv
source venv/bin/activate  # На Windows: venv\Scripts\activate

# Установите зависимости
pip install -r requirements.txt
```

### 2. Настройка окружения

```bash
# Скопируйте шаблон конфигурации
cp .env.example .env

# Отредактируйте .env файл при необходимости
# По умолчанию настроено для локального запуска
```

### 3. Запуск Neo4j

```bash
# Запустите Neo4j через Docker Compose
docker-compose up -d

# Проверьте статус
docker-compose ps

# Neo4j Web UI доступен по адресу: http://localhost:7474
# Логин: neo4j, пароль: password123
```

### 4. Запуск vLLM с Lapa LLM

#### Опция A: vLLM локально (требует GPU)

```bash
# Установите vLLM
pip install vllm

# Запустите сервер с Lapa LLM
vllm serve lapa-llm/lapa-v0.1.2-instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 4096
```

#### Опция B: vLLM через Docker

```bash
docker run --gpus all \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model lapa-llm/lapa-v0.1.2-instruct
```

#### Опция C: Ollama (проще для CPU)

```bash
# Установите Ollama: https://ollama.ai
ollama pull lapa-llm/lapa-v0.1.2-instruct

# Запустите сервер
ollama serve
```

### 5. Запуск demo

```bash
# Откройте Jupyter Notebook
jupyter notebook demo_flow.ipynb

# Или запустите Jupyter Lab
jupyter lab demo_flow.ipynb
```

---

## 📁 Структура проекта

```
llm/graphity_lapa/
├── docker-compose.yml          # Neo4j setup
├── requirements.txt            # Python зависимости
├── .env.example               # Шаблон конфигурации
├── README.md                  # Эта документация
├── config/
│   ├── __init__.py
│   └── settings.py            # Настройки (LLM, DB, Graphiti)
├── clients/
│   ├── __init__.py
│   ├── llm_client.py          # Wrapper для vLLM/OpenAI API
│   └── graphiti_client.py     # Graphiti с кастомным LLM
├── agent/
│   ├── __init__.py
│   ├── state.py               # Определение State для LangGraph
│   ├── nodes.py               # Узлы: retrieve, generate, save
│   └── graph.py               # Сборка LangGraph
├── models/
│   ├── __init__.py
│   └── schemas.py             # Pydantic модели
└── demo_flow.ipynb            # Демонстрационный notebook
```

---

## ⚙️ Конфигурация

Все настройки находятся в файле `.env`. Основные параметры:

### LLM Configuration
```env
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_MODEL_NAME=lapa-llm/lapa-v0.1.2-instruct
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=2048
```

### Neo4j Configuration
```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password123
```

### Graphiti Configuration
```env
GRAPHITI_MAX_EPISODE_LENGTH=10000
GRAPHITI_SEARCH_LIMIT=10
GRAPHITI_RELEVANCE_THRESHOLD=0.7
```

### Embeddings (для украинского языка)
```env
EMBEDDING_MODEL_NAME=sentence-transformers/paraphrase-multilingual-mpnet-base-v2
```

---

## 🧪 Тестирование компонентов

### Проверка vLLM
```bash
curl http://localhost:8000/v1/models
curl http://localhost:8000/health
```

### Проверка Neo4j
Откройте http://localhost:7474 и выполните:
```cypher
MATCH (n) RETURN n LIMIT 25
```

### Проверка Python компонентов
```python
from clients.llm_client import get_llm_client
from clients.graphiti_client import get_graphiti_client

# Тест LLM
llm = get_llm_client()
response = await llm.generate_async([
    {"role": "user", "content": "Привіт!"}
])
print(response)

# Тест Graphiti
graphiti = await get_graphiti_client()
stats = await graphiti.get_graph_stats()
print(stats)
```

---

## 💡 Примеры использования

### Базовый диалог

```python
from langchain_core.messages import HumanMessage
from agent.graph import get_agent_app

agent = get_agent_app()

# Первое сообщение
result = await agent.ainvoke({
    "messages": [HumanMessage(content="Привіт! Мене звати Олег.")],
    "user_id": "user_1",
    "session_id": "session_1",
    # ... остальные поля state
})

print(result["messages"][-1].content)
```

### Поиск в памяти

```python
from clients.graphiti_client import get_graphiti_client

graphiti = await get_graphiti_client()

# Поиск информации
results = await graphiti.search("Олег", limit=5)
for result in results:
    print(result['content'])
```

### Визуализация графа

Откройте Neo4j Browser (http://localhost:7474) и выполните:

```cypher
// Все узлы и связи
MATCH (n)-[r]->(m)
RETURN n, r, m
LIMIT 100

// Узлы определенного типа
MATCH (n:Entity)
WHERE n.name CONTAINS 'Олег'
RETURN n

// Связи между сущностями
MATCH (a:Entity)-[r:RELATIONSHIP]->(b:Entity)
RETURN a.name, type(r), b.name
```

---

## 🔧 Troubleshooting

### Проблема: vLLM не запускается

**Решение:**
- Убедитесь, что модель скачана: `huggingface-cli download lapa-llm/lapa-v0.1.2-instruct`
- Проверьте доступную память: `nvidia-smi` (для GPU)
- Используйте quantized версию для меньшего объема памяти

### Проблема: Neo4j connection refused

**Решение:**
```bash
# Проверьте статус контейнера
docker-compose ps

# Перезапустите Neo4j
docker-compose restart neo4j

# Проверьте логи
docker-compose logs neo4j
```

### Проблема: Graphiti не создает индексы

**Решение:**
```python
# Вручную создайте индексы
graphiti = await get_graphiti_client()
await graphiti.graphiti.build_indices()
```

### Проблема: LLM не возвращает structured output

**Решение:**
- Убедитесь, что vLLM запущен с поддержкой JSON mode
- Проверьте версию vLLM: `pip show vllm` (требуется 0.8.5+)
- Используйте fallback на OpenAI API для тестирования: `USE_OPENAI_FALLBACK=true`

---

## 📚 Документация API

### LLM Client

```python
from clients.llm_client import LLMClient

client = LLMClient(
    base_url="http://localhost:8000/v1",
    model_name="lapa-llm/lapa-v0.1.2-instruct"
)

# Async generation
response = await client.generate_async(
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7,
    max_tokens=1024
)

# Structured output
from models.schemas import AgentResponse
response = await client.generate_async(
    messages=messages,
    response_format=AgentResponse
)
```

### Graphiti Client

```python
from clients.graphiti_client import GraphitiClient

async with GraphitiClient() as graphiti:
    # Добавить эпизод
    await graphiti.add_episode(
        episode_body="User said hello",
        episode_name="episode_1",
        source_description="user_1"
    )

    # Поиск
    results = await graphiti.search(
        query="hello",
        limit=10
    )

    # Статистика
    stats = await graphiti.get_graph_stats()
```

---

## 🎓 Полезные ссылки

### Документация
- [Graphiti GitHub](https://github.com/getzep/graphiti)
- [LangGraph Docs](https://docs.langchain.com/oss/python/langgraph/)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Lapa LLM на HuggingFace](https://huggingface.co/lapa-llm/lapa-v0.1.2-instruct)

### Хакатон
- [Tabula Rasa: Agent Genesis Task](https://www.notion.so/Tabula-Rasa-Agent-Genesis-Lapathon-Task-2dcb51a2f1a880e6a31ddcb7ecb84e00)

---

## 🐛 Known Issues

1. **agent/nodes.py** - функция `save_to_memory_node` неполная (строка 203+)
   - Нужно дополнить код сохранения эпизода в Graphiti
   - См. комментарии в файле

2. **Embeddings** - первый запуск может быть медленным
   - sentence-transformers скачивает модель при первом использовании
   - ~400MB для paraphrase-multilingual-mpnet-base-v2

3. **Memory usage** - 12B модель требует значительно памяти
   - Минимум 16GB RAM для CPU inference
   - Минимум 12GB VRAM для GPU inference

---

## 🤝 Contributing

Этот проект создан для хакатона. Для улучшений:

1. Fork репозиторий
2. Создайте feature branch
3. Commit изменения
4. Push в branch
5. Создайте Pull Request

---

## 📄 License

MIT License - см. LICENSE файл для деталей

---

## 👥 Authors

- Хакатон проект для **Tabula Rasa: Agent Genesis**
- Используемые технологии: Lapa LLM, Graphiti, LangGraph, Neo4j

---

## 🎉 Благодарности

- Команде **Lapa LLM** за украинскую модель
- **Zep** за фреймворк Graphiti
- **LangChain** за LangGraph
- Организаторам **Lapathon** за мотивацию!
