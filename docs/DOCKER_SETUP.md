# 🐳 Docker Setup - Tabula Rasa Agent

## 📦 Що Включено

Docker Compose запускає два сервіси:
1. **agent** - Tabula Rasa Agent (FastAPI API на порту 3000)
2. **neo4j** - Neo4j Graph Database (Web UI на 7474, Bolt на 7687)

## 🚀 Швидкий Старт

### 1. Налаштування Environment Variables

Створіть `.env` файл з налаштуваннями:

```bash
cp .env.example .env
```

Відредагуйте `.env` та встановіть:

**КРИТИЧНО:** URL до вашого vLLM сервера з Lapa LLM:
```bash
# Якщо vLLM запущений локально на host machine:
VLLM_BASE_URL=http://host.docker.internal:8000/v1

# Якщо vLLM на іншому сервері:
VLLM_BASE_URL=http://your-vllm-server:8000/v1
```

### 2. Запуск

```bash
# Збілдити та запустити всі сервіси
docker-compose up --build

# Або в background режимі
docker-compose up -d --build
```

**Перший запуск займе ~2-5 хвилин:**
- Завантаження Docker images
- Встановлення Python залежностей
- Ініціалізація Neo4j
- Завантаження embedding моделі

### 3. Перевірка

Відкрийте в браузері:

- **API Docs**: http://localhost:3000/docs
- **Health Check**: http://localhost:3000/health
- **Neo4j Browser**: http://localhost:7474 (login: neo4j/password123)

## 📡 Використання API

### Test через curl

```bash
# Health check
curl http://localhost:3000/health

# Навчання агента (TEACH)
curl -X POST http://localhost:3000/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Значення pi дорівнює 3.14",
    "user_id": "test-user"
  }'

# Запит до агента (SOLVE)
curl -X POST http://localhost:3000/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Яке значення pi?",
    "user_id": "test-user"
  }'
```

### Test через Swagger UI

Відкрийте http://localhost:3000/docs та використовуйте інтерактивний UI.

## 🔧 Конфігурація

### Environment Variables

Основні змінні в `.env`:

```bash
# === Обов'язкові ===
VLLM_BASE_URL=http://host.docker.internal:8000/v1  # URL до vLLM
NEO4J_PASSWORD=password123                           # Neo4j пароль

# === Опціональні ===
LLM_TEMPERATURE=0.7                # Креативність відповідей
GRAPHITI_SEARCH_LIMIT=10           # Кількість results з пам'яті
LOG_LEVEL=INFO                     # DEBUG для детального логування
```

### Порти

За замовчуванням:
- **3000** - API агента
- **7474** - Neo4j Web UI
- **7687** - Neo4j Bolt

Змінити можна в `docker-compose.yml`:

```yaml
agent:
  ports:
    - "8080:3000"  # Зовнішній порт 8080, внутрішній 3000
```

## 🛠️ Корисні Команди

### Управління

```bash
# Запуск
docker-compose up -d

# Зупинка
docker-compose down

# Зупинка + видалення volumes (ОЧИСТИТЬ БД!)
docker-compose down -v

# Перезапуск одного сервісу
docker-compose restart agent

# Переглянути логи
docker-compose logs -f agent
docker-compose logs -f neo4j

# Переглянути статус
docker-compose ps
```

### Розробка

```bash
# Rebuild після змін коду
docker-compose up --build agent

# Увійти в контейнер
docker-compose exec agent bash

# Виконати команду в контейнері
docker-compose exec agent python -c "from agent.graph import create_agent_graph; print('OK')"
```

### Очистка

```bash
# Видалити всі контейнери
docker-compose down

# Видалити всі контейнери + volumes (ВИДАЛИТЬ ВСІ ДАНІ!)
docker-compose down -v

# Видалити невикористовувані images
docker image prune -a
```

## 🐛 Troubleshooting

### 1. Agent не може підключитись до vLLM

**Проблема:** 
```
Error: Connection refused to vLLM server
```

**Рішення:**

a) Перевірте що vLLM запущений:
```bash
curl http://localhost:8000/v1/models
```

b) У `.env` використовуйте `host.docker.internal`:
```bash
VLLM_BASE_URL=http://host.docker.internal:8000/v1
```

c) Якщо vLLM в Docker, додайте його в `docker-compose.yml` network:
```yaml
services:
  vllm:
    # ... your vLLM config
    networks:
      - graphiti-network
  
  agent:
    environment:
      - VLLM_BASE_URL=http://vllm:8000/v1
```

---

### 2. Neo4j не стартує

**Проблема:**
```
neo4j healthcheck failed
```

**Рішення:**

a) Збільште timeout:
```bash
docker-compose up -d neo4j
# Почекайте 30-60 секунд
docker-compose logs neo4j
```

b) Видаліть старі volumes:
```bash
docker-compose down -v
docker-compose up -d
```

---

### 3. Agent падає при старті

**Проблема:**
```
ModuleNotFoundError or import errors
```

**Рішення:**

a) Rebuild з очисткою кешу:
```bash
docker-compose build --no-cache agent
docker-compose up agent
```

b) Перевірте requirements.txt:
```bash
docker-compose run --rm agent pip list
```

---

### 4. Повільна відповідь API

**Причини:**
- Перший запит завантажує embedding модель (~5-10 сек)
- vLLM генерація може займати час
- Neo4j ініціалізується при першому запиті

**Рішення:**
- Збільште `LLM_TIMEOUT` в `.env`
- Використовуйте `--workers 2` в Dockerfile CMD (більше workers)
- Додайте GPU для vLLM

---

### 5. Порт 3000 зайнятий

**Рішення:**

Змініть порт в `docker-compose.yml`:
```yaml
agent:
  ports:
    - "8080:3000"  # Використовуйте 8080 замість 3000
```

Або знайдіть процес:
```bash
# Linux/Mac
lsof -i :3000
kill -9 <PID>

# Windows
netstat -ano | findstr :3000
taskkill /PID <PID> /F
```

---

## 📊 Моніторинг

### Логи в реальному часі

```bash
# Всі сервіси
docker-compose logs -f

# Тільки agent
docker-compose logs -f agent

# З фільтром
docker-compose logs -f agent | grep ERROR
```

### Метрики

```bash
# Використання ресурсів
docker stats

# Перевірка здоров'я
docker-compose ps
curl http://localhost:3000/health
```

### Neo4j Monitoring

Відкрийте Neo4j Browser: http://localhost:7474

```cypher
// Кількість збережених повідомлень
MATCH (m:Message) RETURN count(m)

// Кількість епізодів
MATCH (e:Episode) RETURN count(e)

// Останні 10 повідомлень
MATCH (m:Message) 
RETURN m.uid, m.text, m.timestamp 
ORDER BY m.timestamp DESC 
LIMIT 10
```

---

## 🔐 Безпека (Production)

### 1. Змініть паролі

У `.env`:
```bash
NEO4J_PASSWORD=your-strong-password-here
```

У `docker-compose.yml`:
```yaml
neo4j:
  environment:
    - NEO4J_AUTH=neo4j/your-strong-password-here
```

### 2. Обмежте CORS

У `app.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-domain.com"],  # Конкретні домени
    allow_credentials=True,
    allow_methods=["POST", "GET"],  # Тільки потрібні методи
    allow_headers=["*"],
)
```

### 3. Додайте HTTPS

Використовуйте nginx reverse proxy або Traefik.

### 4. Обмежте ресурси

У `docker-compose.yml`:
```yaml
agent:
  deploy:
    resources:
      limits:
        cpus: '2'
        memory: 4G
```

---

## 📦 Production Deployment

### З Docker Hub

1. Build та push:
```bash
docker build -t your-username/tabula-rasa-agent:latest .
docker push your-username/tabula-rasa-agent:latest
```

2. На сервері:
```bash
docker pull your-username/tabula-rasa-agent:latest
docker-compose up -d
```

### З Docker Swarm або Kubernetes

Конвертуйте `docker-compose.yml` за допомогою:
- Kompose (для Kubernetes)
- Docker Stack (для Swarm)

---

## ✅ Готово!

Ваш Tabula Rasa Agent тепер доступний на **http://localhost:3000** 🚀

**Перевірте:**
- ✅ http://localhost:3000/health
- ✅ http://localhost:3000/docs
- ✅ http://localhost:7474 (Neo4j UI)

**Протестуйте:**
```bash
curl -X POST http://localhost:3000/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Привіт! Навчи мене чогось", "user_id": "test"}'
```

Дивіться логи в реальному часі:
```bash
docker-compose logs -f agent
```
