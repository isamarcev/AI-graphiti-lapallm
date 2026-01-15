# 🚀 Quickstart - Tabula Rasa Agent

## ⚡ Найшвидший Спосіб (3 команди)

```bash
# 1. Створити .env з налаштуваннями
make env

# 2. Відредагувати .env - встановити URL до vLLM
nano .env  # або vim, code, etc.

# 3. Запустити все одною командою
make quick-start
```

**Готово!** API доступний на http://localhost:3000 🎉

---

## 📋 Детальна Інструкція

### Крок 1: Підготовка

#### 1.1 Перевірте що встановлено:
- ✅ Docker
- ✅ Docker Compose

```bash
docker --version
docker-compose --version
```

#### 1.2 Створіть .env файл:

```bash
# Опція 1: Через Makefile
make env

# Опція 2: Вручну
cp env.example .env
```

#### 1.3 Налаштуйте .env:

**КРИТИЧНО:** Встановіть URL до вашого vLLM сервера:

```bash
# Якщо vLLM запущений локально на host:
VLLM_BASE_URL=http://host.docker.internal:8000/v1

# Якщо vLLM на іншому сервері:
VLLM_BASE_URL=http://your-server-ip:8000/v1
```

Інші важливі налаштування:
```bash
NEO4J_PASSWORD=password123        # Змініть для production
LOG_LEVEL=INFO                    # DEBUG для детального логування
LLM_TEMPERATURE=0.7               # Креативність (0.0-1.0)
```

### Крок 2: Запуск

```bash
# Через Makefile (рекомендовано)
make up-build

# Або через docker-compose
docker-compose up -d --build
```

**Перший запуск займе 2-5 хвилин** (завантаження images та залежностей).

### Крок 3: Перевірка

```bash
# Перевірити статус
make ps

# Перевірити здоров'я API
make health

# Або вручну
curl http://localhost:3000/health
```

**Очікувана відповідь:**
```json
{
  "status": "healthy",
  "timestamp": "2026-01-11T..."
}
```

### Крок 4: Тестування

#### Через Makefile:

```bash
# Тест навчання (TEACH)
make test-teach

# Тест запиту (SOLVE)
make test-solve
```

#### Через curl:

```bash
# Навчити агента
curl -X POST http://localhost:3000/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Значення pi дорівнює 3.14",
    "user_id": "test-user"
  }'

# Запитати агента
curl -X POST http://localhost:3000/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Яке значення pi?",
    "user_id": "test-user"
  }'
```

#### Через Swagger UI:

Відкрийте http://localhost:3000/docs

---

## 🛠️ Корисні Команди

### Управління

```bash
make up          # Запустити
make down        # Зупинити
make restart     # Перезапустити
make logs        # Дивитись логи
make ps          # Статус сервісів
```

### Розробка

```bash
make dev-up         # Setup для розробки
make dev-logs       # Логи з фільтром
make dev-rebuild    # Rebuild без кешу
```

### Тестування

```bash
make health         # Перевірка API
make test-teach     # Тест навчання
make test-solve     # Тест запиту
make check-vllm     # Перевірка vLLM
```

### Моніторинг

```bash
make logs           # Всі логи
make logs-agent     # Логи тільки agent
make stats          # Статистика ресурсів
```

### UI

```bash
make docs           # Відкрити API docs
make neo4j-query    # Відкрити Neo4j Browser
```

---

## 📡 Endpoints

### API (Port 3000)

- **Docs**: http://localhost:3000/docs
- **Root**: http://localhost:3000/
- **Health**: http://localhost:3000/health
- **Text**: POST http://localhost:3000/text

### Neo4j (Port 7474/7687)

- **Browser**: http://localhost:7474
- **Login**: neo4j / password123
- **Bolt**: bolt://localhost:7687

---

## 🐛 Швидке Вирішення Проблем

### Agent не стартує

```bash
# Дивимось логи
make logs-agent

# Rebuild без кешу
make dev-rebuild
```

### vLLM недоступний

```bash
# Перевірка
make check-vllm

# Якщо помилка - виправте VLLM_BASE_URL в .env
nano .env
make restart-agent
```

### Neo4j не стартує

```bash
# Дивимось логи
make logs-neo4j

# Даємо більше часу (може займати до 60 сек)
sleep 30
make ps

# Якщо не допомагає - очистка
make down-volumes
make up-build
```

### Порт зайнятий

Змініть порт в `docker-compose.yml`:
```yaml
agent:
  ports:
    - "8080:3000"  # Використати 8080
```

---

## 🧹 Очистка

```bash
# Зупинити сервіси
make down

# Зупинити + видалити дані (⚠️ ВИДАЛИТЬ ВСЕ!)
make down-volumes

# Повна очистка Docker (⚠️ ВИДАЛИТЬ ВСЕ!)
make clean
```

---

## 📚 Більше Інформації

- **Детальний Setup**: [DOCKER_SETUP.md](docs/DOCKER_SETUP.mdUP.md)
- **Testing Guide**: [TABULA_RASA_TESTING.md](TABULA_RASA_TESTING.md)
- **Implementation**: [TABULA_RASA_IMPLEMENTATION.md](TABULA_RASA_IMPLEMENTATION.md)
- **API Usage**: [README.md](../README.md)

---

## ✅ Чеклист Запуску

- [ ] Docker та Docker Compose встановлені
- [ ] vLLM сервер запущений та доступний
- [ ] `.env` файл створений з правильним `VLLM_BASE_URL`
- [ ] `make up-build` виконано успішно
- [ ] `make health` повертає "healthy"
- [ ] Тестові запити працюють
- [ ] Neo4j UI доступний на http://localhost:7474

**Якщо всі пункти ✅ - агент готовий до роботи!** 🚀

---

## 🆘 Потрібна Допомога?

1. Перевірте логи: `make logs-agent`
2. Перегляньте [DOCKER_SETUP.md](docs/DOCKER_SETUP.mdUP.md) - розділ Troubleshooting
3. Перевірте що vLLM доступний: `make check-vllm`
4. Спробуйте повний rebuild: `make dev-rebuild`

**Все ще не працює?** Створіть issue з виводом:
```bash
make logs-agent > logs.txt
make ps >> logs.txt
docker version >> logs.txt
```
