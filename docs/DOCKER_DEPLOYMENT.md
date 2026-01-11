# 🐳 Docker Deployment - Короткий Огляд

## ✅ Що Створено

### 1. **Dockerfile** (Multi-stage build)
- ✅ Stage 1: Builder - компіляція залежностей
- ✅ Stage 2: Production - мінімальний runtime образ
- ✅ Non-root user для безпеки
- ✅ Healthcheck для моніторингу
- ✅ Uvicorn на порту 3000

**Розмір образу:** ~1.5-2 GB (оптимізовано через multi-stage)

### 2. **docker-compose.yml** (Оновлено)
- ✅ `agent` сервіс - Tabula Rasa Agent (порт 3000)
- ✅ `neo4j` сервіс - Neo4j Database (порти 7474, 7687)
- ✅ Network `graphiti-network` для зв'язку
- ✅ Volumes для персистентності даних
- ✅ Healthchecks для обох сервісів
- ✅ `host.docker.internal` для доступу до vLLM на host

### 3. **env.example**
- ✅ Шаблон змінних оточення
- ✅ Коментарі українською
- ✅ Значення за замовчуванням
- ✅ Всі критичні параметри

### 4. **.dockerignore**
- ✅ Виключення непотрібних файлів
- ✅ Зменшення розміру build context
- ✅ Прискорення build

### 5. **Makefile**
- ✅ 30+ команд для управління
- ✅ Кольоровий вивід
- ✅ Shortcuts для швидкого запуску
- ✅ Тестові команди

### 6. **Документація**
- ✅ `QUICKSTART.md` - швидкий старт (3 команди)
- ✅ `DOCKER_SETUP.md` - детальний setup та troubleshooting
- ✅ `DOCKER_DEPLOYMENT.md` - цей файл

---

## 🚀 Швидкий Запуск

```bash
# 1. Створити .env
make env

# 2. Налаштувати VLLM_BASE_URL в .env
nano .env

# 3. Запустити
make quick-start
```

**API готовий на http://localhost:3000** 🎉

---

## 📦 Структура Docker Setup

```
graphity_lapa/
├── Dockerfile              # Multi-stage build
├── docker-compose.yml      # Orchestration
├── .dockerignore          # Build optimization
├── env.example            # Environment template
├── Makefile               # CLI shortcuts
├── requirements.txt       # Python deps (+ uvicorn)
│
├── QUICKSTART.md          # 🚀 Швидкий старт
├── DOCKER_SETUP.md        # 📚 Детальний setup
└── DOCKER_DEPLOYMENT.md   # 🐳 Цей файл
```

---

## 🔧 Конфігурація

### Змінні Оточення (в .env)

#### Обов'язкові:
```bash
VLLM_BASE_URL=http://host.docker.internal:8000/v1  # URL до vLLM
NEO4J_PASSWORD=password123                          # Neo4j пароль
```

#### Рекомендовані:
```bash
LOG_LEVEL=INFO              # DEBUG для розробки
LLM_TEMPERATURE=0.7         # Креативність (0.0-1.0)
GRAPHITI_SEARCH_LIMIT=10    # К-сть results з пам'яті
```

### Порти

| Сервіс | Внутрішній | Зовнішній | Опис |
|--------|-----------|-----------|------|
| agent  | 3000      | 3000      | FastAPI API |
| neo4j  | 7474      | 7474      | Neo4j Web UI |
| neo4j  | 7687      | 7687      | Neo4j Bolt |

Змінити можна в `docker-compose.yml`:
```yaml
agent:
  ports:
    - "8080:3000"  # Зовнішній 8080
```

### Ресурси

За замовчуванням необмежено. Для production додайте в `docker-compose.yml`:

```yaml
agent:
  deploy:
    resources:
      limits:
        cpus: '2'
        memory: 4G
      reservations:
        cpus: '1'
        memory: 2G
```

---

## 🏗️ Build Process

### Multi-stage Build

**Stage 1: Builder**
```dockerfile
FROM python:3.11-slim as builder
# Встановлення build tools (gcc, g++, git)
# Компіляція Python залежностей
# Результат: /root/.local з compiled packages
```

**Stage 2: Production**
```dockerfile
FROM python:3.11-slim
# Копіювання compiled packages
# Копіювання коду
# Non-root user
# Uvicorn CMD
```

**Переваги:**
- ✅ Менший розмір (~40% економії)
- ✅ Безпечніше (no build tools в production)
- ✅ Швидший deploy

### Build Час

- **Перший build:** 5-10 хв (завантаження dependencies)
- **Rebuild (з кешем):** 30-60 сек
- **Rebuild (no cache):** 5-10 хв

---

## 🔐 Безпека

### Поточна Конфігурація (Development)

- ✅ Non-root user в контейнері
- ✅ Multi-stage build (no build tools)
- ⚠️ CORS дозволяє всі origins
- ⚠️ Neo4j password простий

### Production Hardening

**1. Змініть паролі:**
```bash
# У .env
NEO4J_PASSWORD=your-strong-password-$(openssl rand -base64 32)
```

**2. Обмежте CORS:**
```python
# app.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-domain.com"],
    allow_methods=["POST", "GET"],
)
```

**3. Додайте HTTPS:**
```yaml
# docker-compose.yml
services:
  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
```

**4. Secrets Management:**
```bash
# Використовуйте Docker secrets або Vault
echo "password123" | docker secret create neo4j_password -
```

**5. Network Isolation:**
```yaml
# Внутрішня мережа для neo4j
neo4j:
  networks:
    - internal
agent:
  networks:
    - internal
    - external
```

---

## 📊 Моніторинг

### Healthchecks

Обидва сервіси мають healthchecks:

```yaml
agent:
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
    interval: 30s
    timeout: 10s
    retries: 3

neo4j:
  healthcheck:
    test: ["CMD-SHELL", "wget --spider localhost:7474"]
    interval: 10s
```

Перевірка:
```bash
make ps
# або
docker-compose ps
```

### Логи

```bash
# Всі логи
make logs

# З фільтром
make logs-agent | grep ERROR

# З timestamp
docker-compose logs -f --timestamps agent
```

### Метрики

```bash
# Використання ресурсів
make stats

# Або детально
docker stats tabula-rasa-agent tabula-rasa-neo4j
```

### Alerts (Production)

Інтеграція з:
- Prometheus + Grafana
- DataDog
- New Relic
- ELK Stack

---

## 🚀 Deployment Strategies

### Local Development
```bash
make dev-up
```

### Staging
```bash
# З окремим .env
cp .env.staging .env
make up-build
```

### Production

**Опція 1: Single Server**
```bash
# На сервері
git clone <repo>
cp env.example .env
nano .env  # налаштувати
make quick-start
```

**Опція 2: Docker Hub**
```bash
# Build та push
docker build -t yourusername/tabula-rasa-agent:latest .
docker push yourusername/tabula-rasa-agent:latest

# На сервері
docker pull yourusername/tabula-rasa-agent:latest
docker-compose up -d
```

**Опція 3: Kubernetes**
```bash
# Конвертувати з docker-compose
kompose convert -f docker-compose.yml

# Застосувати
kubectl apply -f agent-deployment.yaml
kubectl apply -f neo4j-statefulset.yaml
```

**Опція 4: Docker Swarm**
```bash
docker swarm init
docker stack deploy -c docker-compose.yml tabula-rasa
```

---

## 🔄 CI/CD Integration

### GitHub Actions Example

```yaml
name: Build and Deploy

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Build Docker image
        run: docker build -t tabula-rasa-agent .
      
      - name: Run tests
        run: |
          docker-compose up -d
          sleep 30
          make health
          make test
      
      - name: Push to Docker Hub
        run: |
          echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
          docker push tabula-rasa-agent:latest
```

---

## 🧪 Testing in Docker

```bash
# Unit tests
docker-compose run --rm agent pytest

# Integration tests
make up-build
make test-teach
make test-solve

# Load tests
docker-compose run --rm agent locust -f tests/load_test.py
```

---

## 📈 Performance

### Рекомендовані Ресурси

**Мінімум:**
- CPU: 2 cores
- RAM: 4 GB
- Disk: 10 GB

**Рекомендовано:**
- CPU: 4 cores
- RAM: 8 GB
- Disk: 20 GB (з урахуванням логів)

**Production:**
- CPU: 8+ cores
- RAM: 16+ GB
- Disk: 50+ GB (SSD)

### Оптимізації

**1. Workers:**
```dockerfile
# Dockerfile
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3000", "--workers", "4"]
```

**2. Gunicorn (альтернатива):**
```dockerfile
CMD ["gunicorn", "app:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:3000"]
```

**3. Caching:**
```yaml
agent:
  volumes:
    - model_cache:/home/appuser/.cache  # Кеш моделей
```

---

## ✅ Production Checklist

Перед production deploy:

**Безпека:**
- [ ] Змінені всі паролі
- [ ] CORS обмежений
- [ ] HTTPS налаштовано
- [ ] Secrets в vault/secrets manager

**Конфігурація:**
- [ ] Resource limits встановлені
- [ ] Healthchecks працюють
- [ ] Логування налаштовано
- [ ] Backups налаштовані

**Моніторинг:**
- [ ] Prometheus/Grafana setup
- [ ] Alerts налаштовані
- [ ] Логи централізовані
- [ ] Metrics збираються

**Надійність:**
- [ ] Auto-restart увімкнено
- [ ] Volumes для персистентності
- [ ] Backup strategy
- [ ] Disaster recovery plan

---

## 📚 Додаткові Ресурси

- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [FastAPI in Docker](https://fastapi.tiangolo.com/deployment/docker/)
- [Neo4j Docker Guide](https://neo4j.com/developer/docker/)

---

## 🆘 Підтримка

**Проблеми з Docker?**
1. Перевірте логи: `make logs-agent`
2. Перевірте ресурси: `make stats`
3. Дивіться [DOCKER_SETUP.md](DOCKER_SETUP.md) - Troubleshooting

**Потрібна допомога?**
- Створіть issue з виводом `make logs-agent`
- Додайте `docker version` та `docker-compose version`
- Опишіть кроки відтворення проблеми

---

## 🎯 Готово!

Ваш Tabula Rasa Agent тепер:
- ✅ Легко розгортається через `docker-compose up`
- ✅ Має всі необхідні сервіси (agent + neo4j)
- ✅ Налаштований через .env
- ✅ Має healthchecks та моніторинг
- ✅ Готовий до production (з hardening)

**Happy Deploying!** 🚀
