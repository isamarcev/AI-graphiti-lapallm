# Multi-stage build для оптимізації розміру образу
FROM python:3.11-slim as builder

# Встановлення системних залежностей для build
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Робоча директорія
WORKDIR /build

# Копіювання requirements
COPY requirements.txt .

# Встановлення Python залежностей (без dev dependencies)
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

# Мітки
LABEL maintainer="Tabula Rasa Agent Team"
LABEL description="Tabula Rasa Agent - Knowledge-centered conversational AI"

# Системні залежності для runtime
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Створення non-root користувача для безпеки
RUN useradd -m -u 1000 appuser

# Робоча директорія
WORKDIR /app

# Копіювання Python пакетів з builder stage
COPY --from=builder /root/.local /home/appuser/.local

# Копіювання коду застосунку
COPY --chown=appuser:appuser . .

# Додавання .local/bin до PATH
ENV PATH=/home/appuser/.local/bin:$PATH
ENV HF_HOME=/home/appuser/.cache/huggingface

# Перемикання на non-root користувача
USER appuser

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

# Expose порт
EXPOSE 3000

# Запуск через uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3000", "--workers", "1"]
