# Stage 1: Builder
# Используем официальный образ uv для копирования бинарного файла
FROM python:3.12-slim AS builder

# Установка uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Переменные окружения для uv
# Отключаем создание виртуального окружения внутри контейнера, если хотим ставить в систему,
# но лучше оставить venv для легкого переноса между стадиями.
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

WORKDIR /app

# Сначала копируем только файлы зависимостей для эффективного кеширования слоев
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# Копируем исходный код
COPY . .

# Финальная синхронизация проекта
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


# Stage 2: Runtime
FROM python:3.12-slim

LABEL maintainer="Tabula Rasa Agent Team"
LABEL description="Tabula Rasa Agent - Knowledge-centered conversational AI"

# Системные зависимости
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Создание non-root пользователя
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Копируем виртуальное окружение, созданное uv, из стадии builder
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
# Копируем код приложения
COPY --chown=appuser:appuser . .

# Добавляем виртуальное окружение в PATH, чтобы не писать полные пути
ENV PATH="/app/.venv/bin:$PATH"
ENV HF_HOME=/home/appuser/.cache/huggingface
ENV PYTHONUNBUFFERED=1

USER appuser

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

EXPOSE 3000

# Запуск через Python entrypoint для автоматического вычисления workers
# Количество workers: (CPU cores × 2) + 1 (можно переопределить через ENV: WORKERS)
CMD ["python", "start.py"]