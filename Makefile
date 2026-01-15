# Makefile для Tabula Rasa Agent

run:
	 uv run uvicorn app:app --reload --port 8080
up-services:
	docker compose -f docker-compose.services.yml up
.PHONY: help build up down restart logs clean test env

# Кольори для виводу
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m # No Color

help: ## Показати допомогу
	@echo "$(BLUE)Tabula Rasa Agent - Makefile Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

env: ## Створити .env файл з прикладу
	@if [ ! -f .env ]; then \
		echo "$(YELLOW)Створюю .env з env.example...$(NC)"; \
		cp .env.example .env; \
		echo "$(GREEN)✓ .env створено! Відредагуйте налаштування перед запуском.$(NC)"; \
	else \
		echo "$(YELLOW)⚠ .env вже існує. Видаліть його щоб створити новий.$(NC)"; \
	fi

build: ## Збілдити Docker images
	@echo "$(BLUE)Building Docker images...$(NC)"
	docker compose build

up: ## Запустити всі сервіси
	@echo "$(BLUE)Starting services...$(NC)"
	docker compose up -d
	@echo "$(GREEN)✓ Services started!$(NC)"
	@echo "$(YELLOW)API: http://localhost:3000$(NC)"
	@echo "$(YELLOW)Docs: http://localhost:3000/docs$(NC)"
	@echo "$(YELLOW)Neo4j: http://localhost:7474$(NC)"

up-build: ## Збілдити та запустити
	@echo "$(BLUE)Building and starting services...$(NC)"
	docker compose up --build
	@echo "$(GREEN)✓ Services started!$(NC)"

down: ## Зупинити всі сервіси
	@echo "$(BLUE)Stopping services...$(NC)"
	docker compose down
	@echo "$(GREEN)✓ Services stopped$(NC)"

down-v: ## Зупинити та видалити volumes (ВИДАЛИТЬ ВСІ ДАНІ!)
	@echo "$(YELLOW)⚠ WARNING: This will delete all data!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker compose down -v; \
		echo "$(GREEN)✓ Services stopped and volumes removed$(NC)"; \
	fi

restart: ## Перезапустити всі сервіси
	@echo "$(BLUE)Restarting services...$(NC)"
	docker compose restart
	@echo "$(GREEN)✓ Services restarted$(NC)"

restart-agent: ## Перезапустити тільки agent
	@echo "$(BLUE)Restarting agent...$(NC)"
	docker compose restart agent
	@echo "$(GREEN)✓ Agent restarted$(NC)"

logs: ## Показати логи (Ctrl+C для виходу)
	docker compose logs -f

logs-agent: ## Показати логи agent
	docker compose logs -f agent

logs-neo4j: ## Показати логи neo4j
	docker compose logs -f neo4j

ps: ## Показати статус сервісів
	docker compose ps

shell-agent: ## Увійти в shell контейнера agent
	docker compose exec agent bash

shell-neo4j: ## Увійти в shell контейнера neo4j
	docker-compose exec neo4j bash

health: ## Перевірити здоров'я API
	@echo "$(BLUE)Checking API health...$(NC)"
	@curl -s http://localhost:3000/health | python -m json.tool || echo "$(YELLOW)⚠ API not responding$(NC)"

test: ## Тестовий запит до API
	@echo "$(BLUE)Testing API...$(NC)"
	@curl -X POST http://localhost:3000/text \
		-H "Content-Type: application/json" \
		-d '{"text": "Привіт! Навчи мене чогось", "user_id": "test-user"}' | python -m json.tool

test-teach: ## Тест навчання (TEACH)
	@echo "$(BLUE)Testing TEACH path...$(NC)"
	@curl -X POST http://localhost:3000/text \
		-H "Content-Type: application/json" \
		-d '{"text": "Значення pi дорівнює 3.14", "user_id": "test-user"}' | python -m json.tool

test-solve: ## Тест запиту (SOLVE)
	@echo "$(BLUE)Testing SOLVE path...$(NC)"
	@curl -X POST http://localhost:3000/text \
		-H "Content-Type: application/json" \
		-d '{"text": "Яке значення pi?", "user_id": "test-user"}' | python -m json.tool

clean: ## Очистити Docker resources (images, containers, volumes)
	@echo "$(YELLOW)⚠ WARNING: This will remove all Docker resources!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker-compose down -v; \
		docker system prune -af --volumes; \
		echo "$(GREEN)✓ Cleanup complete$(NC)"; \
	fi

check-vllm: ## Перевірити доступність vLLM сервера
	@echo "$(BLUE)Checking vLLM server...$(NC)"
	@curl -s http://localhost:8000/v1/models || echo "$(YELLOW)⚠ vLLM не доступний на localhost:8000$(NC)"

check-deps: ## Перевірити залежності в контейнері
	docker-compose run --rm agent pip list

stats: ## Показати статистику використання ресурсів
	docker stats --no-stream

neo4j-query: ## Відкрити Neo4j Browser
	@echo "$(BLUE)Opening Neo4j Browser...$(NC)"
	@echo "$(YELLOW)URL: http://localhost:7474$(NC)"
	@echo "$(YELLOW)Login: neo4j / password123$(NC)"
	@open http://localhost:7474 2>/dev/null || xdg-open http://localhost:7474 2>/dev/null || echo "$(YELLOW)Відкрийте вручну: http://localhost:7474$(NC)"

docs: ## Відкрити API документацію
	@echo "$(BLUE)Opening API docs...$(NC)"
	@open http://localhost:3000/docs 2>/dev/null || xdg-open http://localhost:3000/docs 2>/dev/null || echo "$(YELLOW)Відкрийте вручну: http://localhost:3000/docs$(NC)"

# Development commands
dev-up: env up-build ## Setup для розробки (створює .env та запускає)
	@echo "$(GREEN)✓ Development environment ready!$(NC)"

dev-logs: ## Логи для розробки з фільтром
	docker-compose logs -f agent | grep -E "(INFO|ERROR|WARNING)"

dev-rebuild: ## Rebuild agent без кешу
	@echo "$(BLUE)Rebuilding agent (no cache)...$(NC)"
	docker-compose build --no-cache agent
	docker-compose up -d agent
	@echo "$(GREEN)✓ Agent rebuilt$(NC)"

# Quick commands
quick-start: env up-build health ## Швидкий старт (все в одному)
	@echo "$(GREEN)✓ Tabula Rasa Agent is ready!$(NC)"

quick-stop: down ## Швидка зупинка
	@echo "$(GREEN)✓ Stopped$(NC)"

quick-restart: restart health ## Швидкий рестарт з перевіркою
	@echo "$(GREEN)✓ Restarted$(NC)"
