.PHONY: help dev build test lint typecheck coverage docker-build docker-up docker-down docker-sandbox db-migrate db-rollback db-seed db-reset clean

# Default target
help:
	@echo "Trading Bot Development Commands"
	@echo "================================="
	@echo ""
	@echo "Development:"
	@echo "  make dev          - Start local development environment"
	@echo "  make shell        - Enter Docker development container"
	@echo "  make logs         - Tail all service logs"
	@echo ""
	@echo "Build & Test:"
	@echo "  make build        - Build all services"
	@echo "  make test         - Run full test suite"
	@echo "  make test-unit    - Run unit tests only"
	@echo "  make test-integration - Run integration tests"
	@echo "  make lint         - Run linters (ruff, black, isort)"
	@echo "  make typecheck    - Run mypy"
	@echo "  make coverage     - Generate coverage report"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build - Build all containers"
	@echo "  make docker-up    - Start all services"
	@echo "  make docker-down  - Stop all services"
	@echo "  make docker-sandbox - Start isolated sandbox"
	@echo ""
	@echo "Database:"
	@echo "  make db-migrate   - Run migrations"
	@echo "  make db-rollback  - Rollback last migration"
	@echo "  make db-seed      - Seed development data"
	@echo "  make db-reset     - Drop and recreate database"

# Development
dev:
	docker compose up -d
	@echo "Development environment started"
	@echo "Services: http://localhost:8000"

shell:
	docker compose exec app bash

logs:
	docker compose logs -f

# Build & Test
build:
	docker compose build

test: lint typecheck
	pytest tests/ -v --tb=short

test-unit:
	pytest tests/unit/ -v --tb=short

test-integration:
	pytest tests/integration/ -v --tb=short

lint:
	ruff check .
	black --check .
	isort --check-only .

lint-fix:
	ruff check --fix .
	black .
	isort .

typecheck:
	mypy . --ignore-missing-imports

coverage:
	pytest tests/ --cov=. --cov-report=html --cov-report=term-missing
	@echo "Coverage report: htmlcov/index.html"

# Docker
docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-sandbox:
	docker compose -f docker-compose.sandbox.yml up -d
	@echo "Sandbox started - safe for autonomous development"

# Database
db-migrate:
	alembic upgrade head

db-rollback:
	alembic downgrade -1

db-seed:
	python scripts/seed_data.py

db-reset:
	alembic downgrade base
	alembic upgrade head
	python scripts/seed_data.py

# LEAN
backtest:
	@echo "Usage: lean backtest <AlgorithmName>"

paper-trade:
	@echo "Usage: lean live deploy <AlgorithmName> --brokerage 'Paper Trading'"

# Cleanup
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf htmlcov/
	rm -rf .coverage
