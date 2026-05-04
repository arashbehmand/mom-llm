.PHONY: help install format lint test test-cov clean run docker-build docker-up docker-down pre-commit-install pre-commit-run

# Default target
help:
	@echo "MoM Service - Available Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  make install            Install all dependencies"
	@echo "  make pre-commit-install Install pre-commit hooks"
	@echo ""
	@echo "Code Quality:"
	@echo "  make format             Format code with Black and Ruff"
	@echo "  make lint               Lint code with Ruff"
	@echo "  make pre-commit-run     Run all pre-commit hooks"
	@echo ""
	@echo "Testing:"
	@echo "  make test               Run tests with pytest"
	@echo "  make test-cov           Run tests with coverage report"
	@echo ""
	@echo "Development:"
	@echo "  make run                Run service locally"
	@echo "  make clean              Clean generated files"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build       Build Docker image"
	@echo "  make docker-up          Start services with docker-compose"
	@echo "  make docker-down        Stop services with docker-compose"

# Installation
install:
	pip install -r requirements.txt

pre-commit-install:
	pre-commit install

# Code quality
format:
	@echo "Formatting code with Black..."
	black mom_service/ tests/
	@echo "Formatting code with Ruff..."
	ruff format mom_service/ tests/

lint:
	@echo "Linting code with Ruff..."
	ruff check mom_service/ tests/

pre-commit-run:
	@echo "Running pre-commit hooks on all files..."
	pre-commit run --all-files

# Testing
test:
	pytest

test-cov:
	pytest --cov=mom_service --cov-report=html --cov-report=term

# Development
run:
	uvicorn mom_service.main:app --reload --host 0.0.0.0 --port 8000

clean:
	@echo "Cleaning generated files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	@echo "Clean complete!"

# Docker
docker-build:
	docker build -t mom-service .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

# Combined quality check (format + lint + test)
check: format lint test
	@echo "All quality checks passed!"
