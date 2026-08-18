.PHONY: help install fmt fmt-check lint typecheck imports test cov check run clean \
        docker-build docker-up docker-down

# Tools resolve from the project venv (populated by `make install` / `uv sync`).
PY  := .venv/bin/python
BIN := .venv/bin

help:
	@echo "MoM v2 — Available Commands:"
	@echo "  make install     Sync deps into .venv (uv, dev group)"
	@echo "  make fmt         Format (ruff)"
	@echo "  make lint        Lint (ruff) + import contracts"
	@echo "  make typecheck   Type-check src/mom (mypy --strict)"
	@echo "  make test        Run the v2 test suite"
	@echo "  make cov         v2 tests with coverage"
	@echo "  make check       fmt-check + lint + typecheck + test"
	@echo "  make run         Run the server (dev reload)"

install:
	uv sync --group dev

fmt:
	$(BIN)/ruff format src/mom tests
	$(BIN)/ruff check --fix src/mom tests

fmt-check:
	$(BIN)/ruff format --check src/mom tests

lint:
	$(BIN)/ruff check .
	$(BIN)/lint-imports

typecheck:
	$(BIN)/mypy

test:
	$(PY) -m pytest

cov:
	$(PY) -m pytest --cov=mom --cov-report=term-missing

check: fmt-check lint typecheck test
	@echo "All checks passed."

run:
	$(BIN)/mom serve --reload

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

docker-build:
	docker build -t mom-llm:latest .
docker-up:
	docker compose up -d
docker-down:
	docker compose down
