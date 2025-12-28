.PHONY: help install install-dev test coverage lint format clean pre-commit

help:  ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install production dependencies
	uv pip install -e .

install-dev:  ## Install development dependencies
	uv pip install -e ".[dev]"
	pre-commit install

test:  ## Run tests
	pytest

coverage:  ## Run tests with coverage report
	pytest --cov=src --cov-report=term-missing --cov-report=html

lint:  ## Check code with ruff
	ruff check .

lint-fix:  ## Check and auto-fix code with ruff
	ruff check --fix .

format:  ## Format code with ruff
	ruff format .

format-check:  ## Check if code is formatted
	ruff format --check .

pre-commit:  ## Run pre-commit hooks on all files
	pre-commit run --all-files

clean:  ## Clean up cache and build files
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
