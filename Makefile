.PHONY: help install install-dev precommit-install lint test coverage run docker-build docker-help security typecheck format clean verify venv

DEFAULT_GOAL := help

VENV := .venv


help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' | sort

venv: ## Create virtualenv in .venv (uv)
	uv venv
	@echo "Run: source $(VENV)/bin/activate"

install: ## Install package (uv)
	uv pip install -e .

install-dev: ## Install dev dependencies (uv)
	uv pip install -e .[dev]

precommit-install: ## Install pre-commit hooks
	pre-commit install

lint: ## Run formatters and linters (pre-commit or fallback)
	@if command -v pre-commit >/dev/null 2>&1; then \
	  pre-commit run -a; \
	else \
	  echo "pre-commit not found; running black/isort/flake8 fallback"; \
	  (command -v black >/dev/null 2>&1 && black .) || true; \
	  (command -v isort >/dev/null 2>&1 && isort --profile=black .) || true; \
	  (command -v flake8 >/dev/null 2>&1 && flake8 src/ tests/) || true; \
	fi

test: ## Run tests quickly
	pytest -q

coverage: ## Run tests with coverage
	pytest --cov=src/aws_cost_cli --cov-report=term-missing

# Usage: make run QUERY="EC2 spend last month" PROFILE=prod
QUERY ?= EC2 spend last month
PROFILE ?=
run: ## Run CLI query (override QUERY, PROFILE)
	@if [ -n "$(PROFILE)" ]; then \
	  aws-cost-cli query "$(QUERY)" --profile "$(PROFILE)"; \
	else \
	  aws-cost-cli query "$(QUERY)"; \
	fi

docker-build: ## Build Docker image
	docker build -t aws-cost-cli .

docker-help: docker-build ## Run container to show CLI help
	docker run --rm aws-cost-cli --help || true

security: ## Run security checks (bandit, safety)
	bandit -r src/ || true
	safety check || true

typecheck: ## Run mypy type checking
	mypy src/ --ignore-missing-imports --no-strict-optional --allow-redefinition

format: ## Format code with black and isort
	black .
	isort --profile=black .

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache build dist \
	  **/__pycache__ **/*.pyc src/*.egg-info

verify: ## Run lint, typecheck, and coverage
	$(MAKE) lint
	$(MAKE) typecheck
	$(MAKE) coverage
