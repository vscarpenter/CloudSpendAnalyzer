# Repository Guidelines

## Architecture Overview
- CLI (`aws_cost_cli.cli`): Click-based entrypoint for `aws-cost-cli` commands.
- Query pipeline: `query_parser`, `query_pipeline`, `response_formatter` coordinate parsing → AWS calls → rendering.
- AWS access (`aws_cost_cli.aws_client`): wraps Boto3 and credentials handling + cache integration.
- Caching/Config: `cache_manager`, `config` read YAML config from `~/.aws-cost-cli/config.yaml`.
- Interactive tools: `interactive_query_builder`, favorites/history helpers.

## Project Structure & Module Organization
- src/aws_cost_cli: Python package and CLI entry points (e.g., `cli.py`).
- tests: Pytest suite organized by feature (e.g., `test_cli.py`).
- config: Default and template YAML configs (copy to `~/.aws-cost-cli/config.yaml`).
- docs, build: User docs and packaging artifacts.

## Build, Test, and Development Commands
- Setup: `uv venv && source .venv/bin/activate`
- Install (dev): `uv pip install -e .[dev]`
- Lint/format: `pre-commit install && pre-commit run -a`
- Tests (quick): `pytest -q`
- Coverage: `pytest --cov=src/aws_cost_cli --cov-report=term-missing`
- Run CLI: `aws-cost-cli query "EC2 spend last month" --profile prod`
- Docker (optional): `docker build -t aws-cost-cli . && docker run --rm aws-cost-cli --help`

Note: Install uv (e.g., `brew install uv`) for fast, reproducible envs.

## Coding Style & Naming Conventions
- Formatting: Black (88 cols) + isort (`--profile black`).
- Linting: Flake8 (ignores E203,W503); fix warnings or justify.
- Types: Mypy on CI/pre-commit (`--ignore-missing-imports`). Use type hints.
- Python style: 4-space indent; modules/functions `snake_case`, classes `PascalCase`.
- CLI: Click commands live in `aws_cost_cli.cli`; prefer verbs for command names.

## Testing Guidelines
- Frameworks: pytest + pytest-cov.
- Location/names: `tests/test_*.py`; functions/classes start with `test_`.
- Focused runs: `pytest tests/test_cli.py -k query -q`.
- Coverage goal: maintain ≥90% (target 95%+ per roadmap). Add tests for new code paths and error handling.

## Commit & Pull Request Guidelines
- Commits: Follow Conventional Commits (e.g., `feat:`, `fix:`, `chore:`). Keep messages imperative and scoped.
- Branches: short, descriptive (e.g., `feat/query-suggestions`, `fix/date-parsing`).
- PRs must include: clear description, linked issues, before/after CLI output or screenshots, tests, and docs/CHANGELOG updates when relevant.
- Quality gate: run `pre-commit run -a` and `pytest --cov=src/aws_cost_cli` before requesting review.

## Security & Configuration Tips
- Credentials: never commit secrets. Use AWS profiles or env vars (`AWS_PROFILE`, etc.).
- Config: start from `config/default_config.yaml`; place user config at `~/.aws-cost-cli/config.yaml` and protect it (`chmod 600`).
- Static checks: run `bandit -r src/` and `safety check` for security posture.
