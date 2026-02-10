.PHONY: help install install-dev install-train install-analysis lint format format-check typecheck check fix test test-fast test-cov clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---------- Installation ----------

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

# ---------- Code Quality ----------

lint:
	ruff check metaothello/ tests/

format:
	ruff format metaothello/ tests/

format-check:
	ruff format --check metaothello/ tests/

typecheck:
	pyright metaothello/

check:
	ruff check metaothello/ tests/
	ruff format --check metaothello/ tests/
	pyright metaothello/

fix:
	ruff check --fix metaothello/ tests/
	ruff format metaothello/ tests/

# ---------- Testing ----------

test:
	pytest tests/

test-fast:
	pytest tests/ -m "not slow and not gpu"

test-cov:
	pytest tests/ --cov=metaothello --cov-report=term-missing

# ---------- Cleanup ----------

clean:
	rm -rf build/ dist/ *.egg-info .eggs/
	rm -rf .pytest_cache/ .ruff_cache/ .pyright/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
