.PHONY: help install install-dev lint format format-check typecheck check fix test test-fast test-cov clean download-all download-models download-model download-data download-data-game download-data-split generate-data-game generate-data-train

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---------- Installation ----------

install: ## Install the package in editable mode
	pip install -e .

install-dev: ## Install the package with dev dependencies and pre-commit hooks
	pip install -e ".[dev]"
	pre-commit install

# ---------- Code Quality ----------

lint: ## Run ruff linter
	ruff check metaothello/ tests/

format: ## Auto-format code with ruff
	ruff format metaothello/ tests/

format-check: ## Check formatting without modifying files
	ruff format --check metaothello/ tests/

typecheck: ## Run pyright type checker
	pyright metaothello/

check: ## Run linter, format check, and type checker
	ruff check metaothello/ tests/
	ruff format --check metaothello/ tests/
	pyright metaothello/

fix: ## Auto-fix lint errors and reformat
	ruff check --fix metaothello/ tests/
	ruff format metaothello/ tests/

# ---------- Testing ----------

test: ## Run full test suite
	pytest tests/

test-fast: ## Run tests, skipping slow and gpu tests
	pytest tests/ -m "not slow and not gpu"

test-cov: ## Run tests with coverage report
	pytest tests/ --cov=metaothello --cov-report=term-missing

# ---------- Download ----------
HF_REPO ?= aviralchawla/metaothello
GAME    ?= classic
N_GAMES ?= 1
SPLIT   ?= train

download-all: ## Download all GPT models and all game data from HuggingFace
	python scripts/download.py all

download-models: ## Download all GPT models from HuggingFace
	python scripts/download.py models

download-model: ## Download GPT model for one game  [GAME=classic]
	python scripts/download.py models --game $(GAME)

download-data: ## Download all game data from HuggingFace
	python scripts/download.py data

download-data-game: ## Download all splits for one game  [GAME=classic]
	python scripts/download.py data --game $(GAME)

download-data-split: ## Download one game/split pair  [GAME=classic SPLIT=train]
	python scripts/download.py data --game $(GAME) --split $(SPLIT)

# ---------- Generate Data ----------
generate-data-game: ## Generate game data locally  [GAME=classic N_GAMES=1 SPLIT=train]
	python scripts/generate_data.py --game $(GAME) --num_games $(N_GAMES) --split $(SPLIT)

generate-data-train: ## Generate 20M training games for each game variant
	for game in classic nomidflip delflank iago; do \
		python scripts/generate_data.py --game $$game --num_games 20 --split train; \
	done

# ---------- Cleanup ----------

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info .eggs/
	rm -rf .pytest_cache/ .ruff_cache/ .pyright/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
