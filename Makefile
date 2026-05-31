.PHONY: flagship flagship-ci reproduce test lint

flagship:
	python3 scripts/flagship_metalens_dfm.py --seed-sweep 10

flagship-ci:
	python3 scripts/flagship_metalens_dfm.py --seed-sweep 3

reproduce: ## One-key reproducibility (fixed seed)
	python3 scripts/flagship_metalens_dfm.py --seeds 42 43 44

test:
	python3 -m pytest tests/ -v --tb=short

lint:
	ruff check --fix diffnano/ tests/ scripts/
	ruff format diffnano/ tests/ scripts/
