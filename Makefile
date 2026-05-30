.PHONY: flagship-a flagship-a-ci

flagship-a:
	python3 scripts/flagship_metalens_dfm.py --seed-sweep 10

flagship-a-ci:
	python3 scripts/flagship_metalens_dfm.py --seed-sweep 3
