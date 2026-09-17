PYTHON ?= .venv/bin/python
RUFF ?= .venv/bin/ruff
TY ?= .venv/bin/ty
.PHONY: setup test lint folds audit smoke
setup:
	uv sync --extra dev
test:
	$(PYTHON) -m pytest -q
lint:
	$(RUFF) format .
	$(RUFF) check .
	$(TY) check
folds:
	$(PYTHON) scripts/prepare_folds.py
audit:
	$(PYTHON) scripts/audit_data.py --check-images
smoke:
	$(PYTHON) train.py experiment=smoke
