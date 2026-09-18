PYTHON ?= .venv/bin/python
RUFF ?= .venv/bin/ruff
TY ?= .venv/bin/ty
.PHONY: setup test lint folds audit smoke lock
setup:
	uv sync --extra dev
lock:
	uv lock
	uv export --frozen --no-dev --no-emit-project --format requirements.txt -o requirements/runtime.txt
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
