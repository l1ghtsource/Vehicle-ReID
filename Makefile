PYTHON ?= .venv/bin/python
.PHONY: setup test folds audit smoke
setup:
	uv sync --extra dev
test:
	$(PYTHON) -m pytest -q
folds:
	$(PYTHON) scripts/prepare_folds.py
audit:
	$(PYTHON) scripts/audit_data.py --check-images
smoke:
	$(PYTHON) train.py experiment=smoke
