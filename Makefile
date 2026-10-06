PYTHON ?= python3
DATA_DIR ?= data
WORKERS ?= 6

.PHONY: all download database verify figures idempotency test

all:
	$(MAKE) download
	$(MAKE) database
	$(MAKE) verify
	$(MAKE) figures

download:
	$(PYTHON) scripts/download.py --data-dir "$(DATA_DIR)" --workers $(WORKERS)

database:
	$(PYTHON) scripts/build_db.py --data-dir "$(DATA_DIR)"

verify:
	$(PYTHON) scripts/verify.py --data-dir "$(DATA_DIR)"

figures:
	$(PYTHON) scripts/figures.py --data-dir "$(DATA_DIR)"

idempotency:
	$(PYTHON) scripts/check_idempotency.py --data-dir "$(DATA_DIR)"

test:
	$(PYTHON) -m unittest discover -s tests -v
