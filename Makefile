PYTHON ?= python3
DATA_DIR ?= data
WORKERS ?= 6

.PHONY: all download database verify figures tensors check-tensors check idempotency test

all:
	$(MAKE) download
	$(MAKE) database
	$(MAKE) verify
	$(MAKE) figures
	$(MAKE) tensors
	$(MAKE) check-tensors

download:
	$(PYTHON) src/download.py --data-dir "$(DATA_DIR)" --workers $(WORKERS)

database:
	$(PYTHON) src/build_db.py --data-dir "$(DATA_DIR)"

verify:
	$(PYTHON) src/verify.py --data-dir "$(DATA_DIR)"

figures:
	$(PYTHON) src/figures.py --data-dir "$(DATA_DIR)"

tensors:
	$(PYTHON) src/export_tensors.py --data-dir "$(DATA_DIR)"

check-tensors:
	$(PYTHON) src/check_tensors.py --data-dir "$(DATA_DIR)"

check:
	$(MAKE) verify
	$(MAKE) check-tensors
	$(MAKE) test

idempotency:
	$(PYTHON) src/check_idempotency.py --data-dir "$(DATA_DIR)"

test:
	$(PYTHON) -m unittest discover -s tests -v
