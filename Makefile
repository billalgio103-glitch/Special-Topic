PYTHON ?= python3
DATA_DIR ?= data
WORKERS ?= 6
MODEL_DIR = $(DATA_DIR)/model
# Extra train.py options, e.g. make all TRAIN_ARGS="--epochs 2"
TRAIN_ARGS ?=

.PHONY: all download database verify figures tensors check-tensors model report check idempotency test

all:
	$(MAKE) download
	$(MAKE) database
	$(MAKE) verify
	$(MAKE) figures
	$(MAKE) tensors
	$(MAKE) check-tensors
	$(MAKE) report

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

# The model stages are file targets: make retrains only when the tensors or the
# model code are newer than results.json (export_tensors.py leaves unchanged
# tensor files untouched), then redraws the figures and the PDF.
model: $(MODEL_DIR)/results.json

$(MODEL_DIR)/results.json: $(DATA_DIR)/inputs/train.safetensors $(DATA_DIR)/inputs/eval.safetensors src/train.py src/model.py
	$(PYTHON) src/train.py --data-dir "$(DATA_DIR)" $(TRAIN_ARGS)

$(MODEL_DIR)/figures/roc.png: $(MODEL_DIR)/results.json src/model_figures.py
	$(PYTHON) src/model_figures.py --data-dir "$(DATA_DIR)"

report: report/report.pdf

report/report.pdf: report/report.typ $(MODEL_DIR)/figures/roc.png $(DATA_DIR)/validation.json src/build_report.py
	$(PYTHON) src/build_report.py --data-dir "$(DATA_DIR)"

check:
	$(MAKE) verify
	$(MAKE) check-tensors
	$(MAKE) test

idempotency:
	$(PYTHON) src/check_idempotency.py --data-dir "$(DATA_DIR)"

test:
	$(PYTHON) -m unittest discover -s tests -v
