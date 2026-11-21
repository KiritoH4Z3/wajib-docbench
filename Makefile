# Wajib DocBench. `make bench` rebuilds the dataset, reruns OCR and replays every model
# response from results/cache, so every number in the README is reproduced with no GPU and
# no API key. `make bench FRESH=1` queries the models again (needs `make serve` and keys).

MODEL ?= falcon-h1-7b
SOURCE ?= ocr
PARALLEL ?= 4
SHARED_CTX ?= 0
FRESH ?= 0
PY := uv run python

.PHONY: help setup data check ocr ocr-check models serve stop extract samples guard perf aiperf report bench test lint demo-poison

help:
	@echo "make setup            install Python deps, Chromium and the OCR image"
	@echo "make data             generate the 85 synthetic documents and check them"
	@echo "make ocr              run the fixed OCR step and score it"
	@echo "make models MODEL=x   download a model's weights into models/"
	@echo "make serve MODEL=x    start llama.cpp for a local model (PARALLEL slots)"
	@echo "make stop             stop the model server"
	@echo "make extract MODEL=x  extract fields (SOURCE=ocr|oracle, FRESH=1 to skip the cache)"
	@echo "make samples MODEL=x  repeat 12 documents 3 times to measure run-to-run variance"
	@echo "make perf MODEL=x     throughput and latency at concurrency 1, 4 and 16"
	@echo "make report           write results/results.json, the chart and the README table"
	@echo "make bench            everything above from cached responses"
	@echo "make demo-poison      show the guardrail catching a poisoned invoice"

setup:
	uv sync
	uv run playwright install chromium
	docker build -t wajib-docbench-ocr:1 ocr

data:
	$(PY) -m data.generator
	$(PY) -m data.generator --check

check:
	$(PY) -m data.generator --check

ocr:
	$(PY) -m ocr run
	$(PY) -m ocr score

# Reruns OCR into a scratch directory and fails if any text differs from the committed output.
ocr-check:
	$(PY) -m ocr run --check
	$(PY) -m ocr score

models:
	$(PY) -m bench models --model $(MODEL)

serve:
	$(PY) -m bench serve --model $(MODEL) --parallel $(PARALLEL) --shared-ctx $(SHARED_CTX)

stop:
	$(PY) -m bench stop

extract:
	$(PY) -m bench extract --model $(MODEL) --source $(SOURCE) $(if $(filter 1,$(FRESH)),--fresh,)
	$(PY) -m bench guard --model $(MODEL) --source $(SOURCE)

samples:
	$(PY) -m bench samples --model $(MODEL) --source $(SOURCE)

guard:
	$(PY) -m bench guard --model $(MODEL) --source $(SOURCE)

perf:
	$(PY) -m bench perf --model $(MODEL)

aiperf:
	bash perf/run_aiperf.sh $(MODEL)

report:
	$(PY) -m bench report

bench: data ocr-check
	$(PY) -m bench replay $(if $(filter 1,$(FRESH)),--fresh,)
	$(PY) -m bench report

demo-poison:
	$(PY) -m bench demo-poison --model $(MODEL)

test:
	uv run pytest -q

lint:
	uv run ruff check .
