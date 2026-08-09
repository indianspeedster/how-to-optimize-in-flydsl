# The FlyDSL wheel + ROCm torch live in this venv; nothing here is installable
# from PyPI, so every target goes through it explicitly.
PY ?= /root/flydsl-wgrad-ragged/.venv/bin/python
export PYTHONPATH := $(CURDIR)

.PHONY: test bench bench-json figures list clean

test:                     ## correctness for every variant at its smallest shape
	$(PY) -m pytest tests -q

bench:                    ## the whole ladder, every op, every shape
	$(PY) -m bench

bench-json:               ## same, and write results/bench.json
	@mkdir -p results
	$(PY) -m bench --json results/bench.json

figures:                  ## redraw figure/*.svg from results/bench.json
	$(PY) docs/make_figures.py

list:                     ## show every op and its rungs
	$(PY) -m bench --list

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache results
