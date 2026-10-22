.PHONY: doctor eval pack submit fetch baseline run registry test lint format clean

CONFIG ?= config/harness.yaml
UV ?= $(HOME)/.local/bin/uv
log ?= off

doctor:
	$(UV) run harness --config $(CONFIG) --log $(log) doctor

eval:
	$(UV) run harness --config $(CONFIG) --log $(log) eval

pack:
	@mkdir -p dist
	$(UV) run harness --config $(CONFIG) --log $(log) pack

submit:
	$(UV) run harness --config $(CONFIG) --log $(log) submit

fetch:
	$(UV) run harness --config $(CONFIG) --log $(log) fetch

baseline:
	$(UV) run harness --config $(CONFIG) --log $(log) baseline

run:
	$(UV) run harness --config $(CONFIG) --log $(log) run

registry:
	$(UV) run harness --config $(CONFIG) --log $(log) registry

test:
	$(UV) run pytest

lint:
	$(UV) run ruff check src/ tests/

format:
	$(UV) run ruff format src/ tests/

clean:
	rm -rf reports/ dist/ results/ /tmp/hpc_work/ .pytest_cache/ .ruff_cache/
