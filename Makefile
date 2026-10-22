.PHONY: doctor eval pack submit fetch baseline run registry test lint format clean

CONFIG ?= config/harness.yaml
UV ?= $(HOME)/.local/bin/uv

doctor:
	$(UV) run harness --config $(CONFIG) doctor

eval:
	$(UV) run harness --config $(CONFIG) eval

pack:
	@mkdir -p dist
	$(UV) run harness --config $(CONFIG) pack

submit:
	$(UV) run harness --config $(CONFIG) submit

fetch:
	$(UV) run harness --config $(CONFIG) fetch

baseline:
	$(UV) run harness --config $(CONFIG) baseline

run:
	$(UV) run harness --config $(CONFIG) run

registry:
	$(UV) run harness --config $(CONFIG) registry

test:
	$(UV) run pytest

lint:
	$(UV) run ruff check src/ tests/

format:
	$(UV) run ruff format src/ tests/

clean:
	rm -rf reports/ dist/ results/ /tmp/hpc_work/ .pytest_cache/ .ruff_cache/
