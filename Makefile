.PHONY: doctor eval pack submit fetch baseline run registry test lint format clean

CONFIG ?= config/harness.yaml
UV ?= $(HOME)/.local/bin/uv
LOG ?= off

doctor:
	$(UV) run harness --config $(CONFIG) --log $(LOG) doctor

eval:
	$(UV) run harness --config $(CONFIG) --log $(LOG) eval

pack:
	@mkdir -p dist
	$(UV) run harness --config $(CONFIG) --log $(LOG) pack

submit:
	$(UV) run harness --config $(CONFIG) --log $(LOG) submit

fetch:
	$(UV) run harness --config $(CONFIG) --log $(LOG) fetch

baseline:
	$(UV) run harness --config $(CONFIG) --log $(LOG) baseline

run:
	$(UV) run harness --config $(CONFIG) --log $(LOG) run

registry:
	$(UV) run harness --config $(CONFIG) --log $(LOG) registry

test:
	$(UV) run pytest

lint:
	$(UV) run ruff check src/ tests/

format:
	$(UV) run ruff format src/ tests/

clean:
	rm -rf reports/ dist/ results/ /tmp/hpc_work/ .pytest_cache/ .ruff_cache/
