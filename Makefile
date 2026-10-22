.PHONY: doctor eval pack submit fetch baseline run registry test lint format clean

CONFIG ?= config/harness.yaml

doctor:
	uv run harness --config $(CONFIG) doctor

eval:
	uv run harness --config $(CONFIG) eval

pack:
	@mkdir -p dist
	uv run harness --config $(CONFIG) pack

submit:
	uv run harness --config $(CONFIG) submit

fetch:
	uv run harness --config $(CONFIG) fetch

baseline:
	uv run harness --config $(CONFIG) baseline

run:
	uv run harness --config $(CONFIG) run

registry:
	uv run harness --config $(CONFIG) registry

test:
	uv run pytest

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

clean:
	rm -rf reports/ dist/ results/ /tmp/hpc_work/ .pytest_cache/ .ruff_cache/
