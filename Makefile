# SafeFish — comandos del proyecto. `RUN` permite usar Poetry: make eval RUN="poetry run"
RUN ?= uv run
CONFIG ?= configs/v1.yaml
SPLIT ?= dev
LIMIT ?=
GOLDEN ?= golden

.PHONY: help install test lint format kb eval compare golden-check demo-fixture hooks

help:
	@echo "make install                         Instala dependencias (uv sync)"
	@echo "make test                            Tests unitarios, de contrato y smoke (sin API)"
	@echo "make lint                            Ruff (lint + formato)"
	@echo "make kb CONFIG=configs/v1.yaml       Construye el snapshot de KB del config"
	@echo "make eval CONFIG=... SPLIT=dev       Corre el arnés sobre el split del golden"
	@echo "make compare A=results/<run> B=results/<run>   Deltas, IC bootstrap y McNemar"
	@echo "make golden-check                    Valida el golden (esquema, duplicados, leakage)"
	@echo "make demo-fixture                    KB + eval de punta a punta sobre fixtures (usa la API)"

install:
	uv sync

hooks:
	$(RUN) pre-commit install

test:
	$(RUN) pytest

lint:
	$(RUN) ruff check src eval tests
	$(RUN) ruff format --check src eval tests

format:
	$(RUN) ruff check --fix src eval tests
	$(RUN) ruff format src eval tests

kb:
	$(RUN) python -m safefish.ingest.build_kb --config $(CONFIG)

eval:
	$(RUN) python -m eval.run --config $(CONFIG) --split $(SPLIT) --golden-dir $(GOLDEN) $(if $(LIMIT),--limit $(LIMIT),)

compare:
	$(RUN) python -m eval.compare --a $(A) --b $(B)

golden-check:
	$(RUN) python -m eval.golden --golden-dir $(GOLDEN) --kb-config $(CONFIG)

demo-fixture:
	$(RUN) python -m safefish.ingest.build_kb --config configs/v1-fixture.yaml --force
	$(RUN) python -m eval.run --config configs/v0.yaml --split dev --golden-dir tests/fixtures/golden
	$(RUN) python -m eval.run --config configs/v1-fixture.yaml --split dev --golden-dir tests/fixtures/golden
