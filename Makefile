# SafeFish — comandos del proyecto. `RUN` permite usar Poetry: make eval RUN="poetry run"
RUN ?= uv run
CONFIG ?= configs/v1.yaml
SPLIT ?= dev
LIMIT ?=
GOLDEN ?= golden

.PHONY: help install test lint format data kb coverage eval compare report golden-check demo-fixture demo-fixture-v3 hooks

help:
	@echo "make install                         Instala dependencias (uv sync)"
	@echo "make test                            Tests unitarios, de contrato y smoke (sin API)"
	@echo "make lint                            Ruff (lint + formato)"
	@echo "make data                           Baja las fuentes públicas a data/raw/ y prepara los CSV de mensajes"
	@echo "make data GRUPOS=es,en,ref          Suma datasets en inglés y material de referencia"
	@echo "make kb CONFIG=configs/v1.yaml       Construye el snapshot de KB del config"
	@echo "make eval CONFIG=... SPLIT=dev       Corre el arnés sobre el split del golden"
	@echo "make compare A=results/<run> B=results/<run>   Deltas, IC bootstrap y McNemar"
	@echo "make report SPLIT=test                Tabla de la ablación (última corrida por versión)"
	@echo "make golden-check                    Valida el golden (esquema, duplicados, leakage)"
	@echo "make demo-fixture                    KB + eval de punta a punta sobre fixtures (usa la API)"
	@echo "make demo-fixture-v3                 Igual, para V2 y V3 (4 colecciones, planner, verificador)"
	@echo "make coverage CONFIG=configs/v2.yaml Panel de cobertura: casos por modalidad × canal (§2.3)"
	@echo "make eval ... JUDGE=1                Suma la cobertura del plan con juez LLM (V2+)"
	@echo "make eval ... RETRIEVAL_ONLY=1       Barrido de solo retrieval: sin generación (V2a, §2.0)"

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

GRUPOS ?= es
data:
	$(RUN) python -m safefish.ingest.fetch --grupos $(GRUPOS)

kb:
	$(RUN) python -m safefish.ingest.build_kb --config $(CONFIG)

eval:
	$(RUN) python -m eval.run --config $(CONFIG) --split $(SPLIT) --golden-dir $(GOLDEN) $(if $(LIMIT),--limit $(LIMIT),) $(if $(JUDGE),--judge-plan,) $(if $(RETRIEVAL_ONLY),--retrieval-only,)

coverage:
	$(RUN) python -m safefish.ingest.coverage --config $(CONFIG)

compare:
	$(RUN) python -m eval.compare --a $(A) --b $(B)

report:
	$(RUN) python -m eval.report --split $(SPLIT) --out docs/ablation_tabla_$(SPLIT).md

golden-check:
	$(RUN) python -m eval.golden --golden-dir $(GOLDEN) --kb-config $(CONFIG)

demo-fixture:
	$(RUN) python -m safefish.ingest.build_kb --config configs/v1-fixture.yaml --force
	$(RUN) python -m eval.run --config configs/v0.yaml --split dev --golden-dir tests/fixtures/golden
	$(RUN) python -m eval.run --config configs/v1-fixture.yaml --split dev --golden-dir tests/fixtures/golden

demo-fixture-v3:
	$(RUN) python -m safefish.ingest.build_kb --config configs/v2-fixture.yaml --force
	$(RUN) python -m eval.run --config configs/v2-fixture.yaml --split dev --golden-dir tests/fixtures/golden
	$(RUN) python -m eval.run --config configs/v3-fixture.yaml --split dev --golden-dir tests/fixtures/golden
