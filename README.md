# SafeFish

**TP3 · 73.69 Large Language Models · ITBA 2026** — Diego Badin · Diego Rabinovich

RAG clasificador-explicativo de mensajes sospechosos: entra un mensaje (texto o captura) y sale
un JSON con veredicto en 4 niveles, modalidad, señales con cita literal, casos documentados que
respaldan la decisión y un plan de acción. El plan completo está en
[`docs/SafeFish_Plan_Implementacion.md`](docs/SafeFish_Plan_Implementacion.md).

## Instalación

```bash
uv sync            # o: poetry install   (Poetry >= 2.2)
cp .env.example .env   # completar OPENAI_API_KEY
make hooks         # pre-commit con detección de secretos
make test          # unit + contrato + smoke, sin llamar a la API
```

## Comandos

| Comando | Qué hace |
|---|---|
| `make kb CONFIG=configs/v1.yaml` | Construye el snapshot de KB del config (`data/kb_snapshots/<kb>/`) |
| `make eval CONFIG=configs/v1.yaml SPLIT=dev` | Corre el pipeline sobre el golden y escribe `results/<run_id>/` |
| `make compare A=results/<run_a> B=results/<run_b>` | Deltas, IC 95 % bootstrap y McNemar pareado |
| `make golden-check` | Valida el golden: esquema, ids, splits y leakage de `campaign_id` contra la KB |
| `make demo-fixture` | KB + V0 + V1 de punta a punta sobre la mini-KB de fixtures (usa la API, centavos) |
| `make demo-fixture-v3` | Igual para V2 y V3: KB normalizada en 4 colecciones, planner y verificador |
| `make coverage CONFIG=configs/v2.yaml` | Panel de cobertura: casos por modalidad × canal; marca las modalidades con < 5 casos |
| `make eval CONFIG=configs/v3.yaml JUDGE=1` | Suma la cobertura del plan con juez LLM (`--judge-plan`) |
| `uv run python -m safefish.postprocess.annotate --imagen … --pred … --salida …` | Dibuja las bbox de las señales sobre la captura |
| `uv run python scripts/make_fixture_bundles.py` | Regenera los 5 `RetrievalBundle` de ejemplo de `tests/fixtures/bundles/` |

Con Poetry: `make eval RUN="poetry run"`.

## Estado

| Fase | Estado |
|---|---|
| F0 · Fundaciones | ✅ Contratos (`schemas.py`), taxonomías, configs YAML, guía de etiquetado, arnés, fixtures (mini-KB + 5 bundles), repo con pre-commit |
| Fase 1 · V0 + V1 Naive RAG | ✅ Código: ingesta → snapshot, retriever denso top-6, razonador con salida estructurada, métricas de tarea y retrieval, `compare` (McNemar + bootstrap). Verificado de punta a punta contra la API sobre fixtures |
| Fase 2 · V2 + V3 | ✅ Código (ADR 003): normalizador (plantilla + LLM mini) a `CaseDoc/MessageDoc/SignalDoc/GuideDoc`, 1 caso = 1 chunk, 4 colecciones con `top_n`, cobertura y auditoría de inyección en el manifest, muestra de revisión del 10 %; planner desde Guías; verificador R1–R5; extracción de capturas (visión + OCR + `align`) y captura anotada. Barrido V2a. Verificado sin API sobre fixtures |
| Gate G0 / G1 / G2 | ⏳ Faltan los **datos**: fuentes reales en `data/sources.yaml` → `make kb`, y el golden real (`golden/dev.jsonl`, `golden/test.jsonl`) según la guía de etiquetado. Candidatas relevadas en [`docs/fuentes_candidatas.md`](docs/fuentes_candidatas.md) |

### Probar la Fase 1 sobre fixtures (≈ USD 0,08)

```bash
make demo-fixture
make compare A=results/<run_v0> B=results/<run_v1-fixture>
```

### Correr V0 y V1 con datos reales

```bash
make kb CONFIG=configs/v1.yaml                 # KB desde data/sources.yaml → kb-v1
make golden-check                               # esquema + leakage de campaign_id contra kb-v1
make eval CONFIG=configs/v0.yaml SPLIT=dev
make eval CONFIG=configs/v1.yaml SPLIT=dev
make compare A=results/<run_v0> B=results/<run_v1>   # gate G1
```

### Correr V2 y V3 (Fase 2)

```bash
make kb CONFIG=configs/v2.yaml          # normaliza (LLM mini solo si la fuente no tiene plantilla) → kb-v2
make coverage CONFIG=configs/v2.yaml    # ≥ 5 casos por modalidad antes de medir
# revisar a mano data/kb_snapshots/kb-v2/normalizacion_revision.jsonl (muestra del 10 %)
make kb CONFIG=configs/v2a.yaml         # barrido de solo retrieval: chunk por caso en 1 colección
make eval CONFIG=configs/v2.yaml SPLIT=dev JUDGE=1
make eval CONFIG=configs/v3.yaml SPLIT=dev JUDGE=1   # mismo snapshot kb-v2
make compare A=results/<run_v2> B=results/<run_v3>
```

Para capturas, el OCR necesita el binario de Tesseract con el idioma `spa` (Windows: instalador
de UB Mannheim; Debian/Ubuntu: `apt install tesseract-ocr tesseract-ocr-spa`). Sin Tesseract el
pipeline funciona igual, pero las señales salen sin recuadro.

## Estructura

```
configs/            un YAML por versión de la ablación (v0, v1, …)
data/               raw/ e interim/ no se versionan; kb_snapshots/<kb>/manifest.json sí
golden/             dev.jsonl · test.jsonl · GUIA_ETIQUETADO.md
src/safefish/
  schemas.py        contratos entre módulos (congelados en F0)
  taxonomias.yaml   modalidades y señales (Anexo C)
  config.py         carga y hash de configs
  ingest/           loaders · limpieza · PII · chunking · index → snapshot
  extract/          texto (Fase 1) · visión + OCR (Fase 2)
  retrieval/        denso (V1) · multi-query, BM25, rerank (Fase 3)
  generation/       prompts versionados · razonador
  postprocess/      verificador R1–R5 (V3)
  pipeline.py       arma el pipeline desde el YAML
  observability.py  tokens, costo y latencias por etapa
eval/               run · metrics_task · metrics_retrieval · compare · golden
scripts/            utilidades (bundles de fixtures, primera consulta a la API)
docs/               plan · ADRs · fichas de experimentos · data card
tests/              unit · contract · smoke · fixtures (mini-KB sintética)
```

## Convenciones

- **Contract-first**: los módulos solo comparten `schemas.py`. Cambiar un contrato = PR revisado
  por ambos + tests de `tests/contract/`.
- **Una variable por versión**: cada técnica es un flag del YAML, nunca un branch largo.
- **Trunk-based**: ramas cortas (< 1 día) contra `main`. Tags por hito: `v3-pipeline`, `v1-ej2`,
  `v2-ej4`, `final`.
- **Datos**: nada crudo ni con PII en git. Las corridas oficiales de `results/` se agregan con
  `git add -f`.
