# SafeFish

**TP3 · 73.69 Large Language Models · ITBA 2026** — Diego Badin · Diego Rabinovich

RAG clasificador-explicativo de mensajes sospechosos: entra un mensaje (texto o captura) y sale
un JSON con veredicto en 4 niveles, modalidad, señales con cita literal, casos documentados que
respaldan la decisión y un plan de acción. El plan completo está en
[`docs/SafeFish_Plan_Implementacion.md`](docs/SafeFish_Plan_Implementacion.md).

## Instalación

```bash
uv sync            # o: poetry install   (Poetry >= 2.2)
cp .env.example .env   # completar OPENAI_EMBEDDINGS_API_KEY y CLAUDE_API_KEY
make hooks         # pre-commit con detección de secretos
make test          # unit + contrato + smoke, sin llamar a la API
```

## Comandos

| Comando | Qué hace |
|---|---|
| `make kb CONFIG=configs/v1.yaml` | Construye el snapshot de KB del config (`data/kb_snapshots/<kb>/`) |
| `make eval CONFIG=configs/v1.yaml SPLIT=dev` | Corre el pipeline sobre el golden y escribe `results/<run_id>/` |
| `make compare A=results/<run_a> B=results/<run_b>` | Deltas, IC 95 % bootstrap y McNemar pareado |
| `make report SPLIT=test` | Tabla de la ablación con la última corrida de cada versión → `docs/ablation_tabla_<split>.md` |
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
| Datos | ✅ Fuentes públicas: `make data` las baja (manifiesto `data/descargas.yaml`) y `data/sources.yaml` ya las carga: 12 fuentes de casos, 9 de guías y 4.260 emails etiquetados (SpaPhish + SpearPhishMX). Qué se usa en cada fase y cómo dejar los mensajes propios: [`docs/datos.md`](docs/datos.md) |
| Golden v1 | ✅ 125 casos (35 dev · 90 test): 35 SMS/WhatsApp/email recolectados, 60 emails de SpaPhish, 15 adversariales y 15 capturas; split por `campaign_id`, sin leakage contra la KB. Pre-etiquetado con LLM y revisado (`golden/decisiones_revision.csv`) ([`docs/datos.md`](docs/datos.md#golden-v1)) |
| Fase 3 · V4–V6 | ✅ Multi-query + RRF (V4), híbrido denso + BM25 (V5) y reranking listwise (V6, **Versión 1**). Resultados y análisis: [`docs/ablation.md`](docs/ablation.md) |
| Ej. 4 · V7–V9 | ✅ Análisis de errores de la Versión 1 y un cambio por modo de falla: mensajes de contraste (V7), respaldo dirigido para R2 (V8) y plan adaptado a la entidad del mensaje (V9, **Versión 2**) ([`docs/ablation.md` §3](docs/ablation.md)). Después, V10 suma lo que ve visión en la captura (remitente, insignia, descripción) y da peso a los hechos del contexto del usuario |
| Reporte y UI | ✅ `make report` (tabla de la ablación) y UI local: `python -m safefish.ui` (usa `configs/v10.yaml`) |

### Modelos

| Etapa | Modelo |
|---|---|
| Razonador | Claude Sonnet 5.5 (`claude-sonnet-5-5`, esfuerzo `low`) |
| Lectura de capturas | Claude Sonnet 5.5 |
| Multi-query, reranking, planner | Claude Haiku 5.5 (`claude-haiku-5-5`) |
| Embeddings | `text-embedding-3-small` (OpenAI; Anthropic no tiene embeddings) |
| Juez de la cobertura del plan (evaluación) | Claude Sonnet 5.5 (`--judge-model`; `python -m eval.rejuzgar` re-juzga corridas viejas) |
| Normalización de la base (ingesta, una vez) | GPT-4.1-mini; la base es un snapshot fijo |

La clave de OpenAI va en `OPENAI_EMBEDDINGS_API_KEY` y solo la usan los embeddings. Los clientes
de chat de OpenAI leen `OPENAI_API_KEY`, que no hace falta salvo para reconstruir la base o
re-correr las corridas viejas con GPT-4.1. El cliente LLM elige el proveedor según el modelo.

### Probar la Fase 1 sobre fixtures (≈ USD 0,08)

```bash
make demo-fixture
make compare A=results/<run_v0> B=results/<run_v1-fixture>
```

### Correr V0 y V1 con datos reales

```bash
make data                                       # baja las fuentes públicas (idempotente)
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

### Correr V4–V9 (Fase 3 y Ej. 4) y armar la tabla

```bash
make eval CONFIG=configs/v4.yaml SPLIT=test JUDGE=1   # multi-query + RRF (mismo kb-v2)
make eval CONFIG=configs/v5.yaml SPLIT=test JUDGE=1   # + híbrido BM25 (desde chunks.jsonl)
make eval CONFIG=configs/v6.yaml SPLIT=test JUDGE=1   # + reranking listwise → Versión 1
make eval CONFIG=configs/v7.yaml SPLIT=test JUDGE=1   # + mensajes de contraste (análisis de errores)
make eval CONFIG=configs/v8.yaml SPLIT=test JUDGE=1   # + respaldo dirigido para R2
make eval CONFIG=configs/v9.yaml SPLIT=test JUDGE=1   # + plan adaptado a la entidad → Versión 2
make eval CONFIG=configs/v10.yaml SPLIT=test JUDGE=1  # + metadatos de la captura y contexto con peso (UI)
make report SPLIT=test                                 # → docs/ablation_tabla_test.md
make compare A=results/<run_v6> B=results/<run_v9>     # IC bootstrap + McNemar
```

Análisis, decisiones y fichas de V7–V9: [`docs/ablation.md`](docs/ablation.md). Demo:
`python -m safefish.ui` → http://127.0.0.1:8501. Por defecto usa `configs/v10.yaml` (la Versión 2 más lo que ve visión en la captura y el contexto con peso);
acepta texto o captura, un contexto opcional ("es de un amigo, me pide plata") y la situación del
usuario, que define qué guías arma el plan. El front ([`src/safefish/web/`](src/safefish/web/)) es
Preact + htm sin paso de build; marca las citas sobre la captura en el navegador con Tesseract.js
(necesita conexión para cargarlo desde el CDN; sin conexión, las citas se marcan solo en el texto).
Debajo del resultado, "Paso a paso del análisis" muestra cada etapa: la transcripción, las queries
del multi-query, el top de cada búsqueda densa y BM25 por query y colección, la fusión RRF, el
reranking con sus puntajes, el contexto que recibe el razonador, su salida, los cambios del
verificador, las guías y pasos del plan, y las llamadas al LLM con tokens, costo y tiempos. Lo arma
`safefish.trace`, que solo se activa en la UI: la evaluación no cambia.

En el pipeline de evaluación, el OCR con cajas usa el binario de Tesseract con el idioma `spa`
(Windows: `winget install UB-Mannheim.TesseractOCR`, con Spanish en "Additional language data";
Debian/Ubuntu: `apt install tesseract-ocr tesseract-ocr-spa`). Sin Tesseract el pipeline funciona
igual, pero las métricas de recuadros quedan vacías.

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
