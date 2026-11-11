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

Con Poetry: `make eval RUN="poetry run"`.

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
