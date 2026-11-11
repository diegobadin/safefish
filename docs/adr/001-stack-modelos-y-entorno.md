# ADR 001 — Stack, modelos pinneados y entorno

- Estado: aceptado · Fecha: 2026-11-11 · Fase: F0

## Contexto

El plan (§1.2) fija LangChain + OpenAI + ChromaDB, temperatura 0, seed y snapshots pinneados
para que las corridas de la ablación sean comparables.

## Decisión

| Componente | Elección | Por qué |
|---|---|---|
| Razonador (y visión, Fase 2) | `gpt-4.1-2025-04-14` | Snapshot fechado, acepta `temperature=0` y `seed`, salida estructurada estricta y visión. Los modelos de razonamiento (familia GPT-5) no aceptan `temperature`, lo que rompe el requisito de reproducibilidad del §4.2 |
| Modelo *mini* (tareas auxiliares) | `gpt-4.1-mini-2025-04-14` | Misma familia, mismo control de muestreo, ~5× más barato |
| Acceso a modelos | API de OpenAI vía `langchain-openai`, credencial `OPENAI_API_KEY` | Expone `seed`, salida `json_schema` estricta y `usage` por llamada, que alimenta el registro de tokens y costo |
| Embeddings | `text-embedding-3-small` | Multilingüe y barato (§1.2) |
| Vector DB | ChromaDB 1.5 persistente, distancia coseno | Sin infraestructura; filtros de metadata |
| Orquestación | `langchain-core` 1.8 + `langchain-openai` + `langchain-chroma` + `langchain-text-splitters` | Paquetes de integración actuales |
| Loaders | `langchain-community` 0.4 (`WebBaseLoader`, `PyPDFLoader`, `CSVLoader`, `TextLoader`) | Cubren los cuatro formatos de fuente (web, PDF, CSV, texto) y devuelven `Document` con metadata. El paquete está en *sunset*: si se rompe, se reemplaza por bs4/pypdf/csv directos sin tocar el contrato `Document` |
| Entorno | `pyproject.toml` PEP 621 con backend `poetry-core` | Se instala con `uv sync` o con `poetry install` (Poetry ≥ 2.2). El lockfile versionado es `uv.lock` |
| Python | ≥ 3.12 (desarrollado en 3.14) | |

Los modelos son parámetros del YAML (`generation.reasoner`), así que cambiarlos es una variable
de ablación más y nunca un cambio de código.

## Consecuencias

- `metrics.json` registra el snapshot del modelo; un cambio de modelo exige recorrer el golden.
- Las salidas del LLM y los embeddings se cachean por `(modelo, parámetros, prompt)` en `.cache/`
  (§3.3): re-correr la evaluación es casi gratis y determinístico.
- La tabla de precios de `observability.py` hay que revisarla contra la página de OpenAI antes de
  reportar costos.
