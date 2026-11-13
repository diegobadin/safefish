# ADR 002 — V1 con una sola colección, ruteada por tipo de documento

- Estado: aceptado · Fecha: 2026-11-11 · Fase: 1

## Contexto

V1 es *Naive RAG*: una colección, chunking recursivo, denso top-6 (§2.0). Pero el contrato
`RetrievalBundle` (y las métricas de retrieval) separan `casos`, `mensajes`, `senales` y `guias`.

## Decisión

- Cada fuente declara su `tipo` en `data/sources.yaml`; cada chunk hereda `doc_tipo`.
- En V1 todos los chunks van a **una** colección Chroma (`kb_v1`) y el top-6 es global.
- El retriever **rutea** cada chunk recuperado al campo del bundle según su `doc_tipo`. Si en
  el top-6 no aparece ningún caso, `casos` queda vacío: es justamente lo que V2 (4 colecciones
  con `top_n` por colección) viene a corregir, y queda medido en `items_promedio_por_tipo`.
- Las métricas de retrieval de casos (Hit@k, MRR, nDCG) se calculan sobre la sublista `casos`.
  La relevancia se deriva de la metadata (`modalidad`, `entidad_suplantada`), sin etiquetar ids.

## Ajustes al esqueleto de contratos del §1.5

- `Retriever.retrieve(m)` recibe la configuración en el constructor (se arma desde el YAML) en
  lugar de por parámetro.
- `Reasoner.reason(m, b, s, meta)` recibe el `RunMeta` inicial; el pipeline completa tokens,
  costo y latencias al final.
- `RunMeta` suma `modelos`, `reintentos_parseo` y `cache_hit`; `GoldenCase` suma `entidad`
  (opcional) para la relevancia graduada de nDCG.
- El LLM nunca genera `fuente_url`, `bbox` ni `meta`: el razonador usa un esquema reducido y
  esos campos se completan de forma determinística.

## Consecuencias

- La comparación V1 → V2 aísla el efecto de *chunk por caso + colecciones*; el barrido V2a
  (chunk por caso en 1 colección) sigue disponible cambiando solo `chunking.strategy`.
