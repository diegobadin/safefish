# Data card — SafeFish

> Completar a medida que Track A suma fuentes. Los números salen de
> `data/kb_snapshots/<kb>/manifest.json` y de `make golden-check`.

## KB

| Snapshot | kb_hash | Docs | Chunks | Por tipo | Fuentes |
|---|---|---|---|---|---|
| kb-v1 | — | — | — | — | — |
| kb-v2 | — | — | — | — | — |
| kb-v2a | — | — | — | — | — |

- **Fuentes y licencias:** cada descarga con URL, licencia, SHA-256 y fecha de captura en
  `data/raw/_descargas.json` (lo genera `make data` desde `data/descargas.yaml`). Qué entra a
  cada colección: [`datos.md`](datos.md).
- **Filtros aplicados:** idioma (es), deduplicación exacta y cercana (coseno TF-IDF ≥ 0.95), boilerplate.
- **PII:** redacción con placeholders tipados (`<TEL>`, `<CBU>`, `<ALIAS>`, `<EMAIL>`, `<DNI>`,
  `<TARJETA>`, `<CODIGO>`, `<NOMBRE>`, `<LINK_ACORTADO>`, rutas de URL → `<RUTA>`). La detección de
  nombres es heurística (disparadores + mayúscula): limitación conocida.
- **Normalización (V2+):** `normalizer_version` del manifest; documentos por método (plantilla /
  LLM / crudo), descartados, frases típicas descartadas y contactos redactados
  (`manifest.normalizacion`). Revisión manual de la muestra del 10 %: <n revisados, n ok, errores
  típicos>.
- **Cobertura:** casos por modalidad × canal (`make coverage`); modalidades bajo el umbral de 5:
  <lista>.
- **Auditoría de inyección:** chunks marcados (`manifest.auditoria_inyeccion`) y decisión tomada.
- **Sintéticos / traducidos:** <cantidad; solo en la KB, nunca en test>.

## Golden

| Split | n | Estafas | Legítimos | hard_negative | sin_evidencia | adversarial | capturas |
|---|---|---|---|---|---|---|---|
| dev | — | — | — | — | — | — | — |
| test | — | — | — | — | — | — | — |

- **Leakage:** split por `campaign_id`; auditoría de casi-duplicados test ↔ KB (coseno ≥ 0.92): <resultado>.
- **Anotación:** <quién anotó, quién revisó, kappa si hubo doble anotación>.

## Fixtures (no son datos de evaluación)

`tests/fixtures/` contiene una mini-KB **sintética** (34 docs con entidades ficticias: Banco
Ejemplo, BilleteraYa, CorreoExpress) y 8 casos de golden sintéticos para smoke tests. No se
reportan métricas sobre ellos.
