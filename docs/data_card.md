# Data card — SafeFish

> Completar a medida que Track A suma fuentes. Los números salen de
> `data/kb_snapshots/<kb>/manifest.json` y de `make golden-check`.

## KB

| Snapshot | kb_hash | Docs | Chunks | Por tipo | Fuentes |
|---|---|---|---|---|---|
| kb-v1 | — | — | — | — | — |

- **Fuentes y licencias:** <por fuente: URL, organismo, licencia o términos de uso, fecha de captura>
- **Filtros aplicados:** idioma (es), deduplicación exacta y cercana (coseno TF-IDF ≥ 0.95), boilerplate.
- **PII:** redacción con placeholders tipados (`<TEL>`, `<CBU>`, `<ALIAS>`, `<EMAIL>`, `<DNI>`,
  `<TARJETA>`, `<CODIGO>`, `<NOMBRE>`, `<LINK_ACORTADO>`, rutas de URL → `<RUTA>`). La detección de
  nombres es heurística (disparadores + mayúscula): limitación conocida.
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
