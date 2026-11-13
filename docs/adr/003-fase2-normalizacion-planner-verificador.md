# ADR 003 — Fase 2: normalización, 4 colecciones, planner desde Guías y verificador

- Estado: propuesto · Fecha: 2026-11-13 · Fase: 2 (V2, V3, M-EXT)

## Contexto

El §2.3 del plan define V2 (preprocesamiento estructurado + planner) y V3 (verificador R1–R5),
más la extracción multimodal. Este ADR registra las decisiones de implementación que el plan
deja abiertas o que ajustan el esqueleto de contratos del §1.5.

## Decisiones

### Normalización (M-ING.3)

- `schemas.py` suma `CaseDoc`, `MessageDoc`, `SignalDoc` y `GuideDoc` (cambio aditivo). La
  metadata indexada sale de estos modelos: es el contrato entre ingesta y retrieval/planner.
- `ingest.normalize.mode: auto` (default): **plantilla primero, LLM si hace falta**. Los casos
  curados con secciones `Descripción:/Frases típicas:/Mecanismo:` y las guías con pasos
  numerados no pasan por el LLM (más barato, determinístico y testeable sin API). Las fuentes
  web/PDF sin estructura van al LLM *mini* con salida estructurada.
- Mensajes (CSV etiquetado) y señales (taxonomía) son siempre determinísticos. Un mensaje sin
  `label` se descarta: no aporta evidencia kNN.
- Controles sobre la salida del LLM: las frases típicas que no están en la fuente (fuzzy
  < 0.85) se descartan; los contactos de los pasos de una guía que no están en la fuente se
  reemplazan; la metadata curada (modalidad, canal, situación) manda sobre la del LLM y las
  discrepancias van al manifest.
- `normalizer_version = <modelo>:<prompt>:<hash>` por documento (o `template` / `crudo`).
- Muestra del 10 % para revisión manual en `normalizacion_revision.jsonl`, priorizando lo
  normalizado por LLM.

### Índice y retrieval

- `index.layout`: `single` (V1, V2a) o `per_tipo` (V2+). Con `per_tipo` se crean hasta cuatro
  colecciones (`<snapshot>_casos`, …) y el retriever pide `top_n` por colección, embebiendo la
  query una sola vez. El pipeline falla si el layout del config no coincide con el del snapshot.
- **V2a** (barrido de solo retrieval) = chunk por caso en una colección con top-6 global: aísla
  "1 caso = 1 chunk" de "4 colecciones" (§2.0).
- Las métricas de retrieval de casos usan `k = top_n.casos` en `per_tipo`.

### Planner (M-GEN.2)

- Selección **determinística** por metadata sobre `chunks.jsonl`, por niveles: exacta
  (modalidad + situación) → modalidad (guía sin situación) → general de la situación → general
  → misma modalidad con otra situación → búsqueda densa en `guias` (solo si no hubo ninguna).
  Cada guía usada registra su `via`.
- Si el usuario no informa la situación se asume `no_respondio` (`situacion_asumida: true`).
- El planner **no ve el texto del mensaje**: solo veredicto, modalidad, situación e ids de
  señales. Una inyección en el mensaje no llega a esta etapa.
- Los pasos con `guide_id` que no es de las guías seleccionadas se descartan. Si no queda
  ninguno (o el LLM falla), se usa el **plan extractivo** (pasos literales).
- En V2 las guías no van al contexto del razonador (`top_n.guias: 0`). El **razonador y su
  prompt no cambian** respecto de V1: su plan se reemplaza por el del planner. Así V1 → V2 no
  suma una variable de prompt (el Anexo B sugiere un esquema sin plan desde V2; se puede probar
  como iteración aparte si el costo en tokens importa).

### Verificador (M-POST.1)

- Orden con planner: `reason → R1–R4 → plan → R5`. La modalidad final (post-R2) decide qué
  guías se usan, y R5 controla el plan ya armado. Las reglas son idempotentes.
- R1 descarta citas que no están en el texto (fuzzy ≥ 0.9), deduplica señales y, en capturas,
  alinea la cita aceptada a una bbox OCR.
- R2 además descarta `casos_similares` con ids que no están en la evidencia.
- R4 toma la severidad de la taxonomía; `transferencia_a_tercero` cuenta como crítica **solo
  con `urgencia_temporal`** ("transferencia a tercero con urgencia", §2.3).
- R5: contactos (teléfono, URL, mail) ausentes de la guía del paso se reemplazan por un aviso;
  un paso que nombra un organismo ausente de la guía (lista curada en `textutils.ORGANISMOS`)
  se elimina; si el plan queda vacío, se usan los pasos literales de la primera guía.
- Cada intervención es un evento (`regla`, `accion`, `detalle`) en `predictions.jsonl`; la
  salida previa a las reglas se guarda como `pred_pre_verify` para medir grounding antes y
  después de R1 y contactos antes y después de R5.

### Extracción multimodal (M-EXT)

- LLM con visión → segmentos por emisor; el texto canónico son los segmentos del remitente.
  Entidades por regex sobre ese texto. Con texto e imagen a la vez, manda la imagen.
- OCR con Tesseract (`pytesseract`, preprocesamiento: grises, upscale, Otsu). Tesseract es un
  binario externo: si no está, el pipeline sigue sin OCR y las señales van sin recuadro.
- `postprocess/annotate.py` dibuja las bbox de las señales verificadas.

### Métricas

- `contactos_no_respaldados` ("contactos alucinados", §4.3) se mide contra **la guía fuente
  del paso** (o todas las guías del contexto si el paso no cita una), ya no contra el mensaje:
  un plan que deriva al teléfono del propio mensaje es exactamente el daño que R5 evita.
- Nuevas: `grounding_citas_pre_r1`, `contactos_no_respaldados_pre_r5`, `eventos_verificador`,
  `veredictos_modificados`, métricas del planner, `cobertura_plan` (juez LLM, `--judge-plan`) y
  métricas de extracción (CER/WER, F1 de entidades, % de señales con bbox).

### Ajustes tras la revisión de código y de seguridad

- **Contactos:** `find_contacts` es un detector *amplio* (cualquier TLD y dominios IDN, URLs
  con esquema de cualquier forma, IPs, teléfonos con puntos, números cortos como `134` o
  `*611`; un contacto contenido en otro no se cuenta dos veces): un falso positivo solo
  redacta de más. El `organismo` de la metadata de una guía cuenta para R5 solo como nombre de
  organismo, nunca para respaldar un contacto. La
  comparación (`contacto_respaldado`) es estructurada: mails exactos, URLs por host + prefijo
  de ruta, teléfonos por igualdad tras quitar solo prefijos de país/troncales (la
  característica de área cuenta). R5 también limpia contactos de `explicacion` y `por_que` que
  no están en el mensaje ni en la evidencia.
- **Delimitadores:** todo texto no confiable (mensaje, fuente, KB) se inserta con los ángulos
  escapados y los atributos sin comillas: no puede abrir ni cerrar tags del prompt.
- **R1:** citas de < 5 caracteres normalizados se descartan (`cita_trivial`) y la subcadena
  exacta respeta límites de palabra ("cambio" no está en "intercambio"). **R3** cuenta señales
  *distintas*. **R5** toma el `organismo` de la metadata de la guía como parte de la fuente.
- **Plantilla de guías:** las líneas después del primer paso son continuación (el texto ya
  viene sin sangría) y más de 400 caracteres antes del primer paso mandan la fuente al LLM.
  El planner ve el texto completo de la guía (contexto + pasos), el mismo que verifica R5.
- **Normalizador:** metadata fuera del enum descarta el documento (`metadata_invalida`) sin
  abortar el build; los contactos inventados se redactan también en `contexto`,
  `descripcion` y `mecanismo`.
- **Snapshot:** `kb_hash` incluye la metadata (la curación de modalidad/situación cambia el
  hash). La muestra de revisión es el 10 % de casos + guías, priorizando lo hecho por LLM.
- **Barrido V2a:** `retrieval.k_eval: 4` en V1/V2a/V2/V3 para medir los casos con el mismo k
  aunque cambie el layout, y `make eval … RETRIEVAL_ONLY=1` (sin generación: solo embeddings).
- **Métricas:** `GoldenCase.citas_bbox` y `localizacion_iou50` (§4.3); el juez del plan
  registra tokens, costo y fallidos en `metrics.json → juez_plan`.
- **Concurrencia con Chroma:** la primera consulta a una colección carga el índice HNSW de
  forma perezosa y no es segura entre hilos (el arnés con `workers > 1` fallaba de forma
  intermitente con "Nothing found on disk"). Las colecciones de un snapshot se abren con un
  único cliente y toda consulta pasa por `concurrency.CHROMA_LOCK` (milisegundos).

## Consecuencias

- El `config_hash` de V0/V1 cambia porque el config suma secciones con defaults
  (`index`, `extraction`, campos del planner). No había corridas oficiales todavía.
- El gate G2 (V2 y V3 medidos, grounding post-R1 = 100 %, JSON válido = 100 %, tag
  `v3-pipeline`) queda bloqueado por los datos reales, igual que G0/G1.
