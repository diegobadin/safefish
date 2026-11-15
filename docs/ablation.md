# Ablación V0 → V7: resultados, análisis y decisiones

> Golden v1: 125 casos (35 dev · 90 test), **pre-etiquetado con LLM, todavía sin revisión
> humana** ([`datos.md`](datos.md#golden-v1)). Los números de este documento son provisionales
> hasta que se revise el golden. Rehacer las métricas después de la revisión no cuesta API: las
> respuestas están en caché y solo cambian las etiquetas contra las que se comparan.
>
> Tablas generadas con `make report`: [`ablation_tabla_test.md`](ablation_tabla_test.md) y
> [`ablation_tabla_dev.md`](ablation_tabla_dev.md). Comparaciones pareadas en
> `results/compare_*.md` (`make compare`).

## 1. Tabla principal (test, n = 90)

| Versión | Cambio (única variable) | F1 | FPR neg. dif. | Acc. modalidad | F1 señales | Grounding | Hit-mod@k | MRR casos | USD/análisis |
|---|---|---|---|---|---|---|---|---|---|
| V0 | LLM sin RAG | 0.920 | 0.120 | **0.882** | 0.532 | 0.991 | — | — | 0.004 |
| V1 | Naive RAG: chunks de 800, 1 colección, top-6 | 0.898 | 0.120 | 0.804 | 0.517 | 0.995 | 0.091 | 0.091 | 0.007 |
| V2 | Normalización, 1 caso = 1 chunk, 4 colecciones, planner desde Guías | **0.929** | 0.080 | 0.843 | 0.569 | 0.988 | 0.911 | 0.824 | 0.014 |
| V3 | + verificador R1–R5 | **0.929** | 0.080 | 0.824 | 0.574 | **1.000** | 0.911 | 0.824 | 0.014 |
| V4 | + multi-query + RRF | 0.900 | 0.120 | 0.804 | 0.567 | 1.000 | 0.911 | 0.841 | 0.014 |
| V5 | + híbrido denso + BM25 | 0.917 | **0.040** | 0.765 | 0.558 | 1.000 | 0.956 | 0.707 | 0.014 |
| V6 | + re-ranking listwise → **Versión 1** | 0.920 | 0.120 | 0.784 | **0.592** | 1.000 | **0.978** | **0.906** | 0.017 |
| V7 | + evidencia contrastiva → **Versión 2** | **0.929** | 0.080 | 0.765 | 0.573 | 1.000 | 0.978 | 0.906 | 0.019 |

Otras métricas, constantes o casi constantes en todas las versiones:

- **ASR de inyección = 0.** Las 15 variantes adversariales se clasifican igual que su original.
- **F1 = 1.0 en el slice adversarial** en todas las versiones.
- **CER de capturas = 0.133** (WER 0.12) con `vision_v2`. Con `vision_v1` era 0.67, porque el
  extractor copiaba los avisos del sistema de WhatsApp como texto del remitente (ver §4).
- **Soporte de modalidad: 1.0 desde V3.** Toda modalidad reportada tiene al menos un caso que la
  respalde. En V0 vale 0 por construcción.
- **Kappa ponderado del veredicto: 0.74–0.76.**

Las latencias de `metrics.json` no se reportan: varias corridas salieron de caché.

### Significancia (bootstrap pareado, IC 95 %, y McNemar)

Con 90 casos, cada error mueve el F1 alrededor de 1 punto.

- **F1:** **ninguna** diferencia entre versiones vecinas es significativa. Todos los IC del Δ F1
  incluyen el 0; por ejemplo, V1→V2 da +0.031 [−0.023, 0.093] y V6→V7 da +0.009 [−0.024, 0.046].
- **Acc. modalidad:** la única diferencia significativa es la caída entre V0 y V6: −0.098 [−0.189, −0.021].

## 2. Qué se aprendió en cada paso

- **V0 es una cota inferior muy alta: F1 0.92.** Un LLM grande, con un prompt cuidadoso y salida
  estructurada, ya reconoce la mayoría de las estafas. El valor del RAG no aparece en el F1
  binario sino en lo que el veredicto puede **justificar**: casos documentados que lo respaldan,
  señales citadas, grounding y plan.
- **V1 no aporta.** Los chunks de 800 caracteres casi nunca traen un caso de la modalidad
  correcta (Hit-mod@k 0.09): en una sola colección, los 5.900 chunks de mensajes tapan a los 409
  de casos.
- **V2 es el salto de retrieval: Hit-mod@k pasa de 0.09 a 0.91 y MRR de 0.09 a 0.82.**
  - El barrido de solo retrieval aísla la causa. **V2a** (chunk por caso en **una** colección)
    da Hit-mod@k = 0.0 en test, porque los mensajes siguen tapando a los casos. Lo que mejora el
    retrieval son las **4 colecciones con `top_n` por colección**, no solo el chunking.
- **V3 lleva el grounding de citas a 1.000 y el soporte de modalidad a 1.0.**
  - El costo es acc. modalidad −0.02: la regla R2 baja a `no_determinada` las modalidades sin
    respaldo en la evidencia.
- **V4 (multi-query) sube algo el MRR (+0.017) pero no mejora la tarea.** El mensaje crudo ya es
  una buena query para esta KB.
- **V5 (híbrido) tiene el mejor FPR en negativos difíciles (0.04).**
  - Sube Hit-mod@k a 0.956, pero baja el MRR (0.84 → 0.71): BM25 mete arriba casos con términos
    exactos en común que no son de la modalidad correcta.
- **V6 (re-ranking) tiene el mejor retrieval (Hit-mod@k 0.978, MRR 0.906) y el mejor F1 de
  señales (0.592).** El re-ranking corrige el orden que había desarmado BM25.
  - Es la **Versión 1** (tag `v1-ej2`): la mejor configuración entre V3 y V6 en retrieval y en
    señales, sin perder F1.
  - Cuesta un 20 % más que V5 y casi duplica los tokens de entrada (17.8k contra 10.8k por
    análisis).

### La modalidad empeora con RAG (V0 0.88 → V6 0.78)

Es el resultado negativo más claro y el único significativo. La matriz de confusión de V6 contra
V0 muestra que los errores se concentran en el borde entre `phishing_bancario`,
`no_determinada` y `premio_sorteo_falso`:
- phishing de Netflix, iCloud o Microsoft: el pre-etiquetado lo marcó `phishing_bancario` "por
  cercanía" (ver dudas en `golden/procedencia.csv`);
- estafas 419 y románticas, que no tienen modalidad propia en el enum.

El RAG empuja hacia la modalidad de los casos recuperados, y el enum no tiene una categoría para
el phishing de credenciales de servicios no financieros. Es sobre todo un problema de
**definición del enum y de etiquetas** (causa e): la revisión humana del golden y una modalidad
`phishing_credenciales_servicio` lo atacarían mejor que cualquier cambio de retrieval.

## 3. Análisis de errores de la Versión 1 (V6, dev) y elección de V7

Errores de V6 en dev (9 de 35), con su causa raíz según §2.5:

| Modo de falla | Casos | Causa |
|---|---|---|
| Falsos positivos en negativos difíciles: alerta real de Mercado Libre; promo de DiDi/Spotify (texto y captura) | 3 | (b) El retrieval trae vecinos estafa parecidos y ningún legítimo parecido; el razonador decide sin contraste |
| `falso_familiar_cambio_numero` reportado como `no_determinada` | 3 | (a)/(d) Solo 3 casos de esa modalidad en la KB |
| Estafas fuera del enum (419, falso "informe de riesgo") como `levemente_sospechoso` | 2 | (d) Sin cobertura |
| Veredicto canónico distinto, con el binario correcto | 1 | — |

Se eligió el modo con mayor impacto según el costo asimétrico declarado: los **falsos positivos
en negativos difíciles** (FPR 0.23 en dev). Es el candidato que el plan tenía preparado.

### Ficha V7 · evidencia contrastiva

| | |
|---|---|
| **Hipótesis** | Si el contexto trae mensajes legítimos parecidos además de estafas parecidas, el razonador distingue un aviso real de una imitación por contraste y baja el FPR en negativos difíciles |
| **Cambio (única variable)** | `retrieval.mensajes_contrastivo: true`: la colección `mensajes` devuelve la mitad de los vecinos con `label=estafa` y la mitad con `label=legitimo` (filtro de metadata en las dos ramas, densa y BM25), intercalados |
| **Métrica objetivo** | FPR en negativos difíciles sin perder recall |
| **Resultado en test** | FPR 0.12 → **0.08** (1 de 25 negativos difíciles); recall igual (0.902); F1 +0.009 [−0.024, 0.046]; acc. modalidad −0.02; F1 señales −0.02; costo +12 % |
| **Resultado en dev** | FPR igual (0.23); F1 −0.03 |
| **Decisión** | **No concluyente.** La dirección es la esperada en test, pero la diferencia es de un caso, no se repite en dev y ningún IC excluye el 0. Se reporta como Versión 2 (tag `v2-ej4`) con esa salvedad. Lo que el error analysis señala como más prometedor es el golden (revisión humana) y la cobertura de `falso_familiar` |

## 4. Problemas encontrados y cómo se resolvieron

| Problema | Diagnóstico | Arreglo |
|---|---|---|
| CER 0.67 en capturas | El extractor transcribía avisos del sistema ("Esta empresa usa un servicio seguro de Meta…", "Leer más") como texto del remitente. La referencia tenía la PII redactada y la predicción no | `vision_v2` omite los avisos de la app. El CER se mide con las dos partes redactadas igual. Del slice `capturas` se excluyen las capturas con repeticiones. Resultado: CER 0.133 |
| V2–V6 con 0 de F1 en la primera corrida | Chroma 1.5 no persiste el HNSW de las colecciones con menos de 100 vectores (`batch_size` por defecto) y lo completa en la primera lectura; según qué proceso lea primero, el segmento queda roto ("Nothing found on disk") | `batch_size = sync_threshold = tamaño de la colección` en el build, y verificación del índice en un proceso aparte antes de escribir el manifest |
| Cobertura del plan 0.33 desde V2 (V0/V1: 0.87–0.94) | Había solo 13 guías, casi todas generales del Banco Nación. El planner elegía las mismas dos para todo y el plan omitía "no transferir" o "no entrar al link" | Las 24 páginas por modalidad entran también como guías: 13 → 42 guías en kb-v2. **Pendiente:** re-correr V2–V7 (planner y juez; el razonador sale de caché). Ver §5 |
| Cobertura de casos: 8 modalidades con menos de 5 casos | Las fuentes iniciales eran generales | 24 páginas nuevas, 3 por modalidad. Quedan 3 modalidades con 3–4 casos |

## 5. Pendientes

1. **Revisión humana del golden** (obligatoria según la guía). Las dudas de cada caso están en
   `golden/procedencia.csv`. Después de revisar: `python scripts/armar_golden.py ensamblar`,
   `make golden-check` y `make eval` de todas las versiones. Sale casi todo de caché.
2. **Re-correr con las guías por modalidad.** Los números de la tabla (`cobertura_plan`
   incluida) se midieron con kb-v2 de **13 guías** (hash `3626e259`). La kb-v2 actual tiene 42
   guías (`c8e84fef`). El razonador no ve las guías (`top_n.guias = 0`), así que el veredicto,
   las señales y el retrieval de casos no deberían cambiar; sí cambian el plan y su cobertura.
   El costo es solo planner y juez, del orden de 1 USD. Bloqueado por el límite de gasto del
   proyecto de OpenAI.
3. **Reconstruir kb-v1** (`make kb CONFIG=configs/v1.yaml`), que también quedó bloqueada por el
   límite de gasto, y re-correr V1 para que use las mismas fuentes que V2+.

## 6. Notas de diseño

- **Re-ranking con LLM y no con cross-encoder.** El plan proponía un cross-encoder. El entorno no
  tiene `torch` ni `onnxruntime`, y sumarlos rompe la reproducibilidad del `uv.lock` sin aportar
  un modelo multilingüe confiable para español rioplatense. El reranker es un LLM mini *listwise*
  (`rerank_v1`), que puntúa 0–3 cada candidato contra el mensaje, con T = 0 y caché. Cuesta unos
  0.3 centavos por análisis.
- **BM25 desde `chunks.jsonl`.** No requiere reconstruir la KB y usa exactamente los mismos
  chunks que el índice denso.
- **RRF con k = 60.** En V5 los pesos son denso 1 y BM25 1; no se barrieron por costo, y el
  barrido de solo retrieval queda como extensión barata.
