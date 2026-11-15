# Ablación V0 → V9: resultados, análisis y decisiones

> Golden v1 revisado: 125 casos (35 dev · 90 test). Todas las versiones se corrieron el
> 2026-11-15 sobre el mismo golden y la misma base (kb-v2: 59 casos, 3.412 mensajes, 15 señales,
> 42 guías; kb-v1 para V1). **Modelos:** Claude Sonnet 5.5 (razonador, lectura de capturas y juez
> del plan) y Claude Haiku 5.5 (multi-query, reranking y planner); OpenAI solo para los embeddings
> (`text-embedding-3-small`). La normalización de la base se hizo una vez, al construirla, con
> GPT-4.1-mini; la base es un snapshot fijo (hash en el manifest).
>
> Tablas generadas con `make report`: [`ablation_tabla_test.md`](ablation_tabla_test.md) y
> [`ablation_tabla_dev.md`](ablation_tabla_dev.md). Comparaciones pareadas en
> `results/compare_*.md` (`make compare`).

## 1. Tabla principal (test, n = 90)

| Etapa | Versión | Cambio (única variable) | F1 | FPR neg. dif. | Acc. modalidad | F1 señales | Soporte | Cobertura plan | Hit-mod@k | MRR | nDCG@k | USD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Base | V0 | LLM sin RAG | 0.905 | **0.040** | 0.784 | 0.587 | 0.00 | **0.923** | — | — | — | 0.007 |
| Pipeline RAG | V1 | Naive RAG: Recursive 800/120, 1 colección, top-6 | 0.887 | 0.120 | 0.804 | 0.633 | 0.08 | 0.897 | 0.086 | 0.086 | 0.034 | 0.011 |
| | V2 | Normalización, 1 caso = 1 chunk, 4 colecciones, plan desde guías | 0.896 | 0.080 | 0.824 | 0.672 | 0.95 | 0.436 | 0.844 | 0.774 | 0.579 | 0.019 |
| | V3 | + verificador R1–R5 | 0.896 | 0.080 | 0.784 | 0.672 | 1.00 | 0.420 | 0.844 | 0.774 | 0.579 | 0.019 |
| Advanced RAG | V4 | + query rewriting (multi-query) + RRF | **0.929** | 0.080 | 0.843 | 0.669 | 1.00 | 0.454 | 0.933 | 0.828 | 0.644 | 0.019 |
| | V5 | + híbrido denso + BM25 | 0.907 | 0.080 | 0.784 | 0.664 | 1.00 | 0.423 | 0.956 | 0.757 | 0.528 | 0.019 |
| | V6 | + reranking listwise con LLM → **Versión 1** | 0.920 | 0.120 | **0.863** | **0.672** | 1.00 | 0.469 | **0.978** | **0.878** | **0.764** | 0.020 |
| Iteración Ej. 4 | V7 | + mensajes de contraste (filtro por metadata) | **0.929** | 0.080 | 0.824 | 0.671 | 1.00 | 0.495 | 0.978 | 0.878 | 0.764 | 0.021 |
| | V8 | + respaldo dirigido para R2 | 0.929 | 0.080 | 0.824 | 0.671 | 1.00 | 0.495 | 0.978 | 0.878 | 0.764 | 0.021 |
| | V9 | + plan adaptado a la entidad → **Versión 2** | **0.929** | 0.080 | 0.824 | 0.671 | 1.00 | 0.846 | 0.978 | 0.878 | 0.764 | 0.022 |

V2a (barrido de solo recuperación: 1 caso = 1 chunk en **una** colección): Hit-mod@k 0.000 en
test y 0.067 en dev.

Constantes o casi constantes en todas las versiones:

- **Grounding de citas = 1.000** (R1 descarta las citas que no están literales en el mensaje).
- **ASR de inyección = 0**: las 15 variantes adversariales se clasifican igual que su original.
- **Contactos inventados en el plan: 1 en V0 y V1, 0 desde V2.**
- **CER de capturas = 0.111** (WER 0.098) en test con Claude Sonnet 5.5 (`vision_v2`).
- Kappa ponderado del veredicto 0.78–0.82; recall de estafas 0.84–0.90.

### Significancia (bootstrap pareado, IC 95 %, y McNemar)

Con 90 casos, cada error mueve el F1 alrededor de 1 punto.

- **F1:** ninguna diferencia es significativa (V0 → V6: +0.015 [−0.029, +0.063]; V0 → V9: +0.024
  [−0.023, +0.075]).
- **RAG vs sin RAG (V0 → V6):** recall +0.059 [0.0, +0.131], acc. modalidad +0.078 [0.0, +0.175]
  (McNemar 5 a favor, 1 en contra, p = 0.22). Tendencia a favor, sin significancia estricta.
- **Cobertura del plan:** V0 → V6 −0.454 [−0.541, −0.364] y V6 → V9 **+0.377 [+0.287, +0.464]**,
  ambas significativas. V9 queda −0.077 [−0.138, −0.021] por debajo de V0, que escribe el plan
  libre y sin fuente.
- **Retrieval:** V1 → V2 sube Hit-mod@k +0.76 y MRR +0.69; V5 → V6 sube MRR +0.12 y nDCG +0.24.

## 2. Qué se aprendió en cada paso

**Pipeline RAG completo (V0 → V3).**

- **V0 es una cota alta: F1 0.905.** Un LLM grande con salida estructurada ya reconoce la mayoría
  de las estafas. El valor del RAG no aparece en el F1 sino en lo que el veredicto puede
  **justificar**: soporte de modalidad 0 → 1.0, F1 de señales 0.59 → 0.67 y 0 contactos inventados.
- **V1 (naive RAG) casi no recupera:** Hit-mod@k 0.086. En una sola colección, los chunks de
  mensajes tapan a los de casos.
- **V2 es el salto de recuperación: Hit-mod@k 0.09 → 0.84, MRR 0.09 → 0.77.** V2a (chunk por caso
  en una colección) da 0.0: lo que destraba la recuperación son las **4 colecciones con `top_n`
  por colección**, no el chunking solo.
- **V3 lleva el soporte de modalidad a 1.0** (faithfulness). Cuesta acc. modalidad 0.824 → 0.784:
  R2 baja a `no_determinada` las modalidades sin un caso que las respalde.

**Advanced RAG (V4 → V6).**

- **V4 (query rewriting):** Hit-mod@k 0.84 → 0.93 y MRR 0.77 → 0.83. Es la versión con mejor F1
  junto con las de la iteración.
- **V5 (híbrido BM25):** Hit-mod@k 0.96, pero MRR 0.83 → 0.76 y nDCG 0.64 → 0.53: BM25 sube
  casos con términos en común que no son de la modalidad correcta.
- **V6 (reranking listwise):** el mejor retrieval, Hit-mod@k 0.978, MRR 0.878 y nDCG 0.764; acc.
  modalidad 0.863, la más alta de la ablación. **Versión 1** (tag `v1-ej2`).

### Resultados negativos

- **Noise robustness:** con RAG suben los falsos positivos en negativos difíciles (V0 0.04 → V6
  0.12, 3 de 25): el contexto trae estafas parecidas y el razonador marca como estafa un
  legítimo similar. V7 lo baja a 0.08.
- **Faithfulness vs acierto:** el verificador (V3) baja la modalidad 0.04 porque prefiere "no
  determinada" antes que afirmar sin respaldo.
- **BM25 desordena** hasta que entra el reranking (V5).
- **La cobertura del plan cae de 0.92 (V0) a 0.42–0.47 (V2–V6):** el plan sale de guías
  oficiales, mayormente de prevención general, y el planner no veía el mensaje. Se resolvió en la
  iteración (V9).

## 3. Iteración del Ejercicio 4: análisis de errores de la Versión 1

Errores de V6 (dev y test):

| Modo de falla | Evidencia | Causa |
|---|---|---|
| Falsos positivos en negativos difíciles | 3 de 25 en test (T-0018, T-0054, T-0059), 1 de 13 en dev | El retrieval trae vecinos estafa parecidos y ningún legítimo parecido: el razonador decide sin contraste |
| Modalidad bajada a `no_determinada` | 4 de los 7 errores de modalidad en test (falso soporte, phishing, deuda) | R2 exige un caso recuperado de esa modalidad |
| Plan genérico o con contactos de otra entidad | Cobertura 0.47; 23 de 65 planes con cobertura ≤ 0.33. Ejemplo: un mail de ARCA en el plan de una deuda de otra empresa | El planner no veía el mensaje y copiaba pasos de las dos guías generales del Banco Nación |

Se probó un cambio por modo de falla, de a uno.

### Ficha V7 · mensajes de contraste

| | |
|---|---|
| **Hipótesis** | Si el contexto trae mensajes legítimos parecidos además de estafas parecidas, el razonador distingue un aviso real de una imitación por contraste |
| **Cambio** | `retrieval.mensajes_contrastivo: true`: la colección `mensajes` devuelve la mitad de los vecinos con `label=estafa` y la mitad con `label=legitimo` (filtro de metadata en las ramas densa y BM25), intercalados |
| **Resultado** | Test: FPR 0.12 → 0.08, F1 0.920 → 0.929, acc. modalidad −0.039 (2 casos, no significativo). Dev: F1 0.865 → 0.923, recall 0.80 → 0.90 |
| **Decisión** | Se adopta |

### Ficha V8 · respaldo dirigido para R2

| | |
|---|---|
| **Cambio** | `postprocess.soporte_dirigido_margen: 0.06`: si la modalidad del razonador no está entre los casos recuperados, se busca el caso más parecido de esa modalidad y se acepta como respaldo si su similitud queda a ≤ 0.06 del mejor caso de toda la colección. Calibrado sobre las 60 estafas con modalidad del golden: con 0.05 entra el 92 % de las modalidades correctas y el 15 % de las incorrectas; con 0.07, el 100 % y el 28 % |
| **Resultado** | **Sin efecto**: no actuó en ningún caso de dev ni de test. Con Claude, cuando R2 baja una modalidad, la del razonador tampoco queda cerca de un caso de esa modalidad. Con GPT-4.1 había actuado en 1 de 125 casos |
| **Decisión** | Se mantiene (no cambia ningún resultado y cubre el caso real que lo motivó), documentado como resultado nulo |

### Ficha V9 · plan adaptado a la entidad del mensaje → Versión 2

| | |
|---|---|
| **Cambio** | `generation.planner_prompt: planner_v2`. El planner ve el mensaje como dato y usa las guías como estructura de cómo proceder. Si una guía es de otra entidad, su nombre y sus contactos pasan a "la entidad oficial" / "los canales oficiales de la entidad". Una adaptación determinística en el planner lo garantiza para todas las versiones con guías (quita contactos de entidades que el mensaje no nombra y contactos que no están en la guía); R5 acepta el nombre de un organismo que nombra el mensaje, nunca un contacto sacado de él |
| **Resultado** | Cobertura del plan 0.495 → 0.846 en test (dev 0.433 → 0.820); frente a la Versión 1, +0.377 [+0.287, +0.464]. Ningún otro cambio: mismas métricas de clasificación y retrieval, 0 contactos inventados. Costo +0.001 USD por análisis |
| **Decisión** | **Versión 2** (tag `v2-ej4`) y configuración de la UI |

**Versión 1 → Versión 2 (test):** F1 0.920 → 0.929, FPR 0.12 → 0.08, F1 señales igual, cobertura
del plan 0.47 → 0.85, acc. modalidad 0.863 → 0.824 (−2 casos, no significativo).

## 4. Problemas encontrados y cómo se resolvieron

| Problema | Diagnóstico | Arreglo |
|---|---|---|
| CER 0.67 en capturas | El extractor transcribía avisos del sistema ("Esta empresa usa un servicio seguro de Meta…", "Leer más") como texto del remitente. La referencia tenía la PII redactada y la predicción no | `vision_v2` omite los avisos de la app. El CER se mide con las dos partes redactadas igual. Del slice `capturas` se excluyen las capturas con repeticiones. Resultado: CER 0.133 con GPT-4.1 y 0.111 con Claude |
| V2–V6 con 0 de F1 en la primera corrida | Chroma 1.5 no persiste el HNSW de las colecciones con menos de 100 vectores (`batch_size` por defecto) y lo completa en la primera lectura; según qué proceso lea primero, el segmento queda roto ("Nothing found on disk") | `batch_size = sync_threshold = tamaño de la colección` en el build, y verificación del índice en un proceso aparte antes de escribir el manifest |
| Cobertura del plan baja desde V2 | Había solo 13 guías, casi todas generales; el planner no veía el mensaje | 13 → 42 guías en kb-v2 (24 páginas por modalidad) y planner adaptado a la entidad (V9) |
| Cobertura de casos: 8 modalidades con menos de 5 casos | Las fuentes iniciales eran generales | 24 páginas nuevas, 3 por modalidad. Quedan 3 modalidades con 3–4 casos |
| Presupuesto de API | La primera ablación con GPT-4.1 y una re-corrida con GPT-5 superaron el crédito de la cátedra | Ablación final con Claude; OpenAI solo para embeddings. Costo de la ablación completa (10 versiones, test y dev, con juez): unos USD 13 |

## 5. Comentario: el modelo generador

La primera ablación se hizo con GPT-4.1 (razonador y capturas) y GPT-4.1-mini (multi-query,
rerank, planner y juez); después se probó GPT-5 con razonamiento mínimo y se pasó a Claude. Las
corridas siguen en `results/` (fechas 14 y 15/11 antes de las 19 h UTC).

- Con GPT-4.1, el RAG **bajaba** el acierto de modalidad (V0 0.863 → V6 0.765, significativo); con
  Claude sube (0.784 → 0.863). Con el mismo pipeline (V6), Claude acierta la modalidad 0.86 contra
  0.77 de GPT-4.1 y el F1 de señales 0.67 contra 0.59.
- GPT-5 con razonamiento mínimo rindió peor que GPT-4.1 en modalidad y señales y, sin RAG, inventó
  22–23 contactos en los planes de test (GPT-4.1: 2; Claude: 1).
- La arquitectura no cambió: el cliente LLM elige el proveedor según el modelo. La elección se hizo
  midiendo en el golden, no por benchmarks.

## 6. Trabajo futuro

1. Más mensajes legítimos de contraste en la base (los falsos positivos en negativos difíciles son
   el modo de falla que queda).
2. Modalidad `phishing_credenciales_servicio` en el enum y casos para ella.
3. Más casos de las modalidades con menos cobertura (falso familiar: 3; inversión y empleo: 4).
4. Barrido de solo retrieval de los pesos de RRF y de `top_n`, que no cuesta generación.
5. Batch API para las corridas de evaluación (50 % menos de costo) y observabilidad con LangSmith.

## 7. Notas de diseño

- **Reranking con LLM y no con cross-encoder.** El entorno no tiene `torch` ni `onnxruntime`, y
  sumarlos rompe la reproducibilidad del `uv.lock` sin aportar un modelo multilingüe confiable
  para español rioplatense. El reranker es un LLM chico *listwise* (`rerank_v1`, Claude Haiku 5.5)
  que puntúa 0–3 cada candidato contra el mensaje, con caché.
- **BM25 desde `chunks.jsonl`.** No requiere reconstruir la base y usa exactamente los mismos
  chunks que el índice denso.
- **RRF con k = 60.** En V5 los pesos son denso 1 y BM25 1; no se barrieron, y el barrido de solo
  retrieval queda como extensión barata.
- **Claude no acepta temperatura 0.** La reproducibilidad queda a cargo de la caché de respuestas
  (clave: modelo, esfuerzo, esquema y mensajes) y del snapshot de la base.
