# Ablación V0 → V8: resultados, análisis y decisiones

> Golden v1 revisado: 125 casos (35 dev · 90 test). Todas las versiones se corrieron el
> 2026-11-15 sobre el mismo golden y la misma KB (kb-v2 con 59 casos y 42 guías; kb-v1
> reconstruida desde las mismas fuentes).
>
> Tablas generadas con `make report`: [`ablation_tabla_test.md`](ablation_tabla_test.md) y
> [`ablation_tabla_dev.md`](ablation_tabla_dev.md). Comparaciones pareadas en
> `results/compare_*.md` (`make compare`).

## 1. Tabla principal (test, n = 90)

| Versión | Cambio (única variable) | F1 | FPR neg. dif. | Acc. modalidad | F1 señales | Grounding | Cobertura plan | Hit-mod@k | MRR casos | USD/análisis |
|---|---|---|---|---|---|---|---|---|---|---|
| V0 | LLM sin RAG | 0.920 | 0.120 | **0.863** | 0.525 | 0.991 | **0.936** | — | — | 0.004 |
| V1 | Naive RAG: chunks de 800, 1 colección, top-6 | 0.898 | 0.120 | 0.804 | 0.516 | 0.995 | 0.874 | 0.086 | 0.086 | 0.007 |
| V2 | Normalización, 1 caso = 1 chunk, 4 colecciones, planner desde guías | **0.929** | **0.080** | 0.804 | 0.587 | 0.988 | 0.436 | 0.867 | 0.780 | 0.012 |
| V3 | + verificador R1–R5 | **0.929** | **0.080** | 0.784 | **0.591** | **1.000** | 0.428 | 0.867 | 0.780 | 0.012 |
| V4 | + multi-query + RRF | 0.900 | 0.120 | 0.784 | 0.562 | 1.000 | 0.413 | 0.911 | 0.811 | 0.013 |
| V5 | + híbrido denso + BM25 | 0.907 | **0.080** | 0.765 | 0.565 | 1.000 | 0.426 | 0.956 | 0.696 | 0.013 |
| V6 | + re-ranking listwise → **Versión 1** | 0.920 | 0.120 | 0.765 | 0.587 | 1.000 | 0.420 | **0.978** | **0.876** | 0.016 |
| V7 | + evidencia contrastiva | 0.920 | 0.120 | 0.745 | 0.579 | 1.000 | 0.423 | 0.978 | 0.876 | 0.018 |
| V8 | + respaldo dirigido para R2 → **Versión 2** | 0.920 | 0.120 | 0.745 | 0.579 | 1.000 | 0.423 | 0.978 | 0.876 | 0.018 |

V2a (barrido de solo retrieval: 1 caso = 1 chunk en **una** colección): Hit-mod@k 0.000 en test
y 0.067 en dev.

Otras métricas, constantes o casi constantes en todas las versiones:

- **ASR de inyección = 0.** Las variantes adversariales se clasifican igual que su original.
- **CER de capturas = 0.133** (WER 0.12) con `vision_v2`. Con `vision_v1` era 0.67 (ver §4).
- **Soporte de modalidad: 1.0 desde V3** (V0: 0 por construcción; V1: 0.07).
- **Contactos no respaldados en el plan: 2 en V0 y V1, 0 desde V3** (R5).
- **Kappa ponderado del veredicto: 0.74–0.76.**

Las latencias de `metrics.json` no se reportan: la mayoría de las llamadas salieron de caché.

### Significancia (bootstrap pareado, IC 95 %, y McNemar)

Con 90 casos, cada error mueve el F1 alrededor de 1 punto.

- **F1:** ninguna diferencia es significativa. V1→V2 da +0.031 [−0.023, 0.093] (McNemar p = 0.45);
  V0→V6, 0.000 [−0.046, 0.048].
- **Acc. modalidad:** la única diferencia significativa es la caída entre V0 y V6, −0.098
  [−0.189, −0.021]: 5 casos que V0 acertaba y V6 no, ninguno al revés.
- **Retrieval:** V1→V2 sube Hit-mod@k +0.78 y MRR +0.69; V5→V6 sube MRR +0.18.

## 2. Qué se aprendió en cada paso

- **V0 es una cota inferior muy alta: F1 0.92.** Un LLM grande, con un prompt cuidadoso y salida
  estructurada, ya reconoce la mayoría de las estafas. El valor del RAG no aparece en el F1
  binario sino en lo que el veredicto puede **justificar**: casos documentados que lo respaldan
  (soporte 0 → 1.0), señales citadas (F1 señales 0.52 → 0.59) y citas siempre literales.
- **V1 no aporta.** Los chunks de 800 caracteres casi nunca traen un caso de la modalidad
  correcta (Hit-mod@k 0.09): en una sola colección, los miles de chunks de mensajes tapan a los
  de casos.
- **V2 es el salto de retrieval: Hit-mod@k 0.09 → 0.87 y MRR 0.09 → 0.78.**
  - V2a (chunk por caso en **una** colección) da Hit-mod@k 0.0 en test. Lo que mejora el
    retrieval son las **4 colecciones con `top_n` por colección**, no solo el chunking.
- **V3 lleva el grounding de citas a 1.000, el soporte de modalidad a 1.0 y los contactos
  inventados a 0.** Cuesta acc. modalidad −0.02: R2 baja a `no_determinada` una modalidad sin
  respaldo en la evidencia.
- **V4 (multi-query) sube Hit-mod@k a 0.91 y MRR a 0.81, pero no mejora la tarea.**
- **V5 (híbrido) lleva Hit-mod@k a 0.956 pero baja el MRR a 0.70:** BM25 sube casos con términos
  exactos en común que no son de la modalidad correcta.
- **V6 (re-ranking) tiene el mejor retrieval: Hit-mod@k 0.978 y MRR 0.876** (+0.18 sobre V5).
  Es la **Versión 1** (tag `v1-ej2`). Cuesta un 25 % más que V5 por las llamadas del reranker.

### Resultados negativos

**La modalidad empeora con RAG (V0 0.86 → V6 0.76).** Es la única diferencia significativa. El
RAG empuja hacia la modalidad de los casos recuperados, y los errores se concentran en el borde
entre `phishing_bancario`, `no_determinada` y `premio_sorteo_falso`: phishing de servicios no
financieros (Netflix, iCloud) y estafas sin modalidad propia en el enum (419, románticas). Es
sobre todo un problema de **definición del enum**: una modalidad `phishing_credenciales_servicio`
lo atacaría mejor que cualquier cambio de retrieval.

**La cobertura del plan cae de 0.94 (V0) a 0.42 (V2+).** En V0 y V1 el plan lo escribe el
razonador libremente: cubre los pasos esperados, pero inventa contactos (2 en test). Desde V2 el
plan sale solo de las guías oficiales: no inventa nada (0 contactos), pero las guías son
mayormente de prevención general ("verificá la identidad", "no compartas claves") y no traen los
pasos específicos que espera el golden ("no pagues la tarifa de envío", "no entres al link"). En
test, 20 de 65 planes se arman con una sola guía general, con cobertura 0.33. Es un **trade-off
entre grounding y cobertura**; mejorarlo requiere guías por modalidad y situación más
específicas, o permitir que el planner complete pasos genéricos marcados como sugerencia.

## 3. Análisis de errores de la Versión 1 (V6, dev) y elección de V7

Errores de V6 en dev (9 de 35), con su causa raíz según §2.5. El análisis se hizo con el golden previo a la revisión; los modos de falla siguen presentes con el revisado (FPR en negativos difíciles 0.23 en dev):

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
| **Resultado en test** | FPR igual (0.12); F1 igual (0.920); acc. modalidad −0.02 (1 caso); costo +13 %. En la corrida con el golden anterior el FPR había bajado a 0.08 por un solo caso, que con el golden revisado no se repite |
| **Resultado en dev** | Sin cambios respecto de V6 |
| **Decisión** | **Sin efecto medible.** El contraste no cambia las decisiones del razonador en este golden. Se mantiene en la línea porque no empeora el binario y deja la evidencia más equilibrada para la UI; la Versión 2 es V8 |

### Ficha V8 · respaldo dirigido para R2

| | |
|---|---|
| **Problema** | R2 baja a `no_determinada` modalidades correctas cuando ningún caso recuperado es de esa modalidad. Pasa con falso familiar, la modalidad con menos casos (3): D-0023 y D-0029 en V6, y una captura real de un contacto que pide plata para pagarle a un "vendedor", donde el retrieval trajo casos de compraventa |
| **Cambio (única variable)** | `postprocess.soporte_dirigido_margen: 0.06`. Si la modalidad del razonador no está entre los casos recuperados, antes de R2 se busca el caso más parecido de esa modalidad. Se acepta como respaldo, y se cita, si su similitud queda a ≤ 0.06 del mejor caso de toda la colección |
| **Calibración** | Sobre las 60 estafas con modalidad del golden: con margen 0.05 entra el 92 % de las modalidades correctas y el 15 % de las incorrectas; con 0.07, el 100 % y el 28 %. Una modalidad incorrecta solo podría entrar si además el razonador la propuso |
| **Resultado** | Actuó en 1 de 125 casos (D-0029, modalidad correcta): acc. modalidad en dev 0.80 → 0.85; en test, idéntico a V7. Resuelve además el caso real que lo motivó (captura de un contacto que pide plata: de `no_determinada` a `falso_familiar_cambio_numero`) |
| **Decisión** | **Versión 2** (tag `v2-ej4`) y configuración de la UI. Mejora un modo de falla concreto sin costo de API extra (el embedding del mensaje ya está en caché) y sin cambiar ningún otro resultado |

En la misma iteración se corrigió el **planner**, que afecta a todas las versiones con planner. Con
situación "no respondí" o "respondí" (sin pago ni datos entregados), el plan usa solo guías de
prevención: antes caía en los protocolos generales para víctimas (bloquear cuentas, denuncia
penal). Dentro de cada nivel, las guías de fuentes argentinas van antes que las de INCIBE (España).

## 4. Problemas encontrados y cómo se resolvieron

| Problema | Diagnóstico | Arreglo |
|---|---|---|
| CER 0.67 en capturas | El extractor transcribía avisos del sistema ("Esta empresa usa un servicio seguro de Meta…", "Leer más") como texto del remitente. La referencia tenía la PII redactada y la predicción no | `vision_v2` omite los avisos de la app. El CER se mide con las dos partes redactadas igual. Del slice `capturas` se excluyen las capturas con repeticiones. Resultado: CER 0.133 |
| V2–V6 con 0 de F1 en la primera corrida | Chroma 1.5 no persiste el HNSW de las colecciones con menos de 100 vectores (`batch_size` por defecto) y lo completa en la primera lectura; según qué proceso lea primero, el segmento queda roto ("Nothing found on disk") | `batch_size = sync_threshold = tamaño de la colección` en el build, y verificación del índice en un proceso aparte antes de escribir el manifest |
| Cobertura del plan 0.33 desde V2 (V0/V1: 0.87–0.94) | Había solo 13 guías, casi todas generales del Banco Nación | 13 → 42 guías en kb-v2 (las 24 páginas por modalidad). Subió a 0.42–0.44; el resto es el trade-off de §2 |
| Cobertura de casos: 8 modalidades con menos de 5 casos | Las fuentes iniciales eran generales | 24 páginas nuevas, 3 por modalidad. Quedan 3 modalidades con 3–4 casos |

## 5. Trabajo futuro

1. Modalidad `phishing_credenciales_servicio` en el enum y casos para ella (ataca la caída de
   acc. modalidad).
2. Guías por modalidad y situación más específicas, para subir la cobertura del plan sin perder
   grounding.
3. Más casos de las modalidades con menos cobertura (falso familiar: 3; inversión y empleo: 4).
4. Barrido de solo retrieval de los pesos de RRF y de `top_n`, que no cuesta generación.

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
