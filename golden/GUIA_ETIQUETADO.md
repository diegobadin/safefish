# Guía de etiquetado del golden dataset

Versión: golden-guia-v1 · Responsable: Track B (cura el golden) · Revisor: Track A

El golden mide todas las versiones (V0 → V7). Si cambia una regla de esta guía, se reetiqueta
todo lo afectado y se sube la versión del golden: **nunca** se mezclan criterios entre corridas.

## 1. Formato

Un caso por línea en `golden/dev.jsonl` o `golden/test.jsonl` (esquema `GoldenCase` en
`src/safefish/schemas.py`). Validar siempre con:

```bash
make golden-check
```

```json
{"id": "T-0042", "split": "test", "slices": ["hard_negative"], "origen": "texto",
 "canal": "sms", "texto": "…", "imagen": null, "situacion": "no_respondio",
 "label": "legitimo", "veredicto_esperado": ["parece_legitimo", "levemente_sospechoso"],
 "modalidad": null, "senales_esperadas": ["urgencia_temporal"], "pasos_clave_esperados": [],
 "referencia": "Aviso real de vencimiento; urgencia legítima, dominio oficial, no pide datos.",
 "campaign_id": "banco-x-vto-2026-10", "fuente": "recolectado", "anotador": "DR", "revisor": "DB"}
```

| Campo | Regla |
|---|---|
| `id` | `D-NNNN` (dev) o `T-NNNN` (test). Único en todo el golden |
| `texto` | Transcripción literal del mensaje **con PII ya redactada** con placeholders tipados (`<TEL>`, `<CBU>`, `<NOMBRE>`, `<EMAIL>`, `<DNI>`, `<TARJETA>`, `<LINK_ACORTADO>`). No corregir ortografía ni emojis: son parte de la evidencia |
| `label` | `estafa` o `legitimo`. Es la verdad de fondo, no la opinión sobre el riesgo |
| `veredicto_esperado` | Conjunto de veredictos aceptables, **el primero es el canónico** (se usa para el kappa ponderado) |
| `modalidad` | Obligatoria si `label = estafa` (una del enum o `no_determinada`). `null` si es legítimo |
| `senales_esperadas` | Solo señales de **riesgo** del enum que estén presentes y sean citables en el texto |
| `pasos_clave_esperados` | Frases cortas con lo que el plan no puede omitir (se evalúa desde V2) |
| `referencia` | 1–3 oraciones: por qué es estafa/legítimo y qué lo delata. Es la `reference` de RAGAS |
| `campaign_id` | Misma campaña o plantilla ⇒ mismo `campaign_id`. Controla el split por grupo (§4.5) |
| `fuente` | `recolectado`, `dataset`, `hard_neg`, `alerta_oficial`. `sintetico` y `traducido` **nunca** en test |

## 2. Criterios por veredicto

| Veredicto | Cuándo |
|---|---|
| `estafa_confirmada` | Coincide con una modalidad documentada **y** tiene ≥ 2 señales, al menos una crítica (pide código, clave, tarjeta, transferencia a tercero o instalar acceso remoto) |
| `muy_sospechoso` | Pide una acción riesgosa o tiene varias señales altas, pero no calza del todo con una modalidad documentada |
| `levemente_sospechoso` | Señales débiles (urgencia, remitente raro, errores de forma) sin pedido de acción riesgosa |
| `parece_legitimo` | No pide datos ni pagos fuera de canales oficiales; dominio y remitente coherentes |

Binarización para las métricas: {estafa_confirmada, muy_sospechoso} = estafa;
{levemente_sospechoso, parece_legitimo} = no estafa.

Para un `legitimo` con urgencia real (vencimientos, alertas de seguridad genuinas), el conjunto
aceptable suele ser `["parece_legitimo", "levemente_sospechoso"]`.

## 3. Negativos difíciles (`slices: ["hard_negative"]`)

Mensajes **legítimos** que comparten rasgos superficiales con estafas: urgencia, links,
montos, avisos de bloqueo, pedidos de confirmar algo. Ejemplos: aviso real de un banco o
billetera, código de verificación que *llega* (no que se pide reenviar), delivery con link de
seguimiento oficial, turnos de organismos. Objetivo: ≥ 35 en test.

No es negativo difícil un mensaje personal trivial ("¿venís a cenar?"): ese va sin slice.

## 4. Cómo citar

- La cita debe ser una subcadena **literal** de `texto` (se tolera diferencia de mayúsculas,
  tildes y espacios). Es lo que después verifica la regla R1.
- Citar el fragmento mínimo que demuestra la señal (una frase, no el mensaje entero).
- En capturas (`slices: ["capturas"]`) se agrega el bbox de 1–3 citas clave en `citas_bbox`: `[{"cita": "<fragmento literal>", "bbox": [x0, y0, x1, y1]}]`, en píxeles de la imagen redactada. Se usa para la métrica de localización (IoU ≥ 0.5).

## 5. Otros slices

| Slice | Contenido |
|---|---|
| `sin_evidencia` | Modalidades ausentes de la KB a propósito o mensajes ambiguos. `modalidad: no_determinada` |
| `adversarial` | Estafas con instrucciones embebidas para la IA. Incluir `instrucciones_dirigidas_a_ia` en `senales_esperadas` |
| `capturas` | Mismo caso en formato imagen (`origen: captura`, `imagen: golden/capturas/<id>_red.png`) |

## 6. Controles de leakage (obligatorios antes de congelar)

1. Ningún `campaign_id` puede estar a la vez en test y en la KB (lo chequea `make golden-check`
   contra el manifest del snapshot).
2. Track A cura la KB y Track B cura el test: quien arma la KB no ve el test.
3. Si un mensaje de test sale de una alerta oficial, esa alerta se excluye de la KB.
4. Pre-etiquetado con LLM permitido, pero **revisión humana obligatoria** de cada caso.
