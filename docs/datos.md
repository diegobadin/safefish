# Datos: qué se usa, en qué fase y dónde va

> Las fuentes públicas las baja `make data` (manifiesto: [`data/descargas.yaml`](../data/descargas.yaml)).
> Qué entra a la KB lo decide [`data/sources.yaml`](../data/sources.yaml). Relevamiento y
> veredictos de seguridad: [`fuentes_candidatas.md`](fuentes_candidatas.md).
> Nada de `data/raw/` ni `data/interim/` se versiona.

## Flujo

```
make data                     baja a data/raw/ y convierte los datasets a data/interim/mensajes/*.csv
   │                          (+ data/interim/golden_candidatos/spaphish.csv: 10 % reservado)
   ▼
make kb CONFIG=configs/v1.yaml   kb-v1:  las fuentes de sources.yaml, chunks de 800, 1 colección
make kb CONFIG=configs/v2.yaml   kb-v2:  las mismas fuentes, normalizadas, 4 colecciones
make kb CONFIG=configs/v2a.yaml  kb-v2a: chunk por caso en 1 colección (barrido V2a)
   │
   ▼
make eval CONFIG=configs/vN.yaml SPLIT=dev|test   contra golden/dev.jsonl y golden/test.jsonl
```

## Qué datos usa cada fase

La regla de la ablación es **una variable por paso**: todas las versiones usan las **mismas
fuentes** y el **mismo golden congelado**. Lo que cambia entre fases es cómo se procesan.

| Fase | Versión | KB | Golden | Qué cambia en los datos |
|---|---|---|---|---|
| 1 | V0 | — (sin RAG) | dev / test | Nada: cota inferior |
| 1 | V1 | `kb-v1` | dev / test | Fuentes crudas → chunks de 800 caracteres en 1 colección |
| 2 | V2a | `kb-v2a` | dev (solo retrieval) | Chunk por caso en 1 colección |
| 2 | V2, V3 | `kb-v2` | dev / test, + slice `capturas` (M-EXT) | Normalización a casos/mensajes/señales/guías en 4 colecciones; revisión manual del 10 % |
| 3 | V4–V6 | `kb-v2` | dev / test | Ninguno: BM25 se arma desde `chunks.jsonl` del mismo snapshot |
| 4 | V7 | `kb-v2` (o una variante si V7 toca la KB) | error analysis en **dev**, medición en **test** | Lo define el error analysis |
| 5 | Extras | `kb-v2` | — | UMAP sobre la colección `mensajes` |

## Qué fuente alimenta cada colección

| Colección | Fuentes (ids de `sources.yaml`) | Cantidad hoy |
|---|---|---|
| `senales` | `senales-taxonomia` | 15 señales |
| `casos` | BCRA (prevención, recomendaciones), Banco Nación, cibercuidados (argentina.gob.ar), ARCA, CNV, Ministerio de Seguridad (WhatsApp), Banco Provincia (acceso remoto), ESET (Mercado Libre, Correo Argentino), Chequeado (paquete retenido), INCIBE (smishing, España) y **24 páginas más, tres por cada modalidad con poca cobertura** (INCIBE, ESET, Chequeado, La Nación, Infobae, alertas de la UFECI y de ARCA publicadas en medios, MPBA) | 36 páginas o PDF; el normalizador separa un caso por modalidad. Las 24 nuevas tienen la modalidad fija en `sources.yaml` |
| `guias` | BCRA (reclamo), UFECI, argentina.gob.ar (denunciar), Banco Nación, Defensoría bonaerense (protocolo ciberestafa), Ministerio de Seguridad (WhatsApp), ARCA, CNV, Banco Provincia | 9 páginas o PDF |
| `mensajes` | SpaPhish (emails reales, ES), SpearPhishMX (emails, MX) y los **recolectados** | 1.255 + 3.005 emails (después del 10 % reservado) |

Cobertura de modalidades: la mide `make coverage` después de `make kb`. **Hueco conocido:** no
hay SMS ni WhatsApp en español en ningún dataset público. Esos mensajes salen de los recolectados.

### Datasets bajados pero apagados

| Dataset | Por qué está apagado | Para prenderlo |
|---|---|---|
| SpamShield (ES) | Acceso restringido en Hugging Face | Aceptar los términos en la página del dataset, poner `HF_TOKEN` en `.env` y correr `make data` |
| UCI SMS Spam (`GRUPOS=es,en`) | Inglés: el filtro de idioma de la KB lo descarta | Traducción al español (no implementada) |

**Descartados:** tanaos (sintético: su "spam" mezcla publicidad y contenido adulto con estafas y
sus legítimos son charla genérica, no negativos difíciles) y Mishra & Soni (SMS en inglés).

**Regla de etiquetas:** `label = estafa` solo cuando el dataset marca phishing, smishing o
fraude. El "spam" genérico queda sin etiqueta y el build lo descarta. La etiqueta original queda
en `label_origen`.

**Nunca al test:** datasets sintéticos o traducidos, SpearPhishMX (PII con marcadores y URLs
desactivadas: el sistema tiene que ver el mensaje como lo recibe el usuario).

## Golden v1

El golden necesita etiquetas que ningún dataset trae (veredicto en 4 niveles, modalidad, señales
con cita, situación). Formato y criterios: [`golden/GUIA_ETIQUETADO.md`](../golden/GUIA_ETIQUETADO.md).
Lo arma [`scripts/armar_golden.py`](../scripts/armar_golden.py) en dos pasos:

```
python scripts/armar_golden.py seleccionar   # elige casos; separa los recolectados de la KB
#   (pre-etiquetado → data/interim/golden/etiquetas_*.jsonl; revisión humana obligatoria)
python scripts/armar_golden.py ensamblar     # → golden/dev.jsonl, golden/test.jsonl, golden/procedencia.csv
make golden-check CONFIG=configs/v2.yaml     # esquema + leakage de campaign_id contra la KB
```

| Origen | Casos | Qué aporta |
|---|---|---|
| Recolectados (35 de 73, elegidos por campaña) | 18 estafas + 17 legítimos | SMS y WhatsApp reales, Argentina y otros países; los legítimos son negativos difíciles |
| SpaPhish (60 de los 140 reservados) | 30 estafas + 30 legítimos | Emails reales en español nativo |
| Adversariales (derivados) | 15 | Una estafa del golden con una instrucción para la IA inyectada (5 variantes) |
| Capturas (mismo caso en imagen) | 15 | Slice `capturas`: la imagen está en `data/raw/` y no se versiona |

Total **125 casos**: 35 en dev y 90 en test, partidos **por `campaign_id`** y estratificados por
(fuente, etiqueta). Ningún `campaign_id` del golden está en la KB: los otros 38 recolectados
van a la KB (`raw/recolectados/mensajes_kb.csv`) y los 140 de SpaPhish no entran.

**Estado de las etiquetas:** pre-etiquetadas con LLM (`anotador: pre-etiquetado-llm`,
`revisor: null`). La guía exige revisión humana de cada caso antes de congelar: las dudas de
cada caso están en la columna `dudas_preetiquetado` de `golden/procedencia.csv`.

## Recolectados: dónde y cómo dejarlos

```
data/raw/recolectados/
├── mensajes.csv              mensajes en texto (uno por fila)
└── capturas/
    ├── estafa/               PNG/JPG tal cual (con datos personales: esta carpeta no se versiona)
    └── legitimo/
```

`mensajes.csv` (UTF-8, separado por comas):

| Columna | Obligatoria | Valores |
|---|---|---|
| `id` | sí | Único, p. ej. `R-001` |
| `texto` | sí | El mensaje tal cual llegó (solo lo que escribió el remitente) |
| `label` | sí | `estafa` o `legitimo` |
| `canal` | no | `sms`, `whatsapp` o `email` |
| `modalidad` | no | Un id de `taxonomias.yaml` (si no, se infiere) |
| `entidad` | no | A quién suplanta o quién lo manda (p. ej. `Banco Nación`) |
| `campaign_id` | no | El mismo id para mensajes de una misma campaña (controla el leakage KB ↔ test) |
| `fecha` | no | `AAAA-MM-DD` |
| `parecido_a` | no | Solo legítimos: la modalidad a la que se parece (negativo difícil) |
| `pais` | no | `AR`, `UY`, `MX`… (el golden prioriza `AR`) |
| `captura`, `nota` | no | De qué captura sale y notas de la transcripción |

Hoy hay **73 mensajes** (42 estafas y 31 legítimos, 48 de Argentina), transcriptos de las 68
capturas. La etiqueta es la de la carpeta: las estafas están confirmadas (reportes en Reddit) y
los legítimos los revisó el equipo. Las transcripciones crudas, una línea por mensaje con el remitente visible, están en
`data/raw/recolectados/transcripciones/`. Una captura puede dar varios mensajes, y comparten
`campaign_id`.

Con separar las capturas en `estafa/` y `legitimo/` alcanza: el texto se transcribe con el
extractor de visión y se confirma a mano. Las que vayan al golden se redactan (nombres, números
propios, DNI) y se guardan como `golden/capturas/<id>_red.png`, que es lo único que se versiona.

Cuando exista `mensajes.csv`, se activa la fuente `msg-recolectados` en `sources.yaml`. Los
mensajes que vayan al golden **no** pueden estar también en la KB (`make golden-check` lo audita).
