# Fuentes candidatas para la KB y el golden

> Relevamiento del 13/11/2026: solo se leyeron páginas, sin descargar ni ejecutar nada.
> **Hay que confirmar cada licencia y cada contacto antes de cargarlo en `data/sources.yaml`.**
> Veredictos de seguridad: ✅ seguro · ⚠️ usable con cuidado · ❌ descartar.

## Hallazgo principal

No existe ningún dataset público de mensajes de estafa en español rioplatense, ni de SMS ni de
WhatsApp. El único corpus nativo en español con mensajes legítimos y de estafa es **SpaPhish**,
y son emails. Para el test de SMS y WhatsApp hay que recolectar mensajes propios según la guía
de etiquetado. Los ejemplos citados en las alertas oficiales sirven como punto de partida.

## Top 5 para arrancar

1. **SpaPhish** (Mendeley, CC BY 4.0, CSV): https://data.mendeley.com/datasets/hz2d6gz7pc. Son
   1.395 emails en español (731 phishing y 664 legítimos), con triple anotación y un paper en
   Data in Brief. Sirve para el test del canal email y para la KB. **No visitar sus URLs**,
   porque conservan el dominio original y pueden estar vivas.
2. **BCRA**:
   - [prevención](https://www.bcra.gob.ar/como-prevenir-estafas-virtuales/)
   - [recomendaciones](https://www.bcra.gob.ar/noticias/recomendaciones-del-bcra-para-evitar-estafas-virtuales/)
   - [reclamo por fraude](https://www.bcra.gob.ar/realizar-un-reclamo-ante-el-banco-central-por-fraude-o-estafa/)

   Entre las tres cubren casos, señales y guías.
3. **argentina.gob.ar** (CC BY 4.0 explícita):
   - [alerta de ARCA](https://www.argentina.gob.ar/noticias/alerta-estafas-por-correo-electronico)
   - [alerta Ponzi de la CNV](https://www.argentina.gob.ar/node/298036)
   - [denunciar un delito informático](https://www.argentina.gob.ar/denunciar-un-delito-informatico)
4. **Banco Nación, UFECI y FAQ de WhatsApp**: el plan de acción, con teléfonos verificados, y
   la modalidad de robo de cuenta.
5. **SMS Phishing Dataset de Mishra & Soni** (CC BY 4.0) y **UCI SMS Spam** (CC BY 4.0):
   plantillas en inglés para traducir. **Solo van a la KB, nunca al test.**

## `casos`

| Fuente | URL | Licencia | Seguridad | Modalidades |
|---|---|---|---|---|
| BCRA, prevención | https://www.bcra.gob.ar/como-prevenir-estafas-virtuales/ | No visible | ✅ | Phishing, smishing, vishing, SIM swap, premios, falso error de transferencia, falsa mesa de ayuda |
| BCRA, recomendaciones (06/08/2025) | https://www.bcra.gob.ar/noticias/recomendaciones-del-bcra-para-evitar-estafas-virtuales/ | No visible | ✅ | Falso banco o BCRA por WhatsApp/Telegram, préstamos no solicitados, cajero |
| ARCA, alerta por correo (17/03/2025) | https://www.argentina.gob.ar/noticias/alerta-estafas-por-correo-electronico | CC BY 4.0 | ✅ | Deuda o fiscalización falsa |
| AFIP/ARCA, novedad 4671 | https://servicioscf.afip.gob.ar/publico/sitio/contenido/novedad/ver.aspx?id=4671 | No visible | ✅ | Deuda tributaria falsa |
| CNV, esquemas Ponzi (2022) | https://www.argentina.gob.ar/node/298036 | CC BY 4.0 | ✅ | Inversión falsa |
| Banco Nación, seguridad | https://www.bna.com.ar/Home/home/InformacionDeSeguridad | No visible | ✅ (ojo: `bn.com.pe` es el banco de Perú) | Phishing, smishing, vishing, quishing |
| WhatsApp FAQ, estafas | https://faq.whatsapp.com/2286952358121083/?locale=es_LA | Términos de Meta (parafrasear) | ✅ | Falso familiar, empleo, inversión, lotería |
| Informe UFECI 2024 | https://www.mpf.gob.ar/ufeci/files/2025/06/UFECI_informe_anual_2024-1.pdf | No visible | ⚠️ No se pudo extraer el texto: leerlo a mano | Estadísticas por modalidad |
| Chequeado, resumen del informe UFECI | https://chequeado.com/el-explicador/las-denuncias-de-estafas-virtuales-aumentaron-un-21-en-2024-cuales-son-los-tipos-de-fraude-mas-comunes/ | Copyright | ⚠️ Parafrasear y citar | Priorización de modalidades |
| ESET, Mercado Libre y Mercado Pago | https://www.welivesecurity.com/es/estafas-enganos/estafas-comunes-mercado-libre-mercado-pago/ | © ESET | ⚠️ Parafrasear | Falso comprador o vendedor, tareas pagas, soporte remoto, sorteo |
| PDFs de argentina.gob.ar | [cibercuidados](https://www.argentina.gob.ar/sites/default/files/manual_talleres_ti_cibercuidados_fraudes_y_estafas.pdf) · [fraudes por WhatsApp](https://www.argentina.gob.ar/sites/default/files/2022/07/recomendaciones_evitar_fraudes_whatsapp.pdf) | CC BY 4.0 | ⚠️ No se pudo extraer el texto | Falso familiar, robo de WhatsApp |
| INCIBE, avisos de smishing (España) | https://www.incibe.es/ciudadania/avisos/ | Ambigua | ⚠️ Adaptar al contexto argentino | Paquetería, bancos, fisco |

Datos del informe UFECI 2024, según Chequeado:
- El 63% de las denuncias son fraude en línea: 56% compraventa, 18% homebanking y 10% Ponzi.
- De los accesos ilegítimos, el 30% son cuentas de WhatsApp y el 26% de Mercado Pago.

## `mensajes`

| Dataset | URL | Idioma | Tamaño | Licencia | Seguridad | Uso |
|---|---|---|---|---|---|---|
| **SpaPhish** v5 | https://data.mendeley.com/datasets/hz2d6gz7pc | ES nativo | 1.395 emails | CC BY 4.0 | ✅ No visitar URLs ni abrir el reporte HTML con recursos remotos | Test (email) + KB |
| SMS Phishing Dataset (Mishra & Soni) | https://data.mendeley.com/datasets/f45bkkt8pr | EN | 5.971 SMS | CC BY 4.0 | ✅ No ejecutar el código incluido | Solo KB traducida |
| UCI SMS Spam Collection | https://archive.ics.uci.edu/dataset/228/sms+spam+collection | EN | 5.574 SMS | CC BY 4.0 | ✅ Trae teléfonos de UK | Solo KB |
| Zenodo, Phishing Email Curated | https://zenodo.org/records/8339691 | EN | ~449 MB | CC BY 4.0 | ⚠️ URLs vivas y PII sin anonimizar (Enron) | Solo KB (Nigerian_Fraud, Nazario) |
| Smishing-Dataset-IMC25 | https://github.com/reportsmishing/Smishing-Dataset-IMC25 | 66 idiomas | 64,5k | CC BY 4.0 | ⚠️ % de español no verificado; sender IDs y URLs reales | Revisar el subset en español |
| SpamShield | https://huggingface.co/datasets/M-Arjun/SpamShield-Datasets | ES 1.268 | — | CC BY 4.0 | ⚠️ 20% sintético o traducido sin marcar | Solo KB |
| Gabrielaz/spamspa | https://huggingface.co/datasets/Gabrielaz/spamspa | ES | <10k | Apache-2.0 | ⚠️ Probablemente una traducción de UCI | Solo KB |

## `senales`

Las páginas del BCRA, Banco Nación, WhatsApp, ARCA y CNV (ver arriba). Además:
- FTC, en español: https://consumidor.ftc.gov/articulos/como-reconocer-y-evitar-las-estafas-de-phishing
  (el canal de reporte 7726 no aplica en Argentina).
- INCIBE, smishing: https://www.incibe.es/ciudadania/tematicas/ingenieria-social-fraudes-online/smishing

## `guias` (contactos tal como figuran en la página; volver a verificar antes de cargar)

| Fuente | URL | Contactos |
|---|---|---|
| BCRA, reclamo por fraude | https://www.bcra.gob.ar/realizar-un-reclamo-ante-el-banco-central-por-fraude-o-estafa/ | Avisar al banco, denuncia, reclamo, 10 días hábiles, formulario BCRA · +54 11 4348-3500 |
| UFECI (MPF) | https://www.mpf.gob.ar/ufeci/ | denunciasufeci@mpf.gov.ar · (+54 11) 5071-0040 |
| Denunciar un delito informático | https://www.argentina.gob.ar/denunciar-un-delito-informatico | PFA, División Delitos Tecnológicos: 4800-1120 / 4370-5899 · delitostecnologicos@policiafederal.gov.ar |
| Banco Nación, si sos víctima | https://www.bna.com.ar/Home/home/InformacionDeSeguridad | Red Link 0800-888-5465 · Visa (011) 4379-3333 · Mastercard 0-800-555-0507 · *910 |
| ARCA, reportar phishing | https://www.argentina.gob.ar/noticias/alerta-estafas-por-correo-electronico | phishing@arca.gob.ar |
| WhatsApp, recuperar cuenta | https://faq.whatsapp.com/1131652977717250/?locale=es_LA | Reingresar con el código SMS; PIN de verificación en dos pasos |
| CNV, atención al inversor | https://www.argentina.gob.ar/node/298036 | inversor@cnv.gov.ar |
| Defensa del Consumidor | https://www.argentina.gob.ar/servicio/iniciar-un-reclamo-ante-defensa-del-consumidor | 0800-666-1518 (según la página del BCRA) · ⚠️ solo contra proveedores |

No se verificaron los contactos oficiales de ANSES, CERT.ar/ICIC, Correo Argentino ni Andreani.

## Descartadas

| Fuente | Motivo |
|---|---|
| `ealvaradob/phishing-dataset` (HF) | ❌ Requiere `trust_remote_code`, es decir, ejecutar código arbitrario |
| `dbarbedillo/SMS_Spam_Multilingual_Collection_Dataset` | ❌ Traducción automática de mala calidad, GPL y redundante con UCI |
| Mendeley `vmg875v4xs` | ❌ Generado por LLM y sin licencia visible |
| Smishtank | ❌ El link del dataset da 404 |
| MIMICS-3500 (ULeón) | ⚠️ Sin licencia; se pide por mail institucional (solo si lo gestiona la cátedra) |
| Notas de medios (iProfesional, Cronista, Ámbito…) | Copyright y no son fuente primaria: usarlas solo para llegar a la alerta oficial |
