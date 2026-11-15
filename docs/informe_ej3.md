LLMS · TP3 · ITBA — Diego Badin 63551 · Diego Rabinovich 63155

# TP3 – SafeFish · Ejercicio 3: Responsible AI, Safety y puesta en producción

SafeFish analiza mensajes sospechosos (texto o captura de WhatsApp, SMS o email) y devuelve un
veredicto, la modalidad de estafa, las señales con su cita literal, los casos documentados que lo
respaldan y un plan de acción. Como la salida orienta la decisión de una persona que puede estar
siendo víctima de un fraude, los errores tienen un costo directo y asimétrico. Las consideraciones
siguientes se ordenan según los riesgos propios de ese caso de uso.

## 1. Responsible AI y Safety

### 1.1 Riesgos identificados

- **Falso negativo:** un mensaje fraudulento clasificado como legítimo puede inducir al usuario a
  completar la estafa. Es el error de mayor costo.
- **Falso positivo:** una comunicación legítima del banco marcada como estafa reduce la confianza
  en los canales oficiales.
- **Alucinación:** señales o casos inexistentes, o datos de contacto inventados en el plan de
  acción, que podrían derivar al usuario hacia un tercero malicioso.
- **Prompt injection:** el mensaje analizado es contenido adversarial por naturaleza y puede
  incluir instrucciones dirigidas al modelo.
- **Envenenamiento de la base de conocimiento:** fuentes externas alteradas que modifiquen el
  resultado del retrieval.
- **Privacidad:** los mensajes y capturas contienen nombres, teléfonos, CBU y datos de tarjetas.
- **Cobertura desigual:** modalidades, canales o variantes regionales con pocos casos documentados.

### 1.2 Medidas

**Groundedness y transparencia.** Cada señal incluye una cita literal del mensaje, verificada por
una regla determinística (R1). La modalidad debe estar respaldada por al menos un caso recuperado,
que se muestra con su fuente (R2). El plan de acción se redacta solo a partir de guías oficiales,
y una regla elimina teléfonos, URLs u organismos ausentes en la guía de origen (R5).

**Asimetría de errores.** El veredicto tiene cuatro niveles. Un mensaje con una señal crítica no
puede clasificarse como legítimo (R4), y "estafa confirmada" exige una modalidad respaldada y al
menos dos señales verificadas (R3). Si la evidencia no alcanza, el sistema informa la modalidad
como no determinada en lugar de forzar una respuesta (*negative rejection*).

**Robustez frente a inyección.** El mensaje se delimita como dato y el prompt indica ignorar
cualquier orden contenida en él. El planner recibe solo campos estructurados, nunca el texto del
mensaje. El conjunto de evaluación incluye variantes adversariales y se reporta la tasa de éxito
del ataque (ASR).

**Integridad de la base de conocimiento.** Las fuentes se descargan solo por HTTPS desde una lista
cerrada de dominios oficiales y medios, con registro SHA-256 de cada archivo, y las páginas se
almacenan sin contenido activo. La normalización conserva solo los campos de un esquema fijo,
descarta frases típicas que no aparecen en la fuente y registra en el manifest una auditoría de
patrones de inyección y una muestra del 10 % para revisión manual.

**Privacidad.** Los datos personales se redactan antes de indexar y en el golden. La interfaz no
registra el contenido analizado y elimina las capturas al terminar cada análisis. En producción se
aplicaría retención mínima, consentimiento explícito para reutilizar mensajes con fines de mejora
y cumplimiento de la Ley 25.326 de Protección de Datos Personales.

**Equidad y cobertura.** Las métricas se reportan desagregadas por canal, modalidad y origen del
mensaje. Un control de cobertura exige al menos cinco casos por modalidad; las modalidades por
debajo del umbral se declaran como limitación.

**Comunicación al usuario.** La salida se presenta como apoyo a la decisión: indica el nivel de
certeza y la evidencia utilizada, y recomienda verificar por los canales oficiales de la entidad.
El sistema no solicita claves, códigos ni datos de tarjeta y no sigue los enlaces del mensaje.

**Supervisión humana.** Las etiquetas del golden se revisan manualmente, la *data card* documenta
fuentes y licencias, y cada iteración del sistema parte de un análisis de errores.

## 2. Estrategias para la puesta en producción

### 2.1 Arquitectura de despliegue

- Servicio de inferencia sin estado (API REST en contenedor, por ejemplo FastAPI) con escalado
  horizontal, que expone un endpoint de análisis para texto e imagen.
- Canales de acceso: interfaz web y bot de WhatsApp mediante la WhatsApp Business Platform, al que
  el usuario reenvía el mensaje o la captura.
- Base vectorial gestionada (Chroma en modo servidor o PostgreSQL con pgvector), con un snapshot
  inmutable y versionado por despliegue.
- Ingesta como proceso programado (semanal o ante nuevas alertas oficiales), independiente del
  servicio de inferencia.

### 2.2 Actualización de la base de conocimiento

Cada versión de la base se construye fuera de línea, se valida (cobertura por modalidad, auditoría
de inyección y revisión de la muestra) y se evalúa contra el golden de test. Se promueve solo si no
hay regresión en F1, tasa de falsos positivos sobre negativos difíciles ni grounding; el snapshot
anterior queda disponible para volver atrás de inmediato. Los modelos y prompts se fijan por
versión (identificador exacto del modelo y hash del prompt), lo que mantiene la reproducibilidad.

### 2.3 Monitoreo

- **Operación:** latencia por etapa (p50 y p95), tasa de errores y de reintentos de parseo, tokens
  y costo por análisis, ya instrumentados por etapa, con trazas en LangSmith u OpenTelemetry.
- **Calidad:** distribución de veredictos y modalidades, proporción de modalidad no determinada (un
  aumento sostenido indica modalidades nuevas sin cobertura), intervenciones del verificador y
  valoración explícita del usuario. Una muestra de casos se revisa manualmente y se incorpora al
  golden.
- **Alertas** ante desvíos respecto de la línea base de cada métrica.

### 2.4 Costo y escalabilidad

La versión final cuesta del orden de USD 0,02 por análisis. Las campañas de estafa se distribuyen
de forma masiva con texto casi idéntico, por lo que una caché por hash del mensaje normalizado
evita reprocesar los mismos mensajes. El costo puede reducirse usando un modelo de menor tamaño en
los casos de alta confianza y reservando el modelo grande para los ambiguos; esa variante se
evaluaría con la misma metodología de ablación.

### 2.5 Seguridad operativa

Secretos en un gestor dedicado y fuera del repositorio, autenticación y límite de solicitudes por
usuario, validación de tipo y tamaño de las imágenes, registros sin datos personales y una
respuesta conservadora de contingencia (verificar por el canal oficial) ante una falla del
proveedor del LLM. Una capa de abstracción sobre el cliente del modelo permite cambiar de
proveedor sin modificar el pipeline.
