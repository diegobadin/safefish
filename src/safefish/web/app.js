// SafeFish · front mínimo: Preact + htm (sin build) y Tesseract.js para marcar la captura.
// Preact + htm va en el repo (vendor/, 13 KB). Tesseract.js se carga del CDN con versión fija y
// hash de integridad (SRI); su worker, el núcleo WASM y el idioma los baja la propia librería.
import { html, render, useState, useEffect, useRef } from "/vendor/preact-htm.js";

const TESSERACT_URL = "https://cdn.jsdelivr.net/npm/tesseract.js@5.1.1/dist/tesseract.min.js";
const TESSERACT_SRI = "sha384-GJqSu7vueQ9qN0E9yLPb3Wtpd7OrgK8KmYzC8T1IysG1bcvxvIO4qtYR/D3A991F";

const SITUACIONES = [
  ["no_respondio", "Todavía no respondí", "Solo lo recibí."],
  ["respondio", "Respondí, pero no pagué ni pasé datos", "Contesté el mensaje, nada más."],
  ["pago_transfirio", "Pagué o transferí plata", "Hice una transferencia o un pago."],
  ["compartio_codigo", "Pasé un código", "Un código que me llegó por SMS o WhatsApp."],
  ["compartio_clave_o_tarjeta", "Pasé claves o datos de tarjeta", "Contraseñas, PIN, número o código de la tarjeta."],
  ["instalo_app_remota", "Instalé una app que me pidieron", "Por ejemplo, de acceso remoto o de soporte."],
];

// Íconos dibujados (trazo 1.75, 24px).
const Icon = ({ d, size = 24 }) => html`<svg width=${size} height=${size} viewBox="0 0 24 24" fill="none"
  stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${d}</svg>`;
const I = {
  logo: html`<path d="M12 3l7 3v5c0 4.5-3 8.2-7 10-4-1.8-7-5.5-7-10V6l7-3z"/><path d="M9 12l2 2 4-4"/>`,
  image: html`<rect x="3" y="3" width="18" height="18" rx="3"/><circle cx="9" cy="9" r="1.8"/><path d="M21 15l-5-5L5 21"/>`,
  text: html`<path d="M4 5h16v11H8l-4 4V5z"/><path d="M8 9h8M8 12h5"/>`,
  critico: html`<path d="M8 3h8l5 5v8l-5 5H8l-5-5V8l5-5z"/><path d="M12 8v5M12 16.5v.01"/>`,
  alto: html`<path d="M12 3l9.5 17h-19L12 3z"/><path d="M12 10v4M12 17.5v.01"/>`,
  medio: html`<circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16.5v.01"/>`,
  ok: html`<circle cx="12" cy="12" r="9"/><path d="M8.5 12.5l2.5 2.5 4.5-5"/>`,
};

// --- texto: ubicar citas aunque difieran en tildes, mayúsculas o espacios -------------------
function plano(t) {
  const chars = [], idx = [];
  for (let i = 0; i < t.length; i++) {
    const ch = t[i];
    if (/\s/.test(ch)) { if (chars.length && chars[chars.length - 1] === " ") continue; chars.push(" "); idx.push(i); continue; }
    for (const c of ch.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase()) { chars.push(c); idx.push(i); }
  }
  return [chars.join(""), idx];
}

function tramos(texto, senales) {
  const [p, idx] = plano(texto);
  let spans = [];
  senales.forEach((s) => {
    const c = plano(s.cita.trim())[0].trim();
    const k = c ? p.indexOf(c) : -1;
    if (k < 0) return;
    const a = idx[k], f = idx[k + c.length - 1] + 1;
    const dentro = spans.find((x) => x.a <= a && f <= x.f);
    if (dentro) { dentro.n.push(s.n); return; }
    const contenidos = spans.filter((x) => a <= x.a && x.f <= f);
    if (spans.some((x) => !contenidos.includes(x) && !(f <= x.a || a >= x.f))) return;
    spans = spans.filter((x) => !contenidos.includes(x)).concat([{ a, f, n: [s.n, ...contenidos.flatMap((x) => x.n)] }]);
  });
  spans.sort((x, y) => x.a - y.a);
  const out = []; let pos = 0;
  for (const s of spans) {
    out.push(texto.slice(pos, s.a));
    out.push(html`<mark>${texto.slice(s.a, s.f)}<sup>${s.n.sort((x, y) => x - y).join("·")}</sup></mark>`);
    pos = s.f;
  }
  out.push(texto.slice(pos));
  return out;
}

// --- captura: OCR en el navegador y recuadro sobre cada cita ---------------------------------
let tesseractCarga = null;
function cargarTesseract() {
  if (window.Tesseract) return Promise.resolve(window.Tesseract);
  tesseractCarga ??= new Promise((ok, mal) => {
    const s = document.createElement("script");
    s.src = TESSERACT_URL; s.integrity = TESSERACT_SRI; s.crossOrigin = "anonymous"; s.onload = () => ok(window.Tesseract); s.onerror = () => mal(new Error("No se pudo cargar Tesseract.js"));
    document.head.append(s);
  });
  return tesseractCarga;
}

const tok = (s) => s.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase().replace(/[^a-z0-9]+/g, "");

function ubicar(words, cita) {
  const toks = cita.split(/\s+/).map(tok).filter(Boolean);
  const L = toks.length;
  if (!L || !words.length) return null;
  const igual = (w, t) => w === t || (t.length > 3 && w.length > 2 && (w.startsWith(t) || t.startsWith(w)));
  // Alineación codiciosa: el OCR puede meter "palabras" de ruido (fondos, íconos) entre las de la
  // cita, así que cada token se busca hasta 2 palabras más adelante y solo cuentan las que coinciden.
  let mejor = { score: 0, usadas: [] };
  for (let i = 0; i < words.length; i++) {
    if (!igual(words[i].t, toks[0]) && !(L > 1 && igual(words[i].t, toks[1]))) continue;
    const usadas = []; let k = i;
    for (const t of toks) {
      for (let d = 0; d < 3 && k + d < words.length; d++) {
        if (igual(words[k + d].t, t)) { usadas.push(words[k + d]); k = k + d + 1; break; }
      }
    }
    if (usadas.length > mejor.score) mejor = { score: usadas.length, usadas };
  }
  const umbral = L === 1 ? 1 : 0.6;
  if (mejor.score / L < umbral) return null;
  // Une las palabras de cada renglón en un solo recuadro.
  const filas = [];
  for (const w of mejor.usadas) {
    const b = w.b, ult = filas[filas.length - 1];
    if (ult && Math.abs((b.y0 + b.y1) / 2 - (ult.y0 + ult.y1) / 2) < (b.y1 - b.y0) * 0.6) {
      ult.x0 = Math.min(ult.x0, b.x0); ult.x1 = Math.max(ult.x1, b.x1); ult.y0 = Math.min(ult.y0, b.y0); ult.y1 = Math.max(ult.y1, b.y1);
    } else filas.push({ ...b });
  }
  return filas;
}

function Captura({ file, senales }) {
  const ref = useRef(null);
  const [estado, setEstado] = useState("Marcando las señales sobre la captura…");
  useEffect(() => {
    let vivo = true;
    (async () => {
      const img = new Image();
      img.src = URL.createObjectURL(file);
      await img.decode();
      const cv = ref.current; cv.width = img.naturalWidth; cv.height = img.naturalHeight;
      const ctx = cv.getContext("2d"); ctx.drawImage(img, 0, 0);
      if (!senales.length) { setEstado(""); return; }
      try {
        // Tesseract lee mejor texto oscuro sobre fondo claro: invierte las capturas en modo oscuro.
        const pre = document.createElement("canvas"); pre.width = cv.width; pre.height = cv.height;
        const pc = pre.getContext("2d"); pc.drawImage(img, 0, 0);
        const px = pc.getImageData(0, 0, pre.width, pre.height); let sum = 0;
        for (let i = 0; i < px.data.length; i += 16) sum += px.data[i] * 0.3 + px.data[i + 1] * 0.59 + px.data[i + 2] * 0.11;
        if (sum / (px.data.length / 16) < 128) { pc.filter = "invert(1) grayscale(1)"; pc.drawImage(pre, 0, 0); }
        const T = await cargarTesseract();
        const { data } = await T.recognize(pre, "spa");
        if (!vivo) return;
        const words = data.words.map((w) => ({ t: tok(w.text), b: w.bbox })).filter((w) => w.t);
        const lw = Math.max(2, cv.width / 300);
        let marcadas = 0;
        for (const s of senales) {
          const filas = ubicar(words, s.cita);
          if (!filas) continue;
          marcadas++;
          for (const r of filas) {
            ctx.fillStyle = "rgba(255, 205, 40, 0.28)"; ctx.fillRect(r.x0 - 4, r.y0 - 3, r.x1 - r.x0 + 8, r.y1 - r.y0 + 6);
            ctx.strokeStyle = "#d9a400"; ctx.lineWidth = lw; ctx.strokeRect(r.x0 - 4, r.y0 - 3, r.x1 - r.x0 + 8, r.y1 - r.y0 + 6);
          }
          const r = filas[0], rad = Math.max(11, cv.width / 70);
          ctx.fillStyle = "#d9a400"; ctx.beginPath(); ctx.arc(r.x0 - 4, r.y0 - 3, rad, 0, Math.PI * 2); ctx.fill();
          ctx.fillStyle = "#1a1300"; ctx.font = `700 ${rad * 1.1}px system-ui`; ctx.textAlign = "center"; ctx.textBaseline = "middle";
          ctx.fillText(String(s.n), r.x0 - 4, r.y0 - 2);
        }
        setEstado(marcadas ? "" : "No pudimos ubicar las citas sobre la imagen; están marcadas en el texto.");
      } catch {
        if (vivo) setEstado("No pudimos marcar la imagen (hace falta conexión para cargar el lector). Las señales están marcadas en el texto.");
      }
    })();
    return () => { vivo = false; };
  }, [file]);
  return html`<div class="shot"><canvas ref=${ref} aria-label="Captura analizada con las señales marcadas"></canvas>
    ${estado && html`<p class="hint">${estado}</p>`}</div>`;
}

// --- depuración: el paso a paso del análisis ---------------------------------------------------
const num = (x, d = 3) => (x === null || x === undefined ? "—" : Number(x).toFixed(d).replace(".", ","));
const seg = (ms) => `${(ms / 1000).toFixed(1).replace(".", ",")} s`;
const NOMBRE_COL = { casos: "casos", mensajes: "mensajes", senales: "señales", guias: "guías" };

function Doc({ d, extracto = true }) {
  const tags = [d.modalidad, d.label, d.situacion, d.via].filter(Boolean);
  return html`<div class="doc"><code>${d.doc}</code>${tags.map((x) => html`<span class="tag">${x}</span>`)}
    ${extracto && d.texto && html`<div class="ext">${d.texto}</div>`}</div>`;
}

function Tabla({ cols, filas, marcar }) {
  return html`<div class="tw"><table class="dt"><thead><tr>${cols.map((c) => html`<th class=${c.n ? "n" : ""}>${c.t}</th>`)}</tr></thead>
    <tbody>${filas.map((f, i) => html`<tr class=${marcar && marcar(f, i) ? "keep" : ""}>${cols.map((c) => html`<td class=${c.n ? "n" : ""}>${c.v(f, i)}</td>`)}</tr>`)}</tbody></table></div>`;
}

function Def({ filas }) {
  const f = filas.filter(([, v]) => v !== null && v !== undefined && v !== "");
  return html`<dl class="kv">${f.map(([k, v]) => html`<dt>${k}</dt><dd>${v}</dd>`)}</dl>`;
}

function Salida({ s }) {
  return html`<${Def} filas=${[["Veredicto", s.veredicto], ["Modalidad", s.modalidad]]}/>
    ${s.senales.length > 0 && html`<${Tabla} cols=${[{ t: "Señal", v: (x) => html`<code>${x.id}</code>` }, { t: "Cita", v: (x) => html`<q>${x.cita}</q>` }]} filas=${s.senales}/>`}
    ${s.casos.length > 0 && html`<${Tabla} cols=${[{ t: "Caso citado", v: (x) => html`<code>${x.id}</code>` }, { t: "Por qué", v: (x) => x.por_que }]} filas=${s.casos}/>`}`;
}

function Busqueda({ p }) {
  const rank = { t: "#", n: 1, v: (d) => d.rank };
  const doc = (ext) => ({ t: "Documento", v: (d) => html`<${Doc} d=${d} extracto=${ext}/>` });
  const hayBm25 = p.por_query.some((q) => q.bm25);
  return html`<div class="stack-s">
    <p class="hint">Por cada query: búsqueda densa (similitud coseno)${hayBm25 ? " y BM25" : ""}, ${p.candidatos} candidatos por lista. Quedan ${p.top_n}.</p>
    ${p.por_query.map((q, i) => html`<details class="sub"><summary>Query ${i + 1}${i === 0 ? " (mensaje original)" : ""}: <q>${q.query}</q></summary>
      <div class="pair">
        <div><h5>Densa</h5><${Tabla} cols=${[rank, doc(false), { t: "Sim.", n: 1, v: (d) => num(d.denso) }]} filas=${q.denso}/></div>
        ${q.bm25 && html`<div><h5>BM25</h5><${Tabla} cols=${[rank, doc(false), { t: "Score", n: 1, v: (d) => num(d.bm25, 2) }]} filas=${q.bm25}/></div>`}
      </div></details>`)}
    ${p.fusion && html`<details class="sub"><summary>Fusión RRF (k = ${p.rrf_k}; pesos: densa ${p.pesos.dense}, BM25 ${p.pesos.bm25}) · ${p.fusion.length} candidatos</summary>
      <${Tabla} cols=${[rank, doc(true), { t: "Sim. densa", n: 1, v: (d) => num(d.denso) }, { t: "BM25", n: 1, v: (d) => num(d.bm25, 2) }, { t: "RRF", n: 1, v: (d) => num(d.rrf, 4) }]}
        filas=${p.fusion} marcar=${(d, i) => !p.rerank && i < p.top_n}/></details>`}
    ${p.rerank && html`<details class="sub" open><summary>Reranking con LLM (${p.reranker || "LLM"}, puntaje 0–3) · quedan los ${p.top_n} primeros</summary>
      <${Tabla} cols=${[rank, { t: "Antes", n: 1, v: (d) => d.rank_previo }, doc(true), { t: "Puntaje", n: 1, v: (d) => (d.rerank < 0 ? "—" : d.rerank) }]}
        filas=${p.rerank} marcar=${(d, i) => i < p.top_n}/></details>`}
    ${!p.rerank && !p.fusion && html`<${Tabla} cols=${[rank, doc(true), { t: "Sim.", n: 1, v: (d) => num(d.denso) }]} filas=${p.final}/>`}
  </div>`;
}

const PASOS = {
  extraccion: (p) => ({
    titulo: p.origen === "captura" ? "Lectura de la captura (visión)" : "Texto de entrada",
    meta: [p.canal && `canal ${p.canal}`, p.modelo].filter(Boolean).join(" · "),
    cuerpo: html`<${Def} filas=${[["Origen", p.origen], ["Canal", p.canal], ["Modelo", p.modelo], ["Palabras OCR", p.ocr_palabras], ["Contexto del usuario", p.contexto],
        ["Remitente que muestra la app", p.captura && p.captura.remitente && p.captura.remitente + (p.captura.insignia_verificada === true ? " (verificado)" : p.captura.insignia_verificada === false ? " (sin insignia)" : "")],
        ["Lo que se ve en la captura", p.captura && p.captura.descripcion]]}/>
      ${p.segmentos.length > 0
        ? html`<${Tabla} cols=${[{ t: "Emisor", v: (s) => s.emisor }, { t: "Texto transcripto", v: (s) => s.texto }]} filas=${p.segmentos}/>`
        : html`<pre class="raw">${p.texto}</pre>`}`,
  }),
  query_rewrite: (p) => ({
    titulo: "Reescritura de la consulta (multi-query)",
    meta: `${p.queries.length} queries${p.modelo ? " · " + p.modelo : ""}`,
    cuerpo: html`${p.facetas && html`<${Def} filas=${[["Resumen", p.facetas.resumen], ["Pedido", p.facetas.pedido], ["Entidad", p.facetas.entidad], ["Frases clave", (p.facetas.frases_clave || []).join(" · ")]]}/>`}
      <ol class="qs">${p.queries.map((q, i) => html`<li><q>${q}</q>${i === 0 ? html` <span class="tag">mensaje original</span>` : ""}</li>`)}</ol>`,
  }),
  busqueda: (p) => ({
    titulo: `Búsqueda en ${NOMBRE_COL[p.coleccion] || p.coleccion}${p.filtro ? ` · filtro ${Object.entries(p.filtro).map(([k, v]) => `${k} = ${v}`).join(", ")}` : ""}`,
    meta: `top ${p.top_n}${p.rerank ? " · con reranking" : p.fusion ? " · con RRF" : ""}`,
    cuerpo: html`<${Busqueda} p=${p}/>`,
  }),
  contexto: (p) => ({
    titulo: "Contexto que recibe el razonador",
    meta: Object.entries(p.colecciones).map(([k, v]) => `${v.length} ${NOMBRE_COL[k] || k}`).join(" · ") || "sin contexto (V0)",
    cuerpo: html`${Object.entries(p.colecciones).map(([k, v]) => html`<h5>${NOMBRE_COL[k] || k}</h5>
      <${Tabla} cols=${[{ t: "#", n: 1, v: (d) => d.rank }, { t: "Documento", v: (d) => html`<${Doc} d=${d}/>` }]} filas=${v}/>`)}
      ${Object.keys(p.tiempos_ms || {}).length > 0 && html`<p class="hint">Tiempos de recuperación: ${Object.entries(p.tiempos_ms).map(([k, v]) => `${k} ${Math.round(v)} ms`).join(" · ")}</p>`}`,
  }),
  generacion: (p) => ({
    titulo: "Generación (razonador, salida estructurada)",
    meta: [p.modelo, p.reintentos ? `${p.reintentos} reintento(s)` : ""].filter(Boolean).join(" · "),
    cuerpo: html`<p class="hint">Salida del modelo antes de las reglas del verificador.</p><${Salida} s=${p.salida}/>`,
  }),
  verificacion: (p) => ({
    titulo: `Verificador · ${p.fase}`,
    meta: p.eventos.length ? `${p.eventos.length} cambio(s)` : "sin cambios",
    cuerpo: html`${p.eventos.length
      ? html`<${Tabla} cols=${[{ t: "Regla", v: (e) => html`<b>${e.regla}</b>` }, { t: "Acción", v: (e) => e.accion }, { t: "Detalle", v: (e) => html`<code class="wrap">${JSON.stringify(e.detalle)}</code>` }]} filas=${p.eventos}/>`
      : html`<p class="hint">Ninguna regla modificó la salida.</p>`}
      <h5>Salida después de esta verificación</h5><${Salida} s=${p.salida}/>`,
  }),
  plan: (p) => ({
    titulo: "Plan desde guías oficiales",
    meta: [`${p.guias.length} guías`, `${p.pasos.length} pasos`, p.modelo].filter(Boolean).join(" · "),
    cuerpo: html`<${Def} filas=${[["Situación", p.info.situacion + (p.info.situacion_asumida ? " (asumida)" : "")], ["Modo", p.info.modo], ["Prompt", p.prompt],
        ["Entidades del mensaje", (p.info.entidades_mensaje || []).join(", ") || "ninguna"], ["Pasos descartados", p.info.pasos_descartados],
        ["Plan extractivo de respaldo", p.info.fallback_extractivo ? "sí" : "no"]]}/>
      <h5>Guías seleccionadas</h5><${Tabla} cols=${[{ t: "#", n: 1, v: (d) => d.rank }, { t: "Guía", v: (d) => html`<${Doc} d=${d}/>` }]} filas=${p.guias}/>
      <h5>Pasos generados</h5><${Tabla} cols=${[{ t: "#", n: 1, v: (x, i) => i + 1 }, { t: "Paso", v: (x) => x.paso }, { t: "Guía", v: (x) => html`<code>${x.guia}</code>` }]} filas=${p.pasos}/>
      ${(p.info.adaptaciones || []).length > 0 && html`<h5>Adaptaciones a la entidad</h5><pre class="raw">${JSON.stringify(p.info.adaptaciones, null, 2)}</pre>`}`,
  }),
  uso: (p) => ({
    titulo: "Llamadas al LLM, costo y tiempos",
    meta: `${p.llamadas.length} llamadas · USD ${num(p.costo_usd, 4)}`,
    cuerpo: html`<${Tabla} cols=${[{ t: "Etapa", v: (c) => c.etapa }, { t: "Modelo", v: (c) => c.modelo }, { t: "Tokens in", n: 1, v: (c) => c.in }, { t: "out", n: 1, v: (c) => c.out },
        { t: "Caché", v: (c) => (c.cache ? "sí" : "no") }, { t: "ms", n: 1, v: (c) => c.ms }, { t: "USD", n: 1, v: (c) => num(c.usd, 4) }]} filas=${p.llamadas}/>
      <p class="hint">Tiempo por etapa: ${Object.entries(p.latencias_ms).map(([k, v]) => `${k} ${seg(v)}`).join(" · ")}${Object.keys(p.embeddings || {}).length ? ` · tokens de embeddings: ${Object.values(p.embeddings).reduce((a, b) => a + b, 0)}` : ""}</p>
      ${p.error && html`<p class="error">${p.error}</p>`}`,
  }),
};

function Depuracion({ pasos }) {
  if (!pasos || !pasos.length) return null;
  return html`<section class="panel debug"><details>
    <summary><b>Paso a paso del análisis</b> <span class="hint">· ${pasos.length} pasos, desde la entrada hasta la respuesta</span></summary>
    <ol class="steps">${pasos.map((p, i) => {
      const r = (PASOS[p.etapa] || ((x) => ({ titulo: x.etapa, meta: "", cuerpo: html`<pre class="raw">${JSON.stringify(x, null, 2)}</pre>` })))(p);
      return html`<li><details><summary><span class="sn">${i + 1}</span><b>${r.titulo}</b><span class="sm">${r.meta}</span><span class="st">${seg(p.t_ms)}</span></summary>
        <div class="sb">${r.cuerpo}</div></details></li>`;
    })}</ol></details></section>`;
}

// --- vistas -----------------------------------------------------------------------------------
function Elegir({ onElegir }) {
  return html`<section class="panel narrow">
    <div><h2>¿Qué te llegó?</h2><p class="lead">Elegí cómo querés mostrarnos el mensaje sospechoso.</p></div>
    <div class="choices">
      <button class="choice" onClick=${() => onElegir("captura")}><${Icon} d=${I.image}/><b>Una captura</b><span>Subí la imagen del chat, SMS o mail.</span></button>
      <button class="choice" onClick=${() => onElegir("texto")}><${Icon} d=${I.text}/><b>El texto</b><span>Pegá el contenido del mensaje.</span></button>
    </div></section>`;
}

function Formulario({ modo, onCambiar, onEnviar, cargando, error }) {
  const [texto, setTexto] = useState("");
  const [file, setFile] = useState(null);
  const [prev, setPrev] = useState(null);
  const [contexto, setContexto] = useState("");
  const [situacion, setSituacion] = useState("no_respondio");
  const [over, setOver] = useState(false);
  const elegir = (f) => { if (f && f.type.startsWith("image/")) { setFile(f); setPrev(URL.createObjectURL(f)); } };
  const listo = modo === "captura" ? !!file : texto.trim().length > 0;
  const enviar = (e) => { e.preventDefault(); if (listo && !cargando) onEnviar({ texto, file, contexto, situacion }); };
  return html`<form class="panel narrow" onSubmit=${enviar}>
    <div class="field">
      ${modo === "captura"
        ? html`<label for="img">Captura del mensaje</label>
          <label class=${"drop" + (over ? " over" : "")} onDragOver=${(e) => { e.preventDefault(); setOver(true); }}
            onDragLeave=${() => setOver(false)} onDrop=${(e) => { e.preventDefault(); setOver(false); elegir(e.dataTransfer.files[0]); }}>
            <input id="img" type="file" accept="image/*" onChange=${(e) => elegir(e.target.files[0])}/>
            ${prev ? html`<img src=${prev} alt="Vista previa de la captura"/>` : html`<${Icon} d=${I.image} size=${32}/><span>Arrastrá la imagen o hacé clic para elegirla</span>`}
          </label>`
        : html`<label for="msg">Texto del mensaje</label>
          <textarea id="msg" class="msg-in" value=${texto} onInput=${(e) => setTexto(e.target.value)} placeholder="Pegá acá el mensaje, SMS o mail completo"></textarea>`}
      <button type="button" class="swap" onClick=${onCambiar}>${modo === "captura" ? "Prefiero pegar el texto" : "Prefiero subir una captura"}</button>
    </div>
    <div class="field">
      <label for="ctx">Contexto <span class="hint">(opcional)</span></label>
      <textarea id="ctx" class="ctx-in" maxlength="1000" value=${contexto} onInput=${(e) => setContexto(e.target.value)}
        placeholder="Quién te lo mandó o qué sabés. Ej.: es de un amigo y me pide plata."></textarea>
      <p class="hint">Ayuda a interpretar el mensaje; no se toma como parte de él.</p>
    </div>
    <fieldset><legend>¿Qué hiciste hasta ahora?</legend>
      ${SITUACIONES.map(([v, t, d]) => html`<label class="sit"><input type="radio" name="sit" value=${v} checked=${situacion === v}
        onChange=${() => setSituacion(v)}/><span>${t}</span><small>${d}</small></label>`)}
    </fieldset>
    ${error && html`<p class="error" role="alert">${error}</p>`}
    <button class="go" type="submit" disabled=${!listo || cargando}>
      ${cargando ? html`<span class="spin" aria-hidden="true"></span>Analizando… puede tardar unos 30 segundos` : "Analizar mensaje"}
    </button>
  </form>`;
}

function Resultado({ r, file }) {
  const v = r.veredicto;
  return html`<div class="result">
    ${r.aviso && html`<p class="notice">${r.aviso}</p>`}
    <section class=${"verdict " + v.nivel} aria-live="polite">
      <${Icon} d=${I[v.nivel] || I.medio} size=${32}/>
      <h2>${v.titulo}</h2>
      <p>${v.frase}</p>
      <p class="mod">${r.modalidad
        ? html`<b>${r.modalidad.nombre}.</b> <span>${r.modalidad.descripcion}</span>`
        : html`<b>Modalidad no identificada.</b> <span>No encontramos un caso documentado que coincida con este mensaje.</span>`}</p>
    </section>
    <div class="cols">
      <div class="stack">
        ${r.senales.length > 0 && html`<section class="panel"><h3>Por qué</h3><ol class="signals">
          ${r.senales.map((s) => html`<li><span class="num">${s.n}</span>
            <span class="sname">${s.nombre}${s.severidad && html`<span class=${"sev " + s.severidad}>${s.severidad_texto}</span>`}</span>
            <q class="cita">${s.cita}</q><span class="expl">${s.explicacion}</span></li>`)}</ol></section>`}
        <section class="panel"><h3>Qué hacer</h3><p class="hint" style="margin:-6px 0 12px">Según lo que contaste: ${r.situacion}.</p>
          <ol class="plan">${r.plan.map((p) => html`<li>${p}</li>`)}</ol></section>
      </div>
      <div class="stack">
        <section class="panel"><h3>Mensaje analizado</h3>
          ${file && html`<${Captura} file=${file} senales=${r.senales}/>`}
          <pre class="msg" style=${file ? "margin-top:14px" : ""}>${tramos(r.texto, r.senales)}</pre></section>
        ${r.casos.length > 0 && html`<section class="panel"><h3>Casos documentados parecidos</h3><div class="cases">
          ${r.casos.map((c) => html`<div class="case"><p>${c.por_que}</p>${c.url && html`<a href=${c.url} target="_blank" rel="noopener noreferrer">Ver fuente · ${c.dominio}</a>`}</div>`)}</div></section>`}
      </div>
    </div>
    <section class="panel stack">
      ${r.contexto && html`<details><summary>Contexto que diste</summary><p>${r.contexto}</p></details>`}
      <details><summary>Respuesta completa (JSON)</summary><pre>${JSON.stringify(r.json, null, 2)}</pre></details>
      <p class="meta">SafeFish ${r.meta.version} · ${Math.round(r.meta.segundos)} s${r.meta.costo ? ` · USD ${r.meta.costo.toFixed(3)}` : ""}</p>
    </section>
    <${Depuracion} pasos=${r.debug && r.debug.pasos}/>
  </div>`;
}

function App() {
  const [modo, setModo] = useState(null);
  const [cargando, setCargando] = useState(false);
  const [error, setError] = useState("");
  const [res, setRes] = useState(null);
  const [file, setFile] = useState(null);
  const [debugError, setDebugError] = useState(null);
  const reiniciar = () => { setModo(null); setRes(null); setError(""); setFile(null); setDebugError(null); };
  async function enviar({ texto, file, contexto, situacion }) {
    setCargando(true); setError("");
    const fd = new FormData();
    if (modo === "captura") fd.append("imagen", file); else fd.append("texto", texto);
    fd.append("contexto", contexto); fd.append("situacion", situacion);
    try {
      const r = await fetch("/api/analizar", { method: "POST", body: fd });
      const j = await r.json().catch(() => ({ error: "El servidor respondió algo que no se pudo leer." }));
      if (!r.ok || j.error) { setDebugError(j.debug ? j.debug.pasos : null); throw new Error(j.error || `Error ${r.status}`); }
      setDebugError(null);
      setFile(modo === "captura" ? file : null); setRes(j); window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (e) {
      setError(`No se pudo analizar: ${e.message}. Revisá que el servidor siga corriendo y probá de nuevo.`);
    } finally { setCargando(false); }
  }
  return html`<main class="shell">
    <header class="top"><${Icon} d=${I.logo} size=${30}/><div><h1>SafeFish</h1><p>Revisá un mensaje sospechoso antes de responder.</p></div>
      ${(modo || res) && html`<button class="ghost reset" onClick=${reiniciar}>Analizar otro mensaje</button>`}</header>
    ${res ? html`<${Resultado} r=${res} file=${file}/>`
      : modo ? html`<${Formulario} modo=${modo} onCambiar=${() => setModo(modo === "captura" ? "texto" : "captura")} onEnviar=${enviar} cargando=${cargando} error=${error}/>`
      : html`<${Elegir} onElegir=${setModo}/>`}
    ${!res && debugError && html`<${Depuracion} pasos=${debugError}/>`}
  </main>`;
}

render(html`<${App}/>`, document.getElementById("app"));
