// SafeFish · front mínimo: Preact + htm (sin build) y Tesseract.js para marcar la captura.
import { html, render, useState, useEffect, useRef } from "https://unpkg.com/htm@3.1.1/preact/standalone.module.js";

const TESSERACT_URL = "https://cdn.jsdelivr.net/npm/tesseract.js@5.1.1/dist/tesseract.min.js";

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
    s.src = TESSERACT_URL; s.onload = () => ok(window.Tesseract); s.onerror = () => mal(new Error("No se pudo cargar Tesseract.js"));
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
  </div>`;
}

function App() {
  const [modo, setModo] = useState(null);
  const [cargando, setCargando] = useState(false);
  const [error, setError] = useState("");
  const [res, setRes] = useState(null);
  const [file, setFile] = useState(null);
  const reiniciar = () => { setModo(null); setRes(null); setError(""); setFile(null); };
  async function enviar({ texto, file, contexto, situacion }) {
    setCargando(true); setError("");
    const fd = new FormData();
    if (modo === "captura") fd.append("imagen", file); else fd.append("texto", texto);
    fd.append("contexto", contexto); fd.append("situacion", situacion);
    try {
      const r = await fetch("/api/analizar", { method: "POST", body: fd });
      const j = await r.json().catch(() => ({ error: "El servidor respondió algo que no se pudo leer." }));
      if (!r.ok || j.error) throw new Error(j.error || `Error ${r.status}`);
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
  </main>`;
}

render(html`<${App}/>`, document.getElementById("app"));
