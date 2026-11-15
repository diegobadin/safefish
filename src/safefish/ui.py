"""UI mínima (§2.6): pegar un mensaje o subir una captura → veredicto, señales, casos y plan.

    python -m safefish.ui --config configs/v6.yaml [--port 8501]

Servidor local con la biblioteca estándar (sin dependencias nuevas). Escucha solo en
127.0.0.1: es una demo para el oral, no un servicio. Las capturas subidas se guardan en una
carpeta temporal y se borran al terminar el análisis; la imagen anotada se devuelve en línea.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import html
import sys
import tempfile
import threading
from email.parser import BytesParser
from email.policy import default as email_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from dotenv import load_dotenv

MAX_UPLOAD = 15 * 1024 * 1024
COLOR = {
    "estafa_confirmada": "#b42318", "muy_sospechoso": "#c4320a",
    "levemente_sospechoso": "#b54708", "parece_legitimo": "#067647",
}  # fmt: skip

PAGINA = """<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>SafeFish</title>
<style>
:root{{--bg:#f7f7f5;--fg:#1d1d1b;--muted:#5f5f5a;--card:#fff;--line:#e3e3df}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151514;--fg:#ececea;--muted:#a3a39e;--card:#1f1f1d;--line:#33332f}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}}
main{{max-width:860px;margin:0 auto;padding:24px 16px}}
h1{{font-size:1.6rem;margin:0 0 4px}} p.sub{{color:var(--muted);margin:0 0 20px}}
form,.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin-bottom:16px}}
textarea{{width:100%;box-sizing:border-box;min-height:140px;font:inherit;padding:8px;border-radius:6px;border:1px solid var(--line);background:var(--bg);color:var(--fg)}}
label{{display:block;margin:10px 0 4px;color:var(--muted);font-size:.9rem}}
button{{margin-top:12px;padding:10px 18px;border:0;border-radius:6px;background:#1d4ed8;color:#fff;font:inherit;cursor:pointer}}
.ver{{display:inline-block;padding:4px 10px;border-radius:6px;color:#fff;font-weight:600}}
li{{margin:6px 0}} q{{font-style:italic}} img{{max-width:100%;border-radius:8px}}
details pre{{white-space:pre-wrap;word-break:break-word;font-size:.8rem}} .muted{{color:var(--muted)}}
</style></head><body><main>
<h1>SafeFish</h1><p class="sub">¿Te llegó un mensaje raro? Pegalo o subí la captura. Versión: {version}</p>
<form method="post" action="/" enctype="multipart/form-data">
<label for="t">Texto del mensaje</label><textarea id="t" name="texto">{texto}</textarea>
<label for="i">o una captura (PNG/JPG)</label><input id="i" type="file" name="imagen" accept="image/*">
<label for="s">¿Qué hiciste?</label><select id="s" name="situacion">
<option value="no_respondio">No respondí</option><option value="respondio">Respondí</option>
<option value="pago_transfirio">Pagué o transferí</option><option value="compartio_codigo">Compartí un código</option>
<option value="compartio_clave_o_tarjeta">Compartí clave o tarjeta</option><option value="instalo_app_remota">Instalé una app que me pidieron</option>
</select><br><button type="submit">Analizar</button></form>
{resultado}
</main></body></html>"""


def render_resultado(res, imagen_anotada: str | None) -> str:
    if res.error or res.output is None:
        return f'<div class="card"><b>No se pudo analizar.</b> <span class="muted">{html.escape(res.error or "")}</span></div>'  # fmt: skip
    o = res.output
    v = o.veredicto.value
    partes = [
        f'<div class="card"><span class="ver" style="background:{COLOR.get(v, "#444")}">'
        f"{v.replace('_', ' ')}</span> · modalidad: <b>{o.modalidad.value.replace('_', ' ')}</b>"
    ]
    if o.senales:
        partes.append("<h3>Señales</h3><ul>")
        partes += [
            f"<li><b>{s.signal_id.value.replace('_', ' ')}</b>: <q>{html.escape(s.cita)}</q><br>"
            f'<span class="muted">{html.escape(s.explicacion)}</span></li>'
            for s in o.senales
        ]
        partes.append("</ul>")
    if o.plan:
        partes.append("<h3>Qué hacer</h3><ol>")
        partes += [f"<li>{html.escape(p.paso)}</li>" for p in o.plan]
        partes.append("</ol>")
    if o.casos_similares:
        partes.append("<h3>Casos documentados parecidos</h3><ul>")
        for c in o.casos_similares:
            fuente = (
                f' · <span class="muted">{html.escape(c.fuente_url)}</span>' if c.fuente_url else ""
            )
            partes.append(f"<li>{html.escape(c.por_que)}{fuente}</li>")
        partes.append("</ul>")
    if imagen_anotada:
        partes.append(f'<h3>Captura anotada</h3><img alt="captura con las señales marcadas" src="data:image/png;base64,{imagen_anotada}">')  # fmt: skip
    if res.extracted is not None:
        partes.append(f'<details><summary>Texto leído</summary><pre>{html.escape(res.extracted.texto)}</pre></details>')  # fmt: skip
    partes.append(f"<details><summary>JSON completo</summary><pre>{html.escape(o.model_dump_json(indent=2))}</pre></details></div>")  # fmt: skip
    return "".join(partes)


def make_handler(pipeline, version: str):
    from safefish.postprocess.annotate import annotate
    from safefish.schemas import RawInput

    lock = threading.Lock()  # una consulta por vez: es una demo

    class Handler(BaseHTTPRequestHandler):
        def _send(self, body: str, code: int = 200):
            data = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._send(PAGINA.format(version=html.escape(version), texto="", resultado=""))

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_UPLOAD:
                return self._send("Archivo demasiado grande", 413)
            body = self.rfile.read(n)
            msg = BytesParser(policy=email_policy).parsebytes(
                b"Content-Type: " + self.headers["Content-Type"].encode() + b"\r\n\r\n" + body
            )
            campos: dict[str, bytes | str] = {}
            for part in msg.iter_parts():
                name = part.get_param("name", header="content-disposition")
                if part.get_filename():
                    campos[name] = part.get_payload(decode=True) or b""
                else:
                    campos[name] = (part.get_payload(decode=True) or b"").decode("utf-8", "replace")
            texto = str(campos.get("texto") or "").strip()
            img = campos.get("imagen") or b""
            situacion = str(campos.get("situacion") or "no_respondio")
            with tempfile.TemporaryDirectory() as tmp, lock:
                img_path = None
                if img:
                    img_path = str(Path(tmp) / "captura.png")
                    Path(img_path).write_bytes(img)
                if not texto and not img_path:
                    return self._send(PAGINA.format(version=version, texto="", resultado='<div class="card">Pegá un texto o subí una captura.</div>'))  # fmt: skip
                raw = RawInput(texto=texto or None, imagen_path=img_path, situacion=situacion)
                res = pipeline.run(raw)
                anotada = None
                if img_path and res.output is not None:
                    try:
                        out_png = Path(tmp) / "anotada.png"
                        annotate(img_path, res.output, out_png)
                        anotada = base64.b64encode(out_png.read_bytes()).decode()
                    except Exception:
                        anotada = None
            self._send(PAGINA.format(version=html.escape(version), texto=html.escape(texto),
                                     resultado=render_resultado(res, anotada)))  # fmt: skip

        def log_message(self, fmt, *args):  # sin logs con contenido de mensajes
            pass

    return Handler


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/v6.yaml")
    ap.add_argument("--port", type=int, default=8501)
    args = ap.parse_args(argv)

    from safefish.config import load_config
    from safefish.pipeline import SafeFishPipeline

    cfg = load_config(args.config)
    pipeline = SafeFishPipeline.from_config(cfg)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(pipeline, cfg.version))
    print(f"SafeFish {cfg.version} en http://127.0.0.1:{args.port} (Ctrl+C para cortar)")
    with contextlib.suppress(KeyboardInterrupt):
        srv.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
