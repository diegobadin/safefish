"""UI local (§2.6): front en Preact (src/safefish/web/) + API JSON sobre el pipeline.

    python -m safefish.ui [--config configs/v8.yaml] [--port 8501]

Servidor con la biblioteca estándar (sin dependencias nuevas). Escucha solo en 127.0.0.1 y solo
atiende pedidos dirigidos a sí mismo: es una demo para el oral, no un servicio. Las capturas se
guardan en una carpeta temporal y se borran al terminar el análisis. No se registra el contenido
de los mensajes. La captura se marca en el navegador (Tesseract.js), no en el servidor.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import tempfile
import threading
import time
from email.parser import BytesParser
from email.policy import default as email_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

MAX_UPLOAD = 15 * 1024 * 1024
WEB = Path(__file__).with_name("web")
ESTATICOS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
}

SITUACIONES = {
    "no_respondio": "Todavía no respondí",
    "respondio": "Respondí, pero no pagué ni pasé datos",
    "pago_transfirio": "Pagué o transferí plata",
    "compartio_codigo": "Pasé un código",
    "compartio_clave_o_tarjeta": "Pasé claves o datos de tarjeta",
    "instalo_app_remota": "Instalé una app que me pidieron",
}
VEREDICTOS = {
    "estafa_confirmada": (
        "Es una estafa",
        "critico",
        "Coincide con una modalidad documentada y tiene varias señales claras. No hagas lo que pide.",
    ),  # fmt: skip
    "muy_sospechoso": (
        "Muy sospechoso",
        "alto",
        "Tiene señales típicas de estafa. No hagas lo que pide hasta verificarlo por otro canal.",
    ),  # fmt: skip
    "levemente_sospechoso": (
        "Levemente sospechoso",
        "medio",
        "Hay algo que no cierra. Verificalo por un canal oficial antes de responder.",
    ),  # fmt: skip
    "parece_legitimo": (
        "Parece legítimo",
        "ok",
        "No encontramos señales de estafa. Ante la duda, consultá por el canal oficial de la entidad.",
    ),  # fmt: skip
}
SEVERIDAD = {"critica": "Crítica", "alta": "Alta", "media": "Media", "baja": "Baja"}


def respuesta(res, situacion: str, contexto: str | None, segundos: float, version: str,
              aviso: str | None = None) -> dict:  # fmt: skip
    """Lo que necesita el front, ya traducido a lenguaje de usuario."""
    from safefish.taxonomia import load_taxonomia

    if res.error or res.output is None:
        return {"error": res.error or "El análisis no devolvió resultado."}
    tax, o = load_taxonomia(), res.output
    titulo, nivel, frase = VEREDICTOS.get(o.veredicto.value, (o.veredicto.value, "medio", ""))
    mod = o.modalidad.value
    info = tax["modalidades"].get(mod, {})
    senales = []
    for n, s in enumerate(o.senales, 1):
        sev = tax["senales"].get(s.signal_id.value, {}).get("severidad", "")
        senales.append({
            "n": n, "id": s.signal_id.value, "nombre": s.signal_id.value.replace("_", " ").capitalize(),
            "severidad": sev, "severidad_texto": SEVERIDAD.get(sev, ""), "cita": s.cita,
            "explicacion": s.explicacion,
        })  # fmt: skip
    casos = []
    for c in o.casos_similares:
        url = c.fuente_url if c.fuente_url and c.fuente_url.startswith("https://") else None
        casos.append({"por_que": c.por_que, "url": url, "dominio": urlparse(url).hostname if url else None})  # fmt: skip
    return {
        "veredicto": {"id": o.veredicto.value, "titulo": titulo, "nivel": nivel, "frase": frase},
        "modalidad": None
        if mod == "no_determinada"
        else {
            "id": mod,
            "nombre": info.get("nombre", mod),
            "descripcion": info.get("descripcion", ""),
        },  # fmt: skip
        "senales": senales,
        "texto": res.extracted.texto if res.extracted is not None else "",
        "plan": [p.paso for p in o.plan],
        "situacion": SITUACIONES.get(situacion, situacion),
        "casos": casos,
        "contexto": contexto,
        "aviso": aviso,
        "meta": {
            "version": version,
            "segundos": round(segundos, 1),
            "costo": (res.usage or {}).get("costo_usd"),
        },  # fmt: skip
        "json": json.loads(o.model_dump_json()),
    }


def make_handler(pipeline, version: str, port: int = 8501):
    from safefish.schemas import RawInput

    lock = threading.Lock()  # una consulta por vez: es una demo
    # Solo se aceptan pedidos dirigidos a este servidor local. Sin esto, cualquier página
    # abierta en el navegador podría mandar un POST (CSRF) y gastar la API, o leer resultados
    # con un DNS rebinding.
    hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    origins = {f"http://{h}" for h in hosts}

    class Handler(BaseHTTPRequestHandler):
        def _send(
            self, body: bytes | str, code: int = 200, tipo: str = "text/plain; charset=utf-8"
        ):
            data = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", tipo)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _json(self, obj: dict, code: int = 200):
            self._send(json.dumps(obj, ensure_ascii=False), code, "application/json; charset=utf-8")

        def _host_ok(self) -> bool:
            return (self.headers.get("Host") or "").lower() in hosts

        def do_GET(self):
            if not self._host_ok():
                return self._send("Host no permitido", 403)
            estatico = ESTATICOS.get(self.path.split("?", 1)[0])
            if estatico is None:
                return self._send("No encontrado", 404)
            nombre, tipo = estatico
            self._send((WEB / nombre).read_bytes(), 200, tipo)

        def do_POST(self):
            origin = (self.headers.get("Origin") or "").lower()
            if not self._host_ok() or (origin and origin not in origins):
                return self._json({"error": "Origen no permitido"}, 403)
            if self.path != "/api/analizar":
                return self._json({"error": "No encontrado"}, 404)
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_UPLOAD:
                return self._json({"error": "La imagen supera los 15 MB"}, 413)
            body = self.rfile.read(n)
            tipo = (self.headers.get("Content-Type") or "").encode()
            msg = BytesParser(policy=email_policy).parsebytes(b"Content-Type: " + tipo + b"\r\n\r\n" + body)  # fmt: skip
            campos: dict[str, bytes | str] = {}
            if msg.is_multipart():
                for part in msg.iter_parts():
                    name = part.get_param("name", header="content-disposition")
                    if part.get_filename():
                        campos[name] = part.get_payload(decode=True) or b""
                    else:
                        campos[name] = (part.get_payload(decode=True) or b"").decode(
                            "utf-8", "replace"
                        )
            texto = str(campos.get("texto") or "").strip()
            contexto = str(campos.get("contexto") or "").strip()[:1000]
            img = campos.get("imagen") or b""
            situacion = str(campos.get("situacion") or "no_respondio")
            situacion = situacion if situacion in SITUACIONES else "no_respondio"
            if not texto and not img:
                return self._json({"error": "Pegá el texto del mensaje o subí una captura"}, 400)
            aviso = None
            if img and texto:  # manda la captura; lo pegado se usa como contexto
                contexto = f"{contexto}\n{texto}".strip() if contexto else texto
                aviso = "Analizamos la captura. El texto que pegaste lo usamos como contexto."
            with tempfile.TemporaryDirectory() as tmp, lock:
                img_path = None
                if img:
                    img_path = str(Path(tmp) / "captura.png")
                    Path(img_path).write_bytes(img)
                raw = RawInput(
                    texto=None if img_path else texto, imagen_path=img_path,
                    situacion=situacion, contexto=contexto or None,
                )  # fmt: skip
                t0 = time.perf_counter()
                res = pipeline.run(raw)
                segundos = time.perf_counter() - t0
            r = respuesta(res, situacion, contexto or None, segundos, version, aviso)
            self._json(r, 200 if "error" not in r else 502)

        def log_message(self, fmt, *args):  # sin logs con contenido de mensajes
            pass

    return Handler


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/v8.yaml")
    ap.add_argument("--port", type=int, default=8501)
    args = ap.parse_args(argv)

    from safefish.config import load_config
    from safefish.pipeline import SafeFishPipeline

    cfg = load_config(args.config)
    pipeline = SafeFishPipeline.from_config(cfg)
    srv = ThreadingHTTPServer(
        ("127.0.0.1", args.port), make_handler(pipeline, cfg.version, args.port)
    )
    print(f"SafeFish {cfg.version} en http://127.0.0.1:{args.port} (Ctrl+C para cortar)")
    with contextlib.suppress(KeyboardInterrupt):
        srv.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
