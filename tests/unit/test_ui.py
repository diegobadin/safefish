"""La UI local solo atiende pedidos dirigidos a sí misma (Host y Origin): evita CSRF y DNS
rebinding desde otras páginas abiertas en el navegador. Solo sirve los archivos del front."""

import http.client
import threading
from http.server import ThreadingHTTPServer

import pytest

from safefish.ui import make_handler


class _PipelineQueFalla:
    def run(self, raw):  # pragma: no cover - no debería llamarse
        raise AssertionError("el pipeline no debe correr con un origen ajeno")


@pytest.fixture
def servidor():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(_PipelineQueFalla(), "vtest", 0))
    port = srv.server_address[1]
    srv.RequestHandlerClass = make_handler(_PipelineQueFalla(), "vtest", port)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield port
    srv.shutdown()


def _pedido(port, metodo, headers, body=b"", ruta="/"):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    c.putrequest(metodo, ruta, skip_host=True)
    for k, v in headers.items():
        c.putheader(k, v)
    c.putheader("Content-Length", str(len(body)))
    c.endheaders(body)
    return c.getresponse().status


def test_get_con_host_local(servidor):
    assert _pedido(servidor, "GET", {"Host": f"127.0.0.1:{servidor}"}) == 200


def test_get_con_host_ajeno(servidor):
    assert _pedido(servidor, "GET", {"Host": f"evil.example:{servidor}"}) == 403


def test_post_a_la_api_con_origin_ajeno(servidor):
    headers = {
        "Host": f"localhost:{servidor}",
        "Origin": "https://evil.example",
        "Content-Type": "multipart/form-data; boundary=x",
    }
    assert _pedido(servidor, "POST", headers, b"--x--\r\n", ruta="/api/analizar") == 403


def test_sirve_solo_los_archivos_del_front(servidor):
    h = {"Host": f"localhost:{servidor}"}
    assert _pedido(servidor, "GET", h, ruta="/app.js") == 200
    assert _pedido(servidor, "GET", h, ruta="/styles.css") == 200
    assert _pedido(servidor, "GET", h, ruta="/../pyproject.toml") == 404
