"""Descarga (`ingest.fetch`), conversión de datasets (`ingest.datasets`), loader `html` y la
coherencia entre `data/descargas.yaml` y `data/sources.yaml`. Sin red."""

from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import pytest
from langchain_core.documents import Document

from safefish.config import resolve_path
from safefish.ingest import datasets, fetch
from safefish.ingest.clean import dedupe_exact
from safefish.ingest.loaders import (
    _load_html,
    html_to_text,
    load_sources_file,
    unir_palabras_sueltas,
)

# ---------------------------------------------------------------------------
# Manifiestos versionados
# ---------------------------------------------------------------------------


def test_manifiesto_de_descargas_es_valido():
    ds = fetch.load_manifest(resolve_path("data/descargas.yaml"))
    assert len(ds) >= 20
    for d in ds:
        if d.url:
            assert d.url.startswith("https://")
        assert d.grupo in {"es", "en", "ref"}
    assert {d.id for d in ds} >= {"spaphish", "spearphishmx", "bcra-reclamo"}


def test_sources_apunta_a_archivos_que_baja_make_data():
    """Cada fuente activa con archivo local sale de una descarga declarada (o del CSV común)."""
    ds = fetch.load_manifest(resolve_path("data/descargas.yaml"))
    esperados = {f"raw/web/{d.id}.html" for d in ds if d.tipo == "web"}
    esperados |= {f"raw/{d.dir}/{d.archivo}" for d in ds if d.tipo == "http" and not d.extraer}
    esperados |= {f"interim/mensajes/{i}.csv" for i in datasets.CONVERSORES}
    specs, _ = load_sources_file(resolve_path("data/sources.yaml"))
    for s in specs:
        if s.enabled and s.loader in {"html", "pdf", "csv"} and s.id != "msg-recolectados":
            assert s.path in esperados, f"{s.id}: {s.path} no lo genera make data"


# ---------------------------------------------------------------------------
# Red: allowlist, redirecciones, hash
# ---------------------------------------------------------------------------


class _Resp:
    def __init__(self, status=200, body=b"", headers=None, location=None):
        self.status_code, self._body = status, body
        self.headers = dict(headers or {})
        if location:
            self.headers["Location"] = location
        self.is_redirect = location is not None

    def iter_content(self, n):
        for i in range(0, len(self._body), n):
            yield self._body[i : i + n]

    def close(self):
        pass


class _Session:
    def __init__(self, respuestas: dict[str, _Resp]):
        self.respuestas, self.headers, self.pedidos = respuestas, {}, []

    def get(self, url, **kw):
        self.pedidos.append((url, kw.get("headers", {})))
        return self.respuestas[url]


def _cliente(respuestas, hosts=("ok.example",)):
    c = fetch.Cliente(set(hosts), session=_Session(respuestas))
    c._ultimo = {}
    return c


def test_cliente_rechaza_hosts_fuera_de_la_allowlist_y_http():
    c = _cliente({})
    with pytest.raises(fetch.FetchError):
        c.get("https://otro.example/x")
    with pytest.raises(fetch.FetchError):
        c.get("http://ok.example/x")


def test_cliente_sigue_redirecciones_https_y_corta_las_http(monkeypatch):
    monkeypatch.setattr(fetch, "PAUSA_S", 0)
    c = _cliente({
        "https://ok.example/a": _Resp(302, location="https://cdn.ok.example/a"),
        "https://cdn.ok.example/a": _Resp(200, b"hola", {"Content-Type": "text/plain"}),
        "https://ok.example/b": _Resp(302, location="http://cdn.ok.example/b"),
        "https://ok.example/c": _Resp(302, location="https://bucket.s3.amazonaws.com/c"),
        "https://bucket.s3.amazonaws.com/c": _Resp(200, b"zip", {"Content-Type": "x"}),
    })  # fmt: skip
    assert c.get("https://ok.example/a")[0] == b"hola"
    assert c.get("https://ok.example/c")[0] == b"zip"
    with pytest.raises(fetch.FetchError, match="HTTPS"):
        c.get("https://ok.example/b")


def test_cliente_no_sigue_redirecciones_a_hosts_ajenos(monkeypatch):
    monkeypatch.setattr(fetch, "PAUSA_S", 0)
    c = _cliente({
        "https://ok.example/a": _Resp(302, location="https://otro.example/a"),
        "https://ok.example/b": _Resp(302, location="https://127.0.0.1/admin"),
        "https://ok.example/c": _Resp(302, location="https://evilok.example/c"),
    })  # fmt: skip
    for p in "abc":
        with pytest.raises(fetch.FetchError, match="allowlist"):
            c.get(f"https://ok.example/{p}")


def test_cliente_corta_archivos_demasiado_grandes(monkeypatch):
    monkeypatch.setattr(fetch, "PAUSA_S", 0)
    c = _cliente({"https://ok.example/big": _Resp(200, b"x" * 2000)})
    with pytest.raises(fetch.FetchError, match="supera"):
        c.get("https://ok.example/big", max_bytes=1000)


def test_token_de_hf_va_solo_a_huggingface(monkeypatch):
    monkeypatch.setattr(fetch, "PAUSA_S", 0)
    monkeypatch.setenv("HF_TOKEN", "secreto")
    c = _cliente(
        {
            "https://huggingface.co/f": _Resp(302, location="https://cdn-lfs.hf.co/f"),
            "https://cdn-lfs.hf.co/f": _Resp(200, b"ok"),
        },
        hosts=("huggingface.co",),
    )
    c.get("https://huggingface.co/f")
    (u1, h1), (u2, h2) = c.s.pedidos
    assert h1.get("Authorization") == "Bearer secreto"
    assert "Authorization" not in h2


def _desc(**kw) -> fetch.Descarga:
    return fetch.Descarga(**{"id": "x", "tipo": "http", "url": "https://ok.example/f.csv",
                             "archivo": "f.csv", "destino": "d", **kw})  # fmt: skip


def test_fetch_verifica_sha256_y_es_idempotente(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "PAUSA_S", 0)
    body = b"a,b\n1,2\n"
    c = _cliente({"https://ok.example/f.csv": _Resp(200, body)})
    with pytest.raises(fetch.FetchError, match="SHA-256"):
        fetch.fetch_one(c, _desc(sha256="0" * 64), tmp_path, {})
    d = _desc(sha256=fetch.sha256(body))
    reg = {"x": fetch.fetch_one(c, d, tmp_path, {})}
    assert (tmp_path / "d" / "f.csv").read_bytes() == body
    fetch.fetch_one(c, d, tmp_path, reg)  # ya está: no vuelve a pedir
    assert len(c.s.pedidos) == 2


def test_web_guarda_la_procedencia_y_el_loader_la_usa(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "PAUSA_S", 0)
    html = "<html><title>T</title><body><nav>Menú</nav><main><p>Nunca compartas el código.</p><script>x()</script></main></body></html>"  # fmt: skip
    url = "https://ok.example/alerta"
    c = _cliente({url: _Resp(200, html.encode(), {"Content-Type": "text/html; charset=utf-8"})})
    fetch.fetch_one(c, fetch.Descarga(id="alerta", tipo="web", url=url), tmp_path, {})
    from safefish.ingest.loaders import SourceSpec

    spec = SourceSpec(id="s", tipo="caso", loader="html", path="web/alerta.html")
    (doc,) = _load_html(spec, tmp_path)
    assert doc.page_content == "Nunca compartas el código."
    assert doc.metadata["fuente_url"] == url and doc.metadata["titulo"] == "T"


def test_web_rechaza_lo_que_no_es_html(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "PAUSA_S", 0)
    url = "https://ok.example/x"
    c = _cliente({url: _Resp(200, b"%PDF", {"Content-Type": "application/pdf"})})
    with pytest.raises(fetch.FetchError, match="HTML"):
        fetch.fetch_one(c, fetch.Descarga(id="x", tipo="web", url=url), tmp_path, {})


# ---------------------------------------------------------------------------
# ZIP
# ---------------------------------------------------------------------------


def _zip(miembros: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for n, b in miembros.items():
            z.writestr(n, b)
    return buf.getvalue()


def test_zip_extrae_solo_lo_pedido_y_sin_salir_del_destino(tmp_path):
    data = _zip({"../../evil.csv": b"x", "sub/datos.csv": b"y", "run.py": b"import os",
                 "__MACOSX/._datos.csv": b"z"})  # fmt: skip
    dest = tmp_path / "dest"
    out = fetch.extract_zip(data, dest, [".csv"])
    assert sorted(out) == ["datos.csv", "evil.csv"]
    assert sorted(p.name for p in dest.iterdir()) == ["datos.csv", "evil.csv"]
    assert not (tmp_path / "evil.csv").exists()


def test_zip_sin_lo_pedido_falla(tmp_path):
    with pytest.raises(fetch.FetchError):
        fetch.extract_zip(_zip({"a.py": b""}), tmp_path, [".csv"])


# ---------------------------------------------------------------------------
# Conversores de datasets
# ---------------------------------------------------------------------------


def _write(path: Path, text: str, enc="utf-8"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode(enc))


def _leer(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_spaphish_y_golden_disjunto(tmp_path):
    filas = "\n".join(f'h{i:04d},Asunto {i},"Cuerpo, {i}",2020,1,[],0,,0,,0,{i % 2}' for i in range(200))  # fmt: skip
    _write(tmp_path / "raw/datasets/spaphish/Spaphish dataset - DiB.csv",
           "﻿hash,subject,body,date,url_count,urls,attachments_count,attachments_types,"
           "attachments_total_size,attachments_sizes,hops_count,Label\n" + filas)  # fmt: skip
    datasets.prepare("spaphish", tmp_path / "raw", tmp_path / "interim")
    kb = _leer(tmp_path / "interim/mensajes/spaphish.csv")
    gold = _leer(tmp_path / "interim/golden_candidatos/spaphish.csv")
    assert len(kb) + len(gold) == 200 and 5 <= len(gold) <= 40
    assert not {r["id"] for r in kb} & {r["id"] for r in gold}
    r = next(r for r in kb + gold if r["id"] == "spaphish-h0001")
    assert r["texto"] == "Asunto: Asunto 1\n\nCuerpo, 1" and r["label"] == "estafa" and r["canal"] == "email"  # fmt: skip
    # determinístico
    datasets.prepare("spaphish", tmp_path / "raw", tmp_path / "interim2")
    assert _leer(tmp_path / "interim2/mensajes/spaphish.csv") == kb


def test_spam_generico_no_se_etiqueta_como_estafa(tmp_path):
    _write(tmp_path / "raw/datasets/uci-sms-spam/SMSSpamCollection", "ham\tok\nspam\tWIN £100\n")
    datasets.prepare("uci-sms-spam", tmp_path / "raw", tmp_path / "i")
    u = {r["texto"]: r for r in _leer(tmp_path / "i/mensajes/uci-sms-spam.csv")}
    assert u["ok"]["label"] == "legitimo"
    assert u["WIN £100"]["label"] == "" and u["WIN £100"]["label_origen"] == "spam"


def test_spearphishmx_separado_por_punto_y_coma(tmp_path):
    _write(tmp_path / "raw/datasets/spearphishmx/SpearPhishMX_dataset_publico.csv",
           'hash;id_original;etiqueta;asunto;cuerpo_sanitizado\nabc;1;1;Urgente;"Hola;\nverificá"\n')  # fmt: skip
    datasets.prepare("spearphishmx", tmp_path / "raw", tmp_path / "i")
    (r,) = _leer(tmp_path / "i/mensajes/spearphishmx.csv")
    assert r["texto"] == "Asunto: Urgente\n\nHola;\nverificá" and r["fuente_pais"] == "MX"


def test_dataset_no_bajado_no_rompe(tmp_path):
    assert "no está bajado" in datasets.prepare("spaphish", tmp_path, tmp_path / "i")


# ---------------------------------------------------------------------------
# Loaders y limpieza
# ---------------------------------------------------------------------------


def test_html_to_text_saca_navegacion_y_scripts():
    t, titulo = html_to_text("<title>X</title><header>Logo</header><article>Uno<br>Dos</article><footer>©</footer>")  # fmt: skip
    assert t == "Uno\nDos" and titulo == "X"


def test_pdf_palabra_por_linea_se_une():
    assert unir_palabras_sueltas("Nunca\ncompartas\nel\ncódigo\n\nLlamá\nal\nbanco") == "Nunca compartas el código\n\nLlamá al banco"  # fmt: skip
    normal = "Primera línea con varias palabras\nSegunda línea también"
    assert unir_palabras_sueltas(normal) == normal


def test_dedupe_exacto_por_tipo_permite_misma_pagina_como_caso_y_guia():
    docs = [Document(page_content="misma página", metadata={"doc_tipo": t}) for t in ("caso", "guia", "caso")]  # fmt: skip
    keep, drop = dedupe_exact(docs)
    assert [d.metadata["doc_tipo"] for d in keep] == ["caso", "guia"] and len(drop) == 1


def test_registro_descarta_fuentes_que_salieron_del_manifiesto(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / fetch.REGISTRO).write_text(json.dumps({"vieja": {}, "x": {}}))
    m = tmp_path / "m.yaml"
    m.write_text("descargas:\n  - {id: x, tipo: web, url: 'https://ok.example/', grupo: es}\n")
    reg, _ = fetch.run(m, tmp_path / "raw", {"es"}, dry_run=True)
    assert set(reg) == {"x"}
    assert urlparse("https://ok.example/").hostname in fetch.hosts_de(fetch.load_manifest(m))
