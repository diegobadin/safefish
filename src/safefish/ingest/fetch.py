"""Descarga de las fuentes públicas a `data/raw/` (no se versiona).

    python -m safefish.ingest.fetch [--grupos es,ref] [--solo id1,id2] [--force] [--dry-run]

Lee `data/descargas.yaml`, baja cada fuente y deja un registro en `data/raw/_descargas.json`
(URL, SHA-256, bytes, fecha, licencia). Después convierte los datasets de mensajes a un CSV
común (`safefish.ingest.datasets`). Es idempotente: lo que ya está y coincide no se vuelve a
bajar (`--force` lo fuerza).

Controles (los archivos bajados son datos no confiables):
- solo HTTPS y solo los hosts que aparecen en el manifiesto (también tras redirecciones);
- tamaño máximo por archivo y escritura atómica (`.part` → rename);
- SHA-256 verificado cuando la fuente lo publica (Mendeley) o el manifiesto lo fija;
- de un ZIP se extraen solo los miembros pedidos, aplanados, sin rutas absolutas ni `..`;
- las páginas HTML se guardan sin contenido activo (scripts, iframes, eventos);
- nunca se ejecuta ni se importa nada de lo bajado.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import ipaddress
import json
import os
import re
import sys
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Literal
from urllib.parse import quote, urlparse

import requests
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from urllib3.exceptions import LocationParseError
from urllib3.util import parse_url

USER_AGENT = "safefish-fetch/0.1 (proyecto academico)"
MAX_BYTES = 60 * 1024 * 1024
MAX_EXTRACT_BYTES = 100 * 1024 * 1024
TIMEOUT_S = 60
PAUSA_S = 1.0  # entre pedidos al mismo host
REGISTRO = "_descargas.json"
MARCA_HTML = "safefish-fuente"
# CDN exactos a los que se puede redirigir, según el host de la URL pedida. Mendeley: bucket
# observado en las descargas reales (2026-11). Hugging Face: CDN de archivos LFS/Xet.
CDN_POR_ORIGEN: dict[str, frozenset[str]] = {
    "data.mendeley.com": frozenset({
        "prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com",
        "prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com",
    }),
    "huggingface.co": frozenset({"cdn-lfs.hf.co", "cdn-lfs-us-1.hf.co", "cas-bridge.xethub.hf.co"}),
}  # fmt: skip


class Descarga(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    tipo: Literal["mendeley", "hf", "http", "web"]
    licencia: str = "no visible"
    grupo: str = "es"
    destino: str | None = None
    # mendeley
    dataset: str | None = None
    version: int | None = None
    # hf
    repo: str | None = None
    revision: str | None = None
    # mendeley / hf
    archivos: list[str] = Field(default_factory=list)
    # http / web
    url: str | None = None
    archivo: str | None = None
    sha256: str | None = None
    # zip
    extraer: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _campos(self):
        req = {"mendeley": ("dataset", "version", "archivos"), "hf": ("repo", "revision", "archivos"),
               "http": ("url", "archivo", "destino"), "web": ("url",)}[self.tipo]  # fmt: skip
        if missing := [c for c in req if not getattr(self, c)]:
            raise ValueError(f"{self.id}: faltan {missing}")
        if self.url and not self.url.startswith("https://"):
            raise ValueError(f"{self.id}: solo HTTPS")
        return self

    @property
    def dir(self) -> str:
        return self.destino or ("web" if self.tipo == "web" else f"datasets/{self.id}")


class Manifiesto(BaseModel):
    model_config = ConfigDict(extra="forbid")
    descargas: list[Descarga]


def load_manifest(path: str | Path) -> list[Descarga]:
    with open(path, encoding="utf-8") as f:
        m = Manifiesto.model_validate(yaml.safe_load(f))
    ids = [d.id for d in m.descargas]
    if dup := {i for i in ids if ids.count(i) > 1}:
        raise ValueError(f"ids repetidos: {sorted(dup)}")
    return m.descargas


# ---------------------------------------------------------------------------
# Red
# ---------------------------------------------------------------------------


class FetchError(RuntimeError):
    pass


class Cliente:
    """GET con allowlist de hosts, tope de tamaño y pausa por host."""

    def __init__(self, hosts: set[str], session: requests.Session | None = None):
        self.hosts = hosts
        self.s = session or requests.Session()
        self.s.headers["User-Agent"] = USER_AGENT
        self._ultimo: dict[str, float] = {}

    @staticmethod
    def _host(url: str) -> str:
        """Host de la URL tal como lo va a usar `requests` (urllib3), no `urlparse`: así la
        allowlist y el envío del token deciden sobre el mismo host. Rechaza las formas que los
        parsers interpretan distinto: credenciales (`user@`), barras invertidas, espacios y
        caracteres de control, y hosts que son una IP."""
        if "\\" in url or any(ord(ch) <= 0x20 or ord(ch) == 0x7F for ch in url):
            raise FetchError(f"URL con caracteres no permitidos: {url!r}")
        try:
            u = parse_url(url)
        except LocationParseError as e:
            raise FetchError(f"URL inválida: {url!r}") from e
        host = (u.host or "").lower().rstrip(".")
        if u.scheme != "https":
            raise FetchError(f"solo HTTPS: {url}")
        if u.auth or not host:
            raise FetchError(f"URL con credenciales o sin host: {url}")
        try:
            ipaddress.ip_address(host.strip("[]"))
        except ValueError:
            return host
        raise FetchError(f"URL con IP en lugar de dominio: {url}")

    def _check(self, url: str) -> str:
        host = self._host(url)
        if host not in self.hosts:
            raise FetchError(f"URL fuera de la allowlist: {url}")
        return host

    @staticmethod
    def _check_redireccion(url: str, origen: str) -> str:
        """Una redirección solo puede ir al mismo host de la URL pedida, a su variante con o
        sin `www.`, o a los CDN exactos de ese origen (S3 solo desde Mendeley, hf.co solo
        desde Hugging Face)."""
        host = Cliente._host(url)
        variantes = {origen, origen.removeprefix("www."), "www." + origen.removeprefix("www.")}
        if host in variantes or host in CDN_POR_ORIGEN.get(origen, frozenset()):
            return host
        raise FetchError(f"redirección a un host fuera de la allowlist: {url}")

    def get(self, url: str, max_bytes: int = MAX_BYTES) -> tuple[bytes, str]:
        origen = host = self._check(url)
        espera = PAUSA_S - (time.monotonic() - self._ultimo.get(origen, 0.0))
        if espera > 0:
            time.sleep(espera)
        # La URL pedida tiene que estar en la allowlist; las redirecciones se siguen a mano y
        # cada destino se valida contra el host de origen.
        for _ in range(5):
            headers = {}
            # Datasets "gated" de HF: el token va solo a huggingface.co, nunca al CDN. Se decide
            # con el mismo host validado que usa la allowlist.
            if host == "huggingface.co" and os.getenv("HF_TOKEN"):
                headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
            r = self.s.get(url, headers=headers, timeout=TIMEOUT_S, stream=True, allow_redirects=False)  # fmt: skip
            self._ultimo[origen] = time.monotonic()
            if r.is_redirect:
                url = requests.compat.urljoin(url, r.headers["Location"])
                r.close()
                host = self._check_redireccion(url, origen)
                continue
            break
        else:
            raise FetchError(f"demasiadas redirecciones: {url}")
        if r.status_code in (401, 403) and "huggingface.co" in url:
            raise FetchError(f"HTTP {r.status_code}: dataset con acceso restringido; aceptá los términos en la página del dataset y exportá HF_TOKEN")  # fmt: skip
        if r.status_code != 200:
            raise FetchError(f"HTTP {r.status_code}: {url}")
        buf = io.BytesIO()
        for chunk in r.iter_content(64 * 1024):
            buf.write(chunk)
            if buf.tell() > max_bytes:
                r.close()
                raise FetchError(f"supera {max_bytes // 2**20} MB: {url}")
        return buf.getvalue(), r.headers.get("Content-Type", "")


def hosts_de(descargas: list[Descarga]) -> set[str]:
    hosts = {"data.mendeley.com", "huggingface.co"}
    hosts |= {urlparse(d.url).hostname or "" for d in descargas if d.url}
    return hosts


# ---------------------------------------------------------------------------
# Archivos
# ---------------------------------------------------------------------------


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_name(name: str) -> str:
    base = PurePosixPath(name.replace("\\", "/")).name
    if not base or base in {".", ".."}:
        raise FetchError(f"nombre de archivo inválido: {name!r}")
    return base


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def extract_zip(data: bytes, dest: Path, quiero: list[str]) -> list[str]:
    """Extrae solo los miembros cuyo nombre (sin carpeta) coincide con un elemento de `quiero`
    o termina con él (p. ej. ".csv"). Aplana las rutas: nada puede escribirse fuera de `dest`."""
    out, total = [], 0
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            nombre = PurePosixPath(info.filename.replace("\\", "/")).name
            if nombre.startswith(".") or "__MACOSX" in info.filename:
                continue
            if not any(nombre == q or nombre.endswith(q) for q in quiero):
                continue
            total += info.file_size
            if total > MAX_EXTRACT_BYTES:
                raise FetchError("el ZIP descomprimido supera el máximo")
            write_atomic(dest / _safe_name(nombre), z.read(info))
            out.append(nombre)
    if not out:
        raise FetchError(f"el ZIP no tiene {quiero}")
    return out


# Elementos que ejecutan código o cargan contenido de terceros al abrir la página en un navegador.
_ACTIVOS_HTML = ("script", "noscript", "iframe", "frame", "frameset", "object", "embed", "applet")
_ESQUEMAS_ACTIVOS = ("javascript:", "vbscript:")


def sanitizar_html(html: str) -> str:
    """Saca el contenido activo de una página guardada: scripts, iframes y objetos embebidos,
    atributos de eventos (`onclick`, `onload`…), links `javascript:` y redirecciones por
    `<meta refresh>`. El texto visible no cambia: el loader `html` ya descarta esos elementos
    antes de extraerlo."""
    from bs4 import BeautifulSoup, Comment

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(_ACTIVOS_HTML):
        tag.decompose()
    # Comentarios condicionales de IE (`<!--[if IE]><script …>`): ese navegador los ejecuta.
    # Los demás comentarios se dejan: algunos sitios los usan para separar palabras y sacarlos
    # cambiaría el texto extraído.
    for c in soup.find_all(string=lambda s: isinstance(s, Comment)):
        if re.search(r"\[if\b|<script|<iframe|<object|<embed", c, re.I):
            c.extract()
    for tag in soup.find_all("meta"):
        if str(tag.get("http-equiv", "")).lower() == "refresh":
            tag.decompose()
    for tag in soup.find_all(True):
        for attr in list(tag.attrs):
            valor = str(tag.attrs[attr]).strip().lower()
            if attr.lower().startswith("on") or valor.startswith(_ESQUEMAS_ACTIVOS):
                del tag.attrs[attr]
    return str(soup)


def html_con_procedencia(data: bytes, url: str, cuando: str) -> bytes:
    """Antepone un comentario con la URL y la fecha (el loader `html` lo usa como `fuente_url`)
    y guarda la página sin contenido activo. El `sha256` es el de los bytes descargados."""
    marca = json.dumps({"url": url, "descargado": cuando, "sha256": sha256(data)})
    limpio = sanitizar_html(data.decode("utf-8", errors="replace"))
    return f"<!-- {MARCA_HTML}: {marca} -->\n{limpio}".encode()


# ---------------------------------------------------------------------------
# Descarga por tipo
# ---------------------------------------------------------------------------


def _mendeley_files(c: Cliente, d: Descarga) -> list[dict]:
    url = f"https://data.mendeley.com/public-api/datasets/{d.dataset}/files?folder_id=root&version={d.version}"  # fmt: skip
    data, _ = c.get(url, max_bytes=2 * 2**20)
    files = {f["filename"]: f for f in json.loads(data)}
    if missing := [a for a in d.archivos if a not in files]:
        raise FetchError(f"{d.id}: Mendeley no lista {missing}")
    return [files[a] for a in d.archivos]


def _items(c: Cliente, d: Descarga) -> list[tuple[str, str, str | None]]:
    """(nombre local, URL, sha256 esperado) de cada archivo de la fuente."""
    match d.tipo:
        case "mendeley":
            return [
                (
                    f["filename"],
                    f["content_details"]["download_url"],
                    f["content_details"]["sha256_hash"],
                )  # fmt: skip
                for f in _mendeley_files(c, d)
            ]
        case "hf":
            return [
                (
                    a,
                    f"https://huggingface.co/datasets/{d.repo}/resolve/{d.revision}/{quote(a)}",
                    None,
                )  # fmt: skip
                for a in d.archivos
            ]
        case "http":
            return [(d.archivo, d.url, d.sha256)]
        case "web":
            return [(f"{d.id}.html", d.url, d.sha256)]
    raise AssertionError(d.tipo)


def fetch_one(c: Cliente, d: Descarga, raw: Path, registro: dict, force: bool = False) -> dict:
    dest = raw / d.dir
    previo = registro.get(d.id, {})
    entradas = []
    for nombre, url, esperado in _items(c, d):
        local = dest / _safe_name(nombre.replace("/", "__") if d.tipo == "hf" else nombre)
        prev = next((a for a in previo.get("archivos", []) if a["archivo"] == local.name), None)
        presentes = [dest / x for x in prev["extraidos"]] if prev and prev.get("extraidos") else [local]  # fmt: skip
        if not force and prev and all(p.exists() for p in presentes) and esperado in (None, prev["sha256"]):  # fmt: skip
            entradas.append(prev)
            continue
        data, ctype = c.get(url)
        h = sha256(data)
        if esperado and h != esperado:
            raise FetchError(f"{d.id}/{nombre}: SHA-256 {h[:12]}… ≠ esperado {esperado[:12]}…")
        if d.tipo == "web" and "html" not in ctype.lower():
            raise FetchError(f"{d.id}: se esperaba HTML y llegó {ctype!r}")
        if nombre.lower().endswith(".pdf") and not data.startswith(b"%PDF"):
            raise FetchError(f"{d.id}: no es un PDF")
        cuando = datetime.now(UTC).isoformat(timespec="seconds")
        extraidos = []
        if d.extraer and nombre.lower().endswith(".zip"):
            extraidos = extract_zip(data, dest, d.extraer)
        else:
            write_atomic(
                local, html_con_procedencia(data, url, cuando) if d.tipo == "web" else data
            )
        # De un ZIP se guardan solo los extraídos; el registro conserva el hash del ZIP.
        entradas.append({"archivo": local.name, "url": url, "sha256": h, "bytes": len(data),
                         "descargado": cuando, "extraidos": extraidos})  # fmt: skip
    return {"id": d.id, "tipo": d.tipo, "licencia": d.licencia, "grupo": d.grupo,
            "destino": d.dir, "archivos": entradas}  # fmt: skip


def run(
    manifest: Path,
    raw: Path,
    grupos: set[str],
    solo: set[str] | None = None,
    force: bool = False,
    dry_run: bool = False,
    cliente: Cliente | None = None,
) -> tuple[dict, list[str]]:
    todas = load_manifest(manifest)
    descargas = [d for d in todas if d.grupo in grupos and (not solo or d.id in solo)]
    if solo and (faltan := solo - {d.id for d in descargas}):
        raise SystemExit(f"ids desconocidos o fuera de los grupos {sorted(grupos)}: {sorted(faltan)}")  # fmt: skip
    reg_path = raw / REGISTRO
    registro = json.loads(reg_path.read_text(encoding="utf-8")) if reg_path.exists() else {}
    vigentes = {d.id for d in todas}
    # Las fuentes que salieron del manifiesto se borran del registro.
    registro = {k: v for k, v in registro.items() if k in vigentes}
    if dry_run:
        for d in descargas:
            print(f"  {d.id:30s} {d.tipo:8s} → data/raw/{d.dir}/  [{d.licencia}]")
        return registro, []
    c = cliente or Cliente(hosts_de(descargas))
    errores = []
    for d in descargas:
        try:
            registro[d.id] = fetch_one(c, d, raw, registro, force)
            n = sum(a["bytes"] for a in registro[d.id]["archivos"])
            print(f"  ✓ {d.id:30s} {n / 1024:8.0f} KB → data/raw/{d.dir}/")
        except (FetchError, requests.RequestException, zipfile.BadZipFile, ValueError) as e:
            errores.append(f"{d.id}: {e}")
            print(f"  ✗ {d.id:30s} {e}", file=sys.stderr)
        raw.mkdir(parents=True, exist_ok=True)
        reg_path.write_text(json.dumps(registro, indent=2, ensure_ascii=False), encoding="utf-8")
    return registro, errores


def main(argv: list[str] | None = None) -> int:
    from safefish.config import resolve_path

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    ap.add_argument("--manifest", default="data/descargas.yaml")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--interim", default="data/interim")
    ap.add_argument("--grupos", default="es", help="grupos a bajar, separados por coma (es,ref)")
    ap.add_argument("--solo", default="", help="ids a bajar, separados por coma")
    ap.add_argument("--force", action="store_true", help="vuelve a bajar aunque ya esté")
    ap.add_argument("--dry-run", action="store_true", help="lista qué bajaría")
    ap.add_argument("--sin-preparar", action="store_true", help="no convierte los datasets")
    a = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # consola de Windows (cp1252)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    from dotenv import load_dotenv

    load_dotenv()  # HF_TOKEN opcional, para datasets con acceso restringido
    raw = resolve_path(a.raw)
    grupos = {g.strip() for g in a.grupos.split(",") if g.strip()}
    solo = {s.strip() for s in a.solo.split(",") if s.strip()} or None
    print(f"Fuentes ({', '.join(sorted(grupos))}):")
    _, errores = run(resolve_path(a.manifest), raw, grupos, solo, a.force, a.dry_run)
    if not a.dry_run and not a.sin_preparar:
        from safefish.ingest.datasets import prepare_all

        print("Datasets → CSV común:")
        for linea in prepare_all(raw, resolve_path(a.interim)):
            print(f"  {linea}")
    if errores:
        print(f"\n{len(errores)} fuente(s) con error (el resto quedó bajado):", file=sys.stderr)
        for e in errores:
            print(f"  - {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
