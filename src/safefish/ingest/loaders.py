"""Loaders de fuentes (M-ING.1).

Las fuentes se declaran en un registro YAML (`data/sources.yaml`), con *provenance* explícita
(§5.4): cada documento sale con `source_id`, `doc_tipo`, `fuente_tipo` y `fuente_url`, más la
metadata que el curador declare (modalidad, canal, entidad, campaign_id, label, fecha).

Loaders soportados: `web` (WebBaseLoader), `pdf` (PyPDFLoader), `csv` (CSVLoader),
`text` (archivos .txt/.md, con front matter YAML opcional) y `taxonomia` (señales de
`taxonomias.yaml` como documentos de la colección `senales`).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field

from safefish.schemas import DocTipo
from safefish.taxonomia import load_taxonomia, senal_como_documento

DEFAULT_USER_AGENT = "safefish-ingest/0.1"


class CsvSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content_column: str = "texto"
    id_column: str | None = None
    metadata_columns: list[str] = Field(default_factory=list)
    delimiter: str = ","
    list_separator: str = "|"  # para columnas multivaluadas como `canal`


class SourceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    tipo: DocTipo
    loader: Literal["web", "pdf", "csv", "text", "taxonomia"]
    path: str | None = None  # archivo o glob, relativo al YAML de fuentes
    urls: list[str] = Field(default_factory=list)
    fuente_tipo: str = "desconocido"  # alerta_oficial | banco | prensa | dataset | recolectado | sintetico | traducido
    fuente_url: str | None = None
    pii: bool | None = None  # None = default por tipo (IngestCfg.pii_por_tipo)
    metadata: dict = Field(default_factory=dict)
    csv: CsvSpec | None = None
    enabled: bool = True


class SourcesFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sources: list[SourceSpec]


def load_sources_file(path: str | Path) -> tuple[list[SourceSpec], Path]:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {"sources": []}
    return SourcesFile.model_validate(data).sources, path.parent


# ---------------------------------------------------------------------------


def _base_metadata(spec: SourceSpec) -> dict:
    return {
        "source_id": spec.id,
        "doc_tipo": spec.tipo.value,
        "fuente_tipo": spec.fuente_tipo,
        "fuente_url": spec.fuente_url,
        **spec.metadata,
    }


def _split_front_matter(text: str) -> tuple[dict, str]:
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            meta = yaml.safe_load(text[4:end]) or {}
            return meta, text[end + 4 :].lstrip("\n")
    return {}, text


def _resolve(spec: SourceSpec, base_dir: Path) -> list[Path]:
    if not spec.path:
        raise ValueError(f"La fuente {spec.id} ({spec.loader}) necesita `path`")
    p = Path(spec.path)
    root = p if p.is_absolute() else base_dir / p
    is_glob = any(ch in spec.path for ch in "*?[")
    files = sorted(root.parent.glob(root.name)) if is_glob else [root]
    missing = [f for f in files if not f.exists()]
    if not files or missing:
        raise FileNotFoundError(f"Fuente {spec.id}: no existe {missing or root}")
    return files


def _load_text(spec: SourceSpec, base_dir: Path) -> list[Document]:
    docs = []
    for f in _resolve(spec, base_dir):
        meta, body = _split_front_matter(f.read_text(encoding="utf-8"))
        if "id" in meta:  # el id curado del front matter es el doc_id (p. ej. C-017)
            meta["doc_id"] = str(meta.pop("id"))
        docs.append(
            Document(
                page_content=body, metadata={**_base_metadata(spec), "archivo": f.name, **meta}
            )
        )
    return docs


def _load_pdf(spec: SourceSpec, base_dir: Path) -> list[Document]:
    from langchain_community.document_loaders import PyPDFLoader

    docs = []
    for f in _resolve(spec, base_dir):
        for d in PyPDFLoader(str(f), mode="single").load():
            docs.append(
                Document(
                    page_content=d.page_content,
                    metadata={**_base_metadata(spec), "archivo": f.name},
                )
            )
    return docs


def _load_web(spec: SourceSpec) -> list[Document]:
    from langchain_community.document_loaders import WebBaseLoader

    if not spec.urls:
        raise ValueError(f"La fuente web {spec.id} necesita `urls`")
    loader = WebBaseLoader(
        web_paths=spec.urls,
        header_template={"User-Agent": os.getenv("USER_AGENT", DEFAULT_USER_AGENT)},
        bs_get_text_kwargs={"separator": "\n", "strip": True},
        raise_for_status=True,
        show_progress=False,
    )
    return [
        Document(
            page_content=d.page_content,
            metadata={
                **_base_metadata(spec),
                "fuente_url": d.metadata.get("source") or spec.fuente_url,
                "titulo": d.metadata.get("title"),
            },
        )
        for d in loader.load()
    ]


def _load_csv(spec: SourceSpec, base_dir: Path) -> list[Document]:
    from langchain_community.document_loaders import CSVLoader

    cs = spec.csv or CsvSpec()
    meta_cols = [c for c in [cs.id_column, *cs.metadata_columns] if c]
    docs = []
    for f in _resolve(spec, base_dir):
        loader = CSVLoader(
            str(f),
            content_columns=[cs.content_column],
            metadata_columns=meta_cols,
            csv_args={"delimiter": cs.delimiter},
            encoding="utf-8",
        )
        for d in loader.load():
            # CSVLoader antepone "columna: " al contenido; con una sola columna se quita.
            prefix = f"{cs.content_column}: "
            content = d.page_content.removeprefix(prefix)
            row_meta = {}
            for c in cs.metadata_columns:
                v = d.metadata.get(c)
                if v in ("", None):
                    continue
                row_meta[c] = v.split(cs.list_separator) if cs.list_separator in v else v
            meta = {
                **_base_metadata(spec),
                **row_meta,
                "archivo": f.name,
                "fila": d.metadata.get("row"),
            }
            if cs.id_column and d.metadata.get(cs.id_column):
                meta["doc_id"] = str(d.metadata[cs.id_column])
            docs.append(Document(page_content=content, metadata=meta))
    return docs


def _load_taxonomia(spec: SourceSpec) -> list[Document]:
    tax = load_taxonomia()["senales"]
    return [
        Document(
            page_content=senal_como_documento(sid),
            metadata={
                **_base_metadata(spec),
                "doc_id": f"S-{sid}",
                "signal_id": sid,
                "severidad": s["severidad"],
                "tipo_senal": s["tipo"],
            },
        )
        for sid, s in tax.items()
    ]


def load_source(spec: SourceSpec, base_dir: Path) -> list[Document]:
    match spec.loader:
        case "text":
            return _load_text(spec, base_dir)
        case "pdf":
            return _load_pdf(spec, base_dir)
        case "web":
            return _load_web(spec)
        case "csv":
            return _load_csv(spec, base_dir)
        case "taxonomia":
            return _load_taxonomia(spec)
    raise ValueError(f"Loader desconocido: {spec.loader}")
