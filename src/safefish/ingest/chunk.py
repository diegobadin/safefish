"""Chunking (M-ING.4) y adaptación de metadata para Chroma.

V1: `RecursiveCharacterTextSplitter` (≈800 caracteres, overlap 15 %). Se espera que parta
casos (descripción en un chunk y frases típicas en otro): es la hipótesis que V2 resuelve con
*1 caso = 1 chunk*.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from safefish.config import ChunkingCfg
from safefish.schemas import Canal


def make_splitter(cfg: ChunkingCfg) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=cfg.separators,
        keep_separator="end",  # el punto queda con su oración
        strip_whitespace=True,
    )


def chunk_documents(docs: Sequence[Document], cfg: ChunkingCfg) -> list[Document]:
    if cfg.strategy == "per_case":
        raise NotImplementedError("per_case llega en V2 junto con el normalizador LLM (Track A)")
    splitter = make_splitter(cfg)
    chunks: list[Document] = []
    for doc in docs:
        doc_id = doc.metadata["doc_id"]
        pieces = splitter.split_text(doc.page_content)
        for i, piece in enumerate(pieces):
            chunks.append(
                Document(
                    id=f"{doc_id}#c{i:02d}",
                    page_content=piece,
                    metadata={
                        **doc.metadata,
                        "chunk_id": f"{doc_id}#c{i:02d}",
                        "chunk_index": i,
                        "n_chunks": len(pieces),
                    },
                )
            )
    return chunks


def scalarize_metadata(md: dict) -> dict:
    """Chroma guarda metadata escalar: `canal` (lista) → flags `canal_<x>`; otras listas →
    string separado por comas; `None` se omite."""
    out: dict = {}
    for k, v in md.items():
        if v is None:
            continue
        if k == "canal":
            vals = [v] if isinstance(v, str) else list(v)
            vals = [str(x).strip().lower() for x in vals]
            for c in Canal:
                out[f"canal_{c.value}"] = c.value in vals or "multi" in vals
            out["canal"] = ",".join(vals)
        elif isinstance(v, bool | int | float | str):
            out[k] = v
        elif isinstance(v, list | tuple | set):
            out[k] = ",".join(str(x) for x in v)
        else:
            out[k] = str(v)
    return out
