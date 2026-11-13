"""Indexado en Chroma (M-ING.5) y apertura de snapshots para retrieval."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

COSINE = {"hnsw": {"space": "cosine"}}


def build_index(
    chunks: Sequence[Document],
    embeddings: Embeddings,
    persist_dir: Path,
    collection: str,
    batch_size: int = 256,
) -> Chroma:
    store = Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
        collection_configuration=COSINE,
    )
    for i in range(0, len(chunks), batch_size):
        batch = list(chunks[i : i + batch_size])
        store.add_documents(batch, ids=[d.id for d in batch])
    return store


def read_manifest(snapshot_dir: Path) -> dict:
    path = snapshot_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el snapshot {snapshot_dir.name}: corré `make kb CONFIG=...` primero"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def open_index(snapshot_dir: Path, embeddings: Embeddings, embeddings_model: str) -> Chroma:
    manifest = read_manifest(snapshot_dir)
    built_with = manifest["embeddings"]["model"]
    if built_with != embeddings_model:
        raise ValueError(
            f"El snapshot {snapshot_dir.name} se construyó con {built_with} y el config pide "
            f"{embeddings_model}: reconstruí la KB o cambiá el config"
        )
    return Chroma(
        collection_name=manifest["collection"],
        embedding_function=embeddings,
        persist_directory=str(snapshot_dir / "chroma"),
        create_collection_if_not_exists=False,
    )
