"""Indexado en Chroma (M-ING.5) y apertura de snapshots para retrieval.

Layout `single` (V1, V2a): una colección con los cuatro tipos. Layout `per_tipo` (V2+): una
colección por tipo de documento (`<snapshot>_casos`, `_mensajes`, `_senales`, `_guias`); el
manifest las lista en `collections`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from safefish.schemas import DocTipo

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


COLECCION_POR_TIPO = {DocTipo.CASO: "casos", DocTipo.MENSAJE: "mensajes", DocTipo.SENAL: "senales", DocTipo.GUIA: "guias"}  # fmt: skip


def collection_names(base: str, layout: str) -> dict[str, str]:
    """`{campo del bundle: colección de Chroma}`. En `single` todas apuntan a la misma."""
    if layout == "single":
        return {campo: base for campo in COLECCION_POR_TIPO.values()}
    return {campo: f"{base}_{campo}" for campo in COLECCION_POR_TIPO.values()}


def build_indexes(
    chunks: Sequence[Document], embeddings: Embeddings, persist_dir: Path, base: str, layout: str
) -> dict[str, str]:
    names = collection_names(base, layout)
    if layout == "single":
        build_index(chunks, embeddings, persist_dir, base)
        return names
    for tipo, campo in COLECCION_POR_TIPO.items():
        subset = [c for c in chunks if c.metadata["doc_tipo"] == tipo.value]
        if subset:
            build_index(subset, embeddings, persist_dir, names[campo])
    return names


def open_collection(snapshot_dir: Path, embeddings: Embeddings, name: str) -> Chroma | None:
    """Abre una colección existente del snapshot, o None si no se creó (tipo sin documentos)."""
    import chromadb

    client = chromadb.PersistentClient(path=str(snapshot_dir / "chroma"))
    if name not in {c.name for c in client.list_collections()}:
        return None
    return Chroma(
        client=client,
        collection_name=name,
        embedding_function=embeddings,
        create_collection_if_not_exists=False,
    )


def check_embeddings_model(manifest: dict, embeddings_model: str, snapshot: str) -> None:
    built_with = manifest["embeddings"]["model"]
    if built_with != embeddings_model:
        raise ValueError(
            f"El snapshot {snapshot} se construyó con {built_with} y el config pide "
            f"{embeddings_model}: reconstruí la KB o cambiá el config"
        )
