"""Retriever denso (M-RET.2, V1): una colección, top-k, query = texto crudo.

En V1 la KB es una sola colección con documentos de los cuatro tipos; cada chunk recuperado
se rutea al campo del `RetrievalBundle` que corresponde a su `doc_tipo` (ADR 002). Así el
contrato es el mismo que en V2+ y las métricas de retrieval por colección funcionan igual.
"""

from __future__ import annotations

import time

from langchain_chroma import Chroma

from safefish.concurrency import CHROMA_LOCK
from safefish.schemas import DocTipo, EvidenceItem, ExtractedMessage, RetrievalBundle

_FIELD = {DocTipo.CASO: "casos", DocTipo.MENSAJE: "mensajes", DocTipo.SENAL: "senales", DocTipo.GUIA: "guias"}  # fmt: skip


class DenseRetriever:
    def __init__(self, store: Chroma, top_k: int, collection: str):
        self.store, self.top_k, self.collection = store, top_k, collection

    def retrieve(self, m: ExtractedMessage) -> RetrievalBundle:
        query = m.texto
        t0 = time.perf_counter()
        with CHROMA_LOCK:
            hits = self.store.similarity_search_with_score(query, k=self.top_k)
        ms = (time.perf_counter() - t0) * 1000
        bundle = RetrievalBundle(queries=[query], timings_ms={"dense": round(ms, 1)})
        for doc, distance in hits:  # distancia coseno → similitud
            md = dict(doc.metadata)
            item = EvidenceItem(
                id=md.get("chunk_id", doc.id or ""),
                coleccion=self.collection,
                texto=doc.page_content,
                metadata=md,
                score_dense=round(1.0 - float(distance), 6),
            )
            getattr(bundle, _FIELD[DocTipo(md["doc_tipo"])]).append(item)
        return bundle


class NullRetriever:
    """V0: sin recuperación."""

    def retrieve(self, m: ExtractedMessage) -> RetrievalBundle:
        return RetrievalBundle(queries=[])
