"""Retriever denso por colección (M-RET.2, V2+): una colección por tipo, `top_n` por colección.

La query se embebe una sola vez y se busca por vector en cada colección. Una colección con
`top_n = 0` (o inexistente en el snapshot) no se consulta: en V2 las guías no van al contexto
del razonador porque las selecciona el planner por metadata (M-GEN.2).
"""

from __future__ import annotations

import time

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from safefish.concurrency import CHROMA_LOCK
from safefish.schemas import EvidenceItem, ExtractedMessage, RetrievalBundle

CAMPOS = ("casos", "mensajes", "senales", "guias")


def to_item(doc, distance: float, coleccion: str) -> EvidenceItem:
    md = dict(doc.metadata)
    return EvidenceItem(
        id=md.get("chunk_id", doc.id or ""),
        coleccion=coleccion,
        texto=doc.page_content,
        metadata=md,
        score_dense=round(1.0 - float(distance), 6),  # distancia coseno → similitud
    )


class MultiCollectionRetriever:
    def __init__(
        self,
        stores: dict[str, Chroma | None],
        top_n: dict[str, int],
        embeddings: Embeddings,
        names: dict[str, str],
    ):
        unknown = set(top_n) - set(CAMPOS)
        if unknown:
            raise ValueError(f"top_n con colecciones desconocidas: {sorted(unknown)}")
        self.stores, self.top_n, self.embeddings, self.names = stores, top_n, embeddings, names

    def retrieve(self, m: ExtractedMessage) -> RetrievalBundle:
        query = m.texto
        t0 = time.perf_counter()
        vec = self.embeddings.embed_query(query)
        timings = {"embed_query": round((time.perf_counter() - t0) * 1000, 1)}
        bundle = RetrievalBundle(queries=[query])
        for campo in CAMPOS:
            n, store = self.top_n.get(campo, 0), self.stores.get(campo)
            if n <= 0 or store is None:
                continue
            t0 = time.perf_counter()
            with CHROMA_LOCK:
                hits = store.similarity_search_by_vector_with_relevance_scores(vec, k=n)
            timings[f"dense_{campo}"] = round((time.perf_counter() - t0) * 1000, 1)
            getattr(bundle, campo).extend(to_item(d, dist, self.names[campo]) for d, dist in hits)
        bundle.timings_ms = timings
        return bundle
