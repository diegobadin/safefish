"""Embeddings con caché persistente y conteo de tokens (M-ING.5, M-RET)."""

from __future__ import annotations

from functools import cache

import numpy as np
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings

from safefish.cache import KVCache, cache_enabled, stable_hash
from safefish.config import CACHE_DIR, EmbeddingsCfg
from safefish.observability import current_tracker


@cache
def _encoder():
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    try:
        return len(_encoder().encode(text))
    except Exception:  # sin red para bajar el encoding: estimación
        return max(1, len(text) // 4)


class CachedEmbeddings(Embeddings):
    """Envuelve un `Embeddings` con caché SQLite por (modelo, texto) y reporta tokens al
    `UsageTracker` activo (los aciertos de caché se marcan para el costo pagado)."""

    def __init__(self, inner: Embeddings, model: str, kv: KVCache | None, batch_size: int = 256):
        self.inner, self.model, self.kv, self.batch_size = inner, model, kv, batch_size

    def _key(self, text: str) -> str:
        return stable_hash({"model": self.model, "text": text})

    def _report(self, texts: list[str], cached: bool) -> None:
        tracker = current_tracker()
        if tracker is not None and texts:
            tracker.add_embedding(self.model, sum(count_tokens(t) for t in texts), cached)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        keys = [self._key(t) for t in texts]
        found = self.kv.get_many(keys) if self.kv else {}
        missing = [i for i, k in enumerate(keys) if k not in found]
        new: dict[str, bytes] = {}
        for start in range(0, len(missing), self.batch_size):
            idxs = missing[start : start + self.batch_size]
            vecs = self.inner.embed_documents([texts[i] for i in idxs])
            for i, v in zip(idxs, vecs, strict=True):
                new[keys[i]] = np.asarray(v, dtype=np.float32).tobytes()
        if self.kv and new:
            self.kv.set_many(new)
        missing_set = set(missing)
        self._report([texts[i] for i in missing], cached=False)
        self._report([t for i, t in enumerate(texts) if i not in missing_set], cached=True)
        allv = {**found, **new}
        return [np.frombuffer(allv[k], dtype=np.float32).astype(float).tolist() for k in keys]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def make_embeddings(cfg: EmbeddingsCfg, *, fake: bool = False) -> Embeddings:
    """`fake=True` usa embeddings determinísticos por hash (tests sin API)."""
    if fake:
        inner: Embeddings = DeterministicFakeEmbedding(size=256)
        model = "fake-256"
    else:
        import os

        from langchain_openai import OpenAIEmbeddings

        # La clave de OpenAI para embeddings va aparte (OPENAI_EMBEDDINGS_API_KEY): así se puede
        # dar una clave que sirva solo para esto, sin que los clientes de chat la tomen del entorno.
        clave = os.environ.get("OPENAI_EMBEDDINGS_API_KEY") or os.environ.get("OPENAI_API_KEY")
        inner = OpenAIEmbeddings(model=cfg.model, api_key=clave)
        model = cfg.model
    kv = KVCache(CACHE_DIR / "embeddings.sqlite") if cache_enabled() and not fake else None
    return CachedEmbeddings(inner, model, kv)
