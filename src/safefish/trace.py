"""Traza paso a paso de un análisis, para el panel de depuración de la UI.

Es opcional y no cambia el resultado: los módulos llaman a `traza(...)` y, si no hay una traza
activa (evaluación, tests), la llamada no hace nada. La UI abre una con `trazando()` alrededor de
`pipeline.run` y devuelve los pasos junto con la respuesta.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

MAX_TEXTO = 240


@dataclass
class Traza:
    pasos: list[dict] = field(default_factory=list)
    t0: float = field(default_factory=time.perf_counter)

    def agregar(self, etapa: str, datos: dict) -> None:
        ms = round((time.perf_counter() - self.t0) * 1000)
        self.pasos.append({"etapa": etapa, "t_ms": ms, **datos})


_actual: ContextVar[Traza | None] = ContextVar("safefish_traza", default=None)


@contextmanager
def trazando() -> Iterator[Traza]:
    t = Traza()
    token = _actual.set(t)
    try:
        yield t
    finally:
        _actual.reset(token)


def activa() -> bool:
    return _actual.get() is not None


def traza(etapa: str, **datos) -> None:
    t = _actual.get()
    if t is not None:
        t.agregar(etapa, datos)


def recorte(texto: str | None, n: int = MAX_TEXTO) -> str:
    t = " ".join((texto or "").split())
    return t if len(t) <= n else t[: n - 1] + "…"


def item(it, rank: int | None = None) -> dict:
    """Un EvidenceItem en forma legible: identificador, etiquetas, puntajes y un extracto."""
    md = it.metadata or {}
    d = {
        "doc": str(md.get("doc_id") or md.get("signal_id") or it.id),
        "modalidad": md.get("modalidad"),
        "label": md.get("label"),
        "situacion": md.get("situacion"),
        "denso": it.score_dense,
        "bm25": it.score_bm25,
        "rrf": md.get("score_rrf"),
        "rerank": it.score_rerank,
        "via": md.get("via"),
        "texto": recorte(it.texto),
    }
    if rank is not None:
        d["rank"] = rank
    return {k: v for k, v in d.items() if v is not None}
