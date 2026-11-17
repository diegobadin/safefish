"""Advanced RAG (Fase 3, V4–V6) sobre las 4 colecciones de V2+.

Cada versión suma **una** pieza sobre la anterior y las tres comparten este retriever:

- **V4 · multi-query**: un LLM mini extrae del mensaje {resumen del guion, pedido, entidad,
  frases clave} y arma 3–4 queries además del texto crudo. Las listas se fusionan con RRF.
- **V5 · híbrido**: por cada query se suma una lista BM25 (léxica) a la densa; RRF ponderado.
  El BM25 se arma desde `chunks.jsonl` del mismo snapshot (no hace falta reconstruir la KB).
- **V6 · re-ranking**: los `top_in` candidatos fusionados de cada colección se re-ordenan
  contra el mensaje y quedan los `top_n`. El reranker es un LLM mini *listwise* (puntúa cada
  candidato 0–3); ver la nota de diseño en `docs/ablation.md` sobre por qué no un
  cross-encoder local.

RRF (Cormack et al., 2009): score(d) = Σ_listas w_l / (k + rank_l(d)), con k = 60.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from safefish.concurrency import CHROMA_LOCK
from safefish.prompting import load_prompt_file
from safefish.retrieval.multi import CAMPOS, to_item
from safefish.schemas import EvidenceItem, ExtractedMessage, RetrievalBundle
from safefish.textutils import normalize_for_match
from safefish.trace import activa, item, traza

PROMPTS_DIR = Path(__file__).with_name("prompts")
CAMPO_DE_TIPO = {"caso": "casos", "mensaje": "mensajes", "senal": "senales", "guia": "guias"}

# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

_STOP = frozenset(
    [
        "a",
        "al",
        "algo",
        "ante",
        "con",
        "contra",
        "de",
        "del",
        "desde",
        "el",
        "en",
        "entre",
        "es",
        "esa",
        "ese",
        "eso",
        "esta",
        "este",
        "esto",
        "hay",
        "la",
        "las",
        "le",
        "les",
        "lo",
        "los",
        "me",
        "mi",
        "mis",
        "muy",
        "no",
        "nos",
        "o",
        "para",
        "pero",
        "por",
        "que",
        "se",
        "si",
        "sin",
        "sobre",
        "su",
        "sus",
        "te",
        "tu",
        "tus",
        "un",
        "una",
        "uno",
        "y",
        "ya",
    ]
)


def tokenize(text: str) -> list[str]:
    """Minúsculas sin tildes (como el resto del proyecto), tokens alfanuméricos ≥ 2 chars sin
    stopwords. No hay stemming a propósito: BM25 está para los términos exactos (marcas,
    dominios, "CBU", "código") que el embedding diluye."""
    return [
        t for t in re.findall(r"\w+", normalize_for_match(text)) if len(t) > 1 and t not in _STOP
    ]


class BM25:
    """Okapi BM25 en memoria (k1 = 1.5, b = 0.75)."""

    def __init__(self, rows: Sequence[dict], k1: float = 1.5, b: float = 0.75):
        self.rows = list(rows)
        self.k1, self.b = k1, b
        docs = [tokenize(r["texto"]) for r in self.rows]
        self.len = np.array([len(d) for d in docs], dtype=float)
        self.avgdl = float(self.len.mean()) if len(docs) else 0.0
        self.tf: list[dict[str, int]] = []
        df: dict[str, int] = defaultdict(int)
        for d in docs:
            c: dict[str, int] = defaultdict(int)
            for t in d:
                c[t] += 1
            self.tf.append(c)
            for t in c:
                df[t] += 1
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.postings: dict[str, list[int]] = defaultdict(list)
        for i, c in enumerate(self.tf):
            for t in c:
                self.postings[t].append(i)

    def search(self, query: str, k: int) -> list[tuple[dict, float]]:
        scores: dict[int, float] = defaultdict(float)
        for t in set(tokenize(query)):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i in self.postings[t]:
                f = self.tf[i][t]
                norm = self.k1 * (1 - self.b + self.b * self.len[i] / (self.avgdl or 1.0))
                scores[i] += idf * f * (self.k1 + 1) / (f + norm)
        top = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]
        return [(self.rows[i], s) for i, s in top]


def load_bm25(snapshot_dir: Path) -> dict[str, BM25]:
    """Un índice BM25 por colección, desde `chunks.jsonl` del snapshot."""
    por_campo: dict[str, list[dict]] = defaultdict(list)
    with open(snapshot_dir / "chunks.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            campo = CAMPO_DE_TIPO.get(str(row["metadata"].get("doc_tipo")))
            if campo:
                por_campo[campo].append(row)
    return {campo: BM25(rows) for campo, rows in por_campo.items()}


def bm25_item(row: dict, score: float, coleccion: str) -> EvidenceItem:
    md = dict(row["metadata"])
    return EvidenceItem(
        id=md.get("chunk_id", row.get("id") or ""),
        coleccion=coleccion,
        texto=row["texto"],
        metadata=md,
        score_bm25=round(float(score), 6),
    )


# ---------------------------------------------------------------------------
# RRF
# ---------------------------------------------------------------------------


def rrf(listas: Sequence[tuple[float, Sequence[EvidenceItem]]], k: int = 60) -> list[EvidenceItem]:
    """Fusiona listas rankeadas `(peso, items)`. Conserva el mejor score denso y BM25 de cada
    item y deja el score fusionado en `metadata["score_rrf"]`. Desempata por primera aparición."""
    score: dict[str, float] = defaultdict(float)
    best: dict[str, EvidenceItem] = {}
    orden: dict[str, int] = {}
    for peso, items in listas:
        if peso <= 0:
            continue
        for rank, it in enumerate(items, start=1):
            score[it.id] += peso / (k + rank)
            orden.setdefault(it.id, len(orden))
            prev = best.get(it.id)
            if prev is None:
                best[it.id] = it.model_copy(deep=True)
                continue
            for attr in ("score_dense", "score_bm25"):
                v = getattr(it, attr)
                if v is not None and (getattr(prev, attr) is None or v > getattr(prev, attr)):
                    setattr(prev, attr, v)
    out = []
    for iid in sorted(score, key=lambda i: (-score[i], orden[i])):
        it = best[iid]
        it.metadata["score_rrf"] = round(score[iid], 6)
        out.append(it)
    return out


# ---------------------------------------------------------------------------
# Query rewriting (V4)
# ---------------------------------------------------------------------------


class _LLM(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QueryFacets(_LLM):
    resumen: str = Field(description="el guion del mensaje en una oración, sin datos personales")
    pedido: str = Field(description="qué le pide hacer al destinatario (o 'nada')")
    entidad: str | None = Field(description="empresa, organismo o persona que dice ser, o null")
    frases_clave: list[str] = Field(description="hasta 3 frases literales más características")


class QueryRewriter:
    def __init__(self, llm, prompt: str = "query_rewrite_v1", max_queries: int = 4):
        self.llm, self.prompt, self.max_queries = llm, prompt, max_queries

    @property
    def prompt_hash(self) -> str:
        return load_prompt_file(PROMPTS_DIR / f"{self.prompt}.md")[1]

    def facets(self, texto: str) -> QueryFacets | None:
        template, _ = load_prompt_file(PROMPTS_DIR / f"{self.prompt}.md")
        system, human = template.format_messages(mensaje=texto)
        r = self.llm.invoke(
            [SystemMessage(content=system.content), HumanMessage(content=human.content)],
            QueryFacets,
            stage="query_rewrite",
        )
        return r.parsed

    def queries(self, texto: str, contexto: str | None = None) -> list[str]:
        """Texto crudo + queries focalizadas (sin repetidos). Si el LLM falla, queda el crudo.
        El contexto del usuario, si lo hay, solo orienta las facetas; la primera query sigue
        siendo el mensaje tal cual."""
        entrada = f"{texto}\n\n(Contexto de quien lo recibió: {contexto})" if contexto else texto
        f = self.facets(entrada)
        out = [texto]
        facetas = f.model_dump() if f is not None else None
        if f is not None:
            ent = f"{f.entidad}: " if f.entidad else ""
            pedido = "" if f.pedido.strip().lower() in {"", "nada"} else f.pedido
            cands = [f.resumen, f"{ent}{pedido}".strip(": "), " · ".join(f.frases_clave[:3])]
            for q in cands:
                q = q.strip()
                if q and q not in out:
                    out.append(q)
        out = out[: self.max_queries]
        traza("query_rewrite", modelo=getattr(self.llm, "model", None), facetas=facetas,
              queries=out, con_contexto=bool(contexto))  # fmt: skip
        return out


# ---------------------------------------------------------------------------
# Re-ranking (V6)
# ---------------------------------------------------------------------------


class Puntaje(_LLM):
    i: int = Field(description="número del candidato")
    score: int = Field(ge=0, le=3, description="0 no relacionado · 3 mismo guion y pedido")


class Ranking(_LLM):
    puntajes: list[Puntaje]


class LLMReranker:
    """Re-ranking listwise: el LLM puntúa 0–3 cada candidato contra el mensaje. Se ordena por
    puntaje y, a igual puntaje, por el orden de la fusión (estable)."""

    def __init__(self, llm, prompt: str = "rerank_v1", max_chars: int = 600):
        self.llm, self.prompt, self.max_chars = llm, prompt, max_chars

    @property
    def prompt_hash(self) -> str:
        return load_prompt_file(PROMPTS_DIR / f"{self.prompt}.md")[1]

    def rerank(self, texto: str, items: list[EvidenceItem], coleccion: str) -> list[EvidenceItem]:
        if len(items) <= 1:
            return items
        cands = "\n".join(
            f'<c i="{i}">{" ".join(it.texto.split())[: self.max_chars]}</c>'
            for i, it in enumerate(items)
        )
        template, _ = load_prompt_file(PROMPTS_DIR / f"{self.prompt}.md")
        system, human = template.format_messages(
            mensaje=texto[:3000], candidatos=cands, coleccion=coleccion
        )
        r = self.llm.invoke(
            [SystemMessage(content=system.content), HumanMessage(content=human.content)],
            Ranking,
            stage="rerank",
        )
        if r.parsed is None:
            return items  # sin rerank antes que sin contexto
        pts = {p.i: p.score for p in r.parsed.puntajes if 0 <= p.i < len(items)}
        orden = sorted(range(len(items)), key=lambda i: (-pts.get(i, -1), i))
        out = []
        for i in orden:
            it = items[i].model_copy(deep=True)
            it.score_rerank = float(pts.get(i, -1))
            out.append(it)
        return out


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class AdvancedRetriever:
    def __init__(
        self,
        stores: dict,
        top_n: dict[str, int],
        embeddings: Embeddings,
        names: dict[str, str],
        *,
        rewriter: QueryRewriter | None = None,
        bm25: dict[str, BM25] | None = None,
        weights: dict[str, float] | None = None,
        reranker: LLMReranker | None = None,
        top_in: int = 20,
        rrf_k: int = 60,
        contrastivo: bool = False,
    ):
        unknown = set(top_n) - set(CAMPOS)
        if unknown:
            raise ValueError(f"top_n con colecciones desconocidas: {sorted(unknown)}")
        self.stores, self.top_n, self.embeddings, self.names = stores, top_n, embeddings, names
        self.rewriter, self.bm25, self.reranker = rewriter, bm25 or {}, reranker
        self.weights = weights or {"dense": 1.0, "bm25": 1.0 if bm25 else 0.0}
        self.top_in, self.rrf_k, self.contrastivo = top_in, rrf_k, contrastivo

    def retrieve(self, m: ExtractedMessage) -> RetrievalBundle:
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        queries = self.rewriter.queries(m.texto, m.contexto) if self.rewriter else [m.texto]
        timings["query_rewrite"] = round((time.perf_counter() - t0) * 1000, 1)
        t0 = time.perf_counter()
        vecs = [self.embeddings.embed_query(q) for q in queries]
        timings["embed_query"] = round((time.perf_counter() - t0) * 1000, 1)

        bundle = RetrievalBundle(queries=queries)
        for campo in CAMPOS:
            n, store = self.top_n.get(campo, 0), self.stores.get(campo)
            if n <= 0 or store is None:
                continue
            if campo == "mensajes" and self.contrastivo:
                # V7 candidato · evidencia contrastiva: la mitad de los vecinos con label=estafa y
                # la mitad con label=legitimo, para que el razonador decida por contraste.
                mitad = (n + 1) // 2
                por_label = [
                    self._buscar(
                        campo, store, queries, vecs, mitad, m.texto, timings, {"label": lab}
                    )
                    for lab in ("estafa", "legitimo")
                ]
                mezcla = [it for par in zip(*por_label, strict=False) for it in par]
                resto = [it for lst in por_label for it in lst[len(mezcla) // 2 :]]
                getattr(bundle, campo).extend((mezcla + resto)[:n])
            else:
                getattr(bundle, campo).extend(
                    self._buscar(campo, store, queries, vecs, n, m.texto, timings)
                )
        bundle.timings_ms = timings
        return bundle

    def _buscar(self, campo, store, queries, vecs, n, texto, timings, filtro=None):
        """Denso (+ BM25) por cada query → RRF → [rerank] → top n. `filtro` (metadata exacta)
        restringe las dos ramas."""
        k_cand = max(n, self.top_in)
        name = self.names[campo]
        sufijo = "" if not filtro else "_" + "_".join(str(v) for v in filtro.values())
        t0 = time.perf_counter()
        listas: list[tuple[float, list[EvidenceItem]]] = []
        detalle: list[dict] = []
        for q, vec in zip(queries, vecs, strict=True):
            kw = {"filter": filtro} if filtro else {}
            with CHROMA_LOCK:
                hits = store.similarity_search_by_vector_with_relevance_scores(vec, k=k_cand, **kw)
            densos = [to_item(d, s, name) for d, s in hits]
            listas.append((self.weights.get("dense", 1.0), densos))
            lexicos: list[EvidenceItem] = []
            if (idx := self.bm25.get(campo)) is not None and self.weights.get("bm25", 0) > 0:
                res = idx.search(q, k_cand * (10 if filtro else 1))
                if filtro:
                    res = [(r, sc) for r, sc in res
                           if all(r["metadata"].get(k) == v for k, v in filtro.items())][:k_cand]  # fmt: skip
                lexicos = [bm25_item(r, sc, name) for r, sc in res]
                listas.append((self.weights["bm25"], lexicos))
            if activa():
                detalle.append({
                    "query": q,
                    "denso": [item(x, i) for i, x in enumerate(densos, 1)],
                    "bm25": [item(x, i) for i, x in enumerate(lexicos, 1)] if lexicos else None,
                })  # fmt: skip
        fused = rrf(listas, self.rrf_k)
        fusion = fused
        timings[f"fusion_{campo}{sufijo}"] = round((time.perf_counter() - t0) * 1000, 1)
        reordenado = None
        if self.reranker is not None and len(fused) > n:
            t0 = time.perf_counter()
            fused = self.reranker.rerank(texto, fused[: self.top_in], campo)
            timings[f"rerank_{campo}{sufijo}"] = round((time.perf_counter() - t0) * 1000, 1)
            reordenado = fused
        if activa():
            previo = {x.id: i for i, x in enumerate(fusion, 1)}
            traza(
                "busqueda", coleccion=campo, filtro=filtro, top_n=n, candidatos=k_cand,
                pesos=dict(self.weights), rrf_k=self.rrf_k, por_query=detalle,
                fusion=[item(x, i) for i, x in enumerate(fusion[: self.top_in], 1)],
                rerank=None if reordenado is None else [
                    {**item(x, i), "rank_previo": previo.get(x.id)} for i, x in enumerate(reordenado, 1)
                ],
                final=[item(x, i) for i, x in enumerate(fused[:n], 1)],
                reranker=getattr(getattr(self.reranker, "llm", None), "model", None),
            )  # fmt: skip
        return fused[:n]
