"""Fase 3 (V4–V6): BM25, RRF, multi-query y rerank, sin red."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from langchain_core.documents import Document

from safefish.llm import LLMResult
from safefish.retrieval.advanced import (
    BM25,
    AdvancedRetriever,
    LLMReranker,
    Puntaje,
    QueryFacets,
    QueryRewriter,
    Ranking,
    load_bm25,
    rrf,
    tokenize,
)
from safefish.schemas import EvidenceItem, ExtractedMessage


def _rows(*textos):
    return [{"id": f"c{i}", "texto": t, "metadata": {"chunk_id": f"c{i}", "doc_tipo": "caso"}}
            for i, t in enumerate(textos)]  # fmt: skip


def _it(i, dense=None, bm25=None):
    return EvidenceItem(id=i, coleccion="c", texto=i, score_dense=dense, score_bm25=bm25)


def test_tokenize_sin_tildes_ni_stopwords():
    assert tokenize("Pasame el CÓDIGO de verificación ya") == ["pasame", "codigo", "verificacion"]


def test_bm25_prioriza_termino_exacto():
    idx = BM25(_rows("transferí al CBU del abogado", "tu paquete está retenido en aduana",
                     "el banco nunca pide tu clave"))  # fmt: skip
    hits = idx.search("me piden un CBU", 2)
    assert hits[0][0]["id"] == "c0" and len(hits) == 1  # solo matchea el que tiene "cbu"
    assert idx.search("palabra inexistente", 3) == []


def test_load_bm25_por_coleccion(tmp_path):
    rows = _rows("a b caso") + [{"id": "m0", "texto": "hola mensaje",
                                 "metadata": {"chunk_id": "m0", "doc_tipo": "mensaje"}}]  # fmt: skip
    (tmp_path / "chunks.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    idx = load_bm25(tmp_path)
    assert set(idx) == {"casos", "mensajes"} and idx["mensajes"].search("mensaje", 1)[0][0]["id"] == "m0"  # fmt: skip


def test_rrf_suma_listas_y_conserva_scores():
    fused = rrf([(1.0, [_it("a", dense=0.9), _it("b", dense=0.8)]),
                 (1.0, [_it("b", bm25=7.0), _it("c", bm25=2.0)])])  # fmt: skip
    assert [x.id for x in fused] == ["b", "a", "c"]  # b aparece en las dos listas
    b = fused[0]
    assert b.score_dense == 0.8 and b.score_bm25 == 7.0 and b.metadata["score_rrf"] > 0
    assert [x.id for x in rrf([(0.0, [_it("a")]), (1.0, [_it("c")])])] == ["c"]  # peso 0 = fuera


@dataclass
class FakeLLM:
    parsed: object
    calls: list = field(default_factory=list)

    def invoke(self, messages, schema, stage="reason"):
        self.calls.append((stage, messages[-1].content))
        return LLMResult(parsed=self.parsed, raw_text="")


def test_query_rewriter_arma_queries_sin_repetidos():
    f = QueryFacets(resumen="Un supuesto correo pide pagar un arancel", pedido="pagar con tarjeta",
                    entidad="Correo Argentino", frases_clave=["paquete retenido", "aduana"])  # fmt: skip
    qr = QueryRewriter(FakeLLM(f))
    qs = qr.queries("Su paquete está retenido")
    assert qs == ["Su paquete está retenido", "Un supuesto correo pide pagar un arancel",
                  "Correo Argentino: pagar con tarjeta", "paquete retenido · aduana"]  # fmt: skip
    assert QueryRewriter(FakeLLM(None)).queries("hola") == ["hola"]  # si falla el LLM, texto crudo


def test_reranker_ordena_por_puntaje_y_es_estable():
    rr = LLMReranker(FakeLLM(Ranking(puntajes=[Puntaje(i=0, score=1), Puntaje(i=2, score=3)])))
    out = rr.rerank("msg", [_it("a"), _it("b"), _it("c")], "casos")
    assert [x.id for x in out] == ["c", "a", "b"] and out[0].score_rerank == 3.0
    assert out[2].score_rerank == -1.0  # sin puntaje: al final
    falla = LLMReranker(FakeLLM(None)).rerank("msg", [_it("a"), _it("b")], "casos")
    assert [x.id for x in falla] == ["a", "b"]


class FakeStore:
    def __init__(self, docs):
        self.docs = docs

    def similarity_search_by_vector_with_relevance_scores(self, vec, k):
        return [(Document(page_content=d, metadata={"chunk_id": d, "doc_tipo": "caso"}), 0.1 * i)
                for i, d in enumerate(self.docs[:k])]  # fmt: skip


class FakeEmb:
    def embed_query(self, q):
        return [0.0]


def test_advanced_retriever_hibrido_y_rerank():
    bm = BM25([{"id": "x", "texto": "codigo de verificacion", "metadata": {"chunk_id": "z"}}])
    rerank = FakeLLM(Ranking(puntajes=[Puntaje(i=1, score=3)]))  # fusión: a, z, b
    ret = AdvancedRetriever(
        {"casos": FakeStore(["a", "b"])}, {"casos": 2}, FakeEmb(), {"casos": "kb_casos"},
        bm25={"casos": bm}, weights={"dense": 1.0, "bm25": 1.0},
        reranker=LLMReranker(rerank), top_in=5,
    )  # fmt: skip
    b = ret.retrieve(ExtractedMessage(texto="pasame el codigo"))
    assert b.queries == ["pasame el codigo"]
    assert [x.id for x in b.casos] == ["z", "a"]  # el BM25 entró y el rerank lo subió
    assert rerank.calls and rerank.calls[0][0] == "rerank"
    assert "rerank_casos" in b.timings_ms


class FakeLabelStore:
    def __init__(self, docs):  # docs: [(texto, label)]
        self.docs = docs

    def similarity_search_by_vector_with_relevance_scores(self, vec, k, filter=None):
        docs = [(t, lab) for t, lab in self.docs if not filter or filter.get("label") == lab]
        return [(Document(page_content=t, metadata={"chunk_id": t, "doc_tipo": "mensaje", "label": lab}), 0.1 * i)
                for i, (t, lab) in enumerate(docs[:k])]  # fmt: skip


def test_mensajes_contrastivo_trae_ambas_etiquetas():
    store = FakeLabelStore(
        [("e1", "estafa"), ("e2", "estafa"), ("e3", "estafa"), ("l1", "legitimo")]
    )
    ret = AdvancedRetriever({"mensajes": store}, {"mensajes": 4}, FakeEmb(),
                            {"mensajes": "kb_mensajes"}, contrastivo=True)  # fmt: skip
    ids = [x.id for x in ret.retrieve(ExtractedMessage(texto="x")).mensajes]
    assert ids[:2] == ["e1", "l1"] and set(ids) == {
        "e1",
        "e2",
        "l1",
    }  # intercalados; falta 1 legítimo
