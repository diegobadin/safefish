"""Arma el pipeline desde el YAML de la versión (§1.5) y lo ejecuta con observabilidad.

    extract → retrieve → reason → [verify R1–R4] → [plan desde Guías] → [verify R5] → output

- V0: sin retrieval. V1: denso en 1 colección. V2a: chunk por caso en 1 colección.
- V2: 4 colecciones con `top_n` por colección + planner desde Guías.
- V3: V2 + verificador. Con planner, el análisis se verifica **antes** de planificar (la
  modalidad final decide las guías) y el plan **después** (R5).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from langchain_core.embeddings import Embeddings

from safefish.concurrency import CHROMA_LOCK
from safefish.config import PipelineConfig
from safefish.extract.capture import MultimodalExtractor
from safefish.generation.planner import GuidePlanner, GuideStore
from safefish.generation.reasoner import LLMClient, LLMReasoner
from safefish.observability import UsageTracker, reset_tracker, set_tracker
from safefish.postprocess.verifier import REGLAS, RuleVerifier
from safefish.schemas import (
    CasoRef,
    ExtractedMessage,
    Extractor,
    Modalidad,
    RawInput,
    Reasoner,
    RetrievalBundle,
    Retriever,
    RunMeta,
    SafeFishOutput,
    Verifier,
)


@dataclass
class PipelineResult:
    output: SafeFishOutput | None
    extracted: ExtractedMessage | None
    bundle: RetrievalBundle | None
    usage: dict = field(default_factory=dict)
    error: str | None = None
    reintentos_parseo: int = 0
    output_pre_verify: SafeFishOutput | None = None  # salida del razonador (+ plan) sin reglas
    verificacion: list[dict] = field(default_factory=list)  # eventos R1–R5
    plan_info: dict | None = None


def _advanced_retriever(cfg: PipelineConfig, stores, emb, names, seed):
    """V4–V6: multi-query, híbrido BM25 y rerank sobre las 4 colecciones (Fase 3)."""
    from safefish.llm import StructuredLLM
    from safefish.retrieval.advanced import AdvancedRetriever, LLMReranker, QueryRewriter, load_bm25

    r, t = cfg.retrieval, cfg.generation.timeout_s
    rewriter = reranker = bm25 = None
    if r.query.strategy == "multi_query":
        qlm = StructuredLLM.from_cfg(r.query.model, 0.0, seed, t)
        rewriter = QueryRewriter(qlm, r.query.prompt, r.query.max_queries)
    if r.hybrid.enabled:
        bm25 = load_bm25(cfg.snapshot_dir)
    if r.rerank.enabled:
        if not r.rerank.model:
            raise ValueError("rerank.enabled necesita rerank.model")
        rlm = StructuredLLM.from_cfg(r.rerank.model, 0.0, seed, t)
        reranker = LLMReranker(rlm, r.rerank.prompt, r.rerank.max_chars)
    weights = r.hybrid.weights if r.hybrid.enabled else {"dense": 1.0, "bm25": 0.0}
    return AdvancedRetriever(
        stores, r.top_n, emb, names,
        rewriter=rewriter, bm25=bm25, weights=weights, reranker=reranker, top_in=r.rerank.top_in,
        contrastivo=r.mensajes_contrastivo,
    )  # fmt: skip


def soporte_dirigido(
    retriever, m: ExtractedMessage, b: RetrievalBundle, out: SafeFishOutput, margen: float
) -> tuple[RetrievalBundle, SafeFishOutput, dict | None]:
    """Respaldo dirigido para R2 (V8). Si la modalidad propuesta no aparece entre los casos
    recuperados, busca el caso más parecido **de esa modalidad** y lo suma a la evidencia si su
    similitud queda a lo sumo `margen` por debajo del mejor caso de toda la colección. Sin
    respaldo así, R2 baja la modalidad a `no_determinada` como siempre."""
    from safefish.retrieval.multi import to_item

    mod = out.modalidad.value
    if out.modalidad is Modalidad.NO_DETERMINADA or any(
        str(c.metadata.get("modalidad")) == mod for c in b.casos
    ):
        return b, out, None
    stores = getattr(retriever, "stores", None) or {}
    store, emb = stores.get("casos"), getattr(retriever, "embeddings", None)
    if store is None or emb is None:
        return b, out, None
    vec = emb.embed_query(m.texto)
    with CHROMA_LOCK:
        mejor = store.similarity_search_by_vector_with_relevance_scores(vec, k=1)
        propio = store.similarity_search_by_vector_with_relevance_scores(
            vec, k=1, filter={"modalidad": mod}
        )
    if not mejor or not propio:
        return b, out, None
    (doc, sim), sim_mejor = propio[0], mejor[0][1]
    detalle = {"modalidad": mod, "case_id": doc.metadata.get("doc_id"),
               "similitud": round(sim, 4), "mejor_similitud": round(sim_mejor, 4)}  # fmt: skip
    if sim_mejor - sim > margen:
        return b, out, {**detalle, "aceptado": False}
    name = getattr(retriever, "names", {}).get("casos", "casos")
    item = to_item(doc, 1.0 - sim, name)
    item.metadata["via"] = "soporte_dirigido"
    ref = CasoRef(
        case_id=str(doc.metadata.get("doc_id", item.id)),
        fuente_url=doc.metadata.get("fuente_url"),
        por_que="Caso documentado de la misma modalidad, encontrado con una búsqueda dirigida.",
    )
    b = b.model_copy(update={"casos": [*b.casos, item]})
    out = out.model_copy(update={"casos_similares": [*out.casos_similares, ref]})
    return b, out, {**detalle, "aceptado": True}


def _merge_guias(b: RetrievalBundle, guias) -> RetrievalBundle:
    ids = {g.id for g in b.guias}
    return b.model_copy(update={"guias": [*b.guias, *(g for g in guias if g.id not in ids)]})


class SafeFishPipeline:
    def __init__(
        self,
        cfg: PipelineConfig,
        extractor: Extractor,
        retriever: Retriever,
        reasoner: Reasoner,
        verifier: Verifier | None = None,
        kb_manifest: dict | None = None,
        planner: GuidePlanner | None = None,
    ):
        self.cfg, self.extractor, self.retriever = cfg, extractor, retriever
        self.reasoner, self.verifier, self.kb_manifest = reasoner, verifier, kb_manifest or {}
        self.planner = planner

    @classmethod
    def from_config(
        cls,
        cfg: PipelineConfig,
        *,
        llm: LLMClient | None = None,
        embeddings: Embeddings | None = None,
        planner_llm: LLMClient | None = None,
        vision_llm: LLMClient | None = None,
    ) -> SafeFishPipeline:
        from safefish.embeddings import make_embeddings
        from safefish.extract.ocr import TesseractOCR
        from safefish.extract.vision import VisionTranscriber
        from safefish.ingest.index import (
            check_embeddings_model,
            open_collections,
            open_index,
            read_manifest,
        )
        from safefish.llm import StructuredLLM
        from safefish.retrieval.dense import DenseRetriever, NullRetriever
        from safefish.retrieval.multi import MultiCollectionRetriever

        r, g = cfg.retrieval, cfg.generation
        manifest: dict = {}
        guias_dense = None
        if not r.enabled:
            retriever: Retriever = NullRetriever()
        else:
            advanced = (
                r.query.strategy != "raw"
                or r.hybrid.enabled
                or r.rerank.enabled
                or r.mensajes_contrastivo
            )
            if advanced and cfg.index.layout != "per_tipo":
                raise ValueError(
                    "multi-query, híbrido y rerank (V4–V6) van sobre las 4 colecciones"
                )
            if r.filter_canal != "off":
                raise NotImplementedError("filtro por canal: barrido de Fase 3")
            emb = embeddings or make_embeddings(cfg.embeddings)
            snap = cfg.snapshot_dir
            manifest = read_manifest(snap)
            model = getattr(emb, "model", cfg.embeddings.model)
            check_embeddings_model(manifest, model, snap.name)
            layout = manifest.get("index_layout", "single")
            if layout != cfg.index.layout:
                raise ValueError(
                    f"El snapshot {snap.name} tiene layout {layout} y el config pide "
                    f"{cfg.index.layout}: reconstruí la KB o cambiá el config"
                )
            if layout == "single":
                store = open_index(snap, emb, model)
                retriever = DenseRetriever(store, r.top_k, manifest["collection"])

                def guias_dense(q: str, k: int) -> list[str]:
                    with CHROMA_LOCK:
                        hits = store.similarity_search(q, k=k, filter={"doc_tipo": "guia"})
                    return [str(d.metadata.get("doc_id")) for d in hits]
            else:
                names = manifest["collections"]
                stores = open_collections(snap, emb, names)
                if advanced:
                    retriever = _advanced_retriever(cfg, stores, emb, names, g.seed)
                else:
                    retriever = MultiCollectionRetriever(stores, r.top_n, emb, names)
                if (gstore := stores.get("guias")) is not None:

                    def guias_dense(q: str, k: int) -> list[str]:
                        with CHROMA_LOCK:
                            hits = gstore.similarity_search(q, k=k)
                        return [str(d.metadata.get("doc_id")) for d in hits]

        planner = None
        if g.planner:
            if not r.enabled:
                raise ValueError("El planner desde Guías necesita una KB (retrieval.enabled)")
            plm = None
            if g.planner_mode == "llm":
                plm = planner_llm or llm or StructuredLLM.from_cfg(g.planner_model, 0.0, g.seed, g.timeout_s)  # fmt: skip
            planner = GuidePlanner(
                GuideStore.from_snapshot(cfg.snapshot_dir, guias_dense),
                plm,
                mode=g.planner_mode,
                prompt=g.planner_prompt,
                top_guias=g.planner_top_guias,
                max_pasos=g.planner_max_pasos,
            )

        p = cfg.postprocess
        verifier = (
            RuleVerifier(p.rules or REGLAS, p.cita_threshold, p.bbox_threshold)
            if p.verifier
            else None
        )

        x = cfg.extraction
        vlm = vision_llm or StructuredLLM.from_cfg(x.vision_model, 0.0, g.seed, x.timeout_s)
        extractor = MultimodalExtractor(
            vision=VisionTranscriber(vlm, x.vision_prompt),
            ocr=TesseractOCR(x.ocr_lang, x.ocr_min_conf, x.preprocess)
            if x.ocr == "tesseract"
            else None,  # fmt: skip
        )

        llm = llm or StructuredLLM.from_cfg(g.reasoner, g.temperature, g.seed, g.timeout_s)
        reasoner = LLMReasoner(llm, prompt=g.prompt, max_retries=g.max_retries)
        return cls(cfg, extractor, retriever, reasoner, verifier, manifest, planner)

    def retrieve_only(self, raw: RawInput) -> PipelineResult:
        """Barridos de solo retrieval (§2.0): extract + retrieve, sin generación."""
        tracker = UsageTracker()
        token = set_tracker(tracker)
        m = b = None
        error = None
        try:
            with tracker.stage("extract"):
                m = self.extractor.extract(raw)
            with tracker.stage("retrieve"):
                b = self.retriever.retrieve(m)
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
        finally:
            reset_tracker(token)
        return PipelineResult(None, m, b, tracker.summary(), error)

    def run(self, raw: RawInput) -> PipelineResult:
        tracker = UsageTracker()
        token = set_tracker(tracker)
        m = b = out = pre = None
        error, retries = None, 0
        eventos: list[dict] = []
        plan_info = None
        v = self.verifier
        staged = v is not None and hasattr(v, "check_analysis") and hasattr(v, "check_plan")
        try:
            with tracker.stage("extract"):
                m = self.extractor.extract(raw)
            with tracker.stage("retrieve"):
                b = self.retriever.retrieve(m)
            meta = RunMeta(
                run_id=uuid.uuid4().hex[:12],
                config_hash=self.cfg.config_hash(),
                kb_snapshot=self.cfg.kb_snapshot if self.cfg.retrieval.enabled else None,
            )
            with tracker.stage("reason"):
                out = self.reasoner.reason(m, b, raw.situacion, meta)
            retries = out.meta.reintentos_parseo
            pre = out

            margen = self.cfg.postprocess.soporte_dirigido_margen
            if v is not None and margen is not None:
                with tracker.stage("verify"):
                    b, out, det = soporte_dirigido(self.retriever, m, b, out, margen)
                if det is not None:
                    eventos.append({"regla": "R2", "accion": "soporte_dirigido", "detalle": det})
            if staged:
                with tracker.stage("verify"):
                    out, ev = v.check_analysis(m, b, out)
                eventos += [e.to_dict() for e in ev]
            if self.planner is not None:
                with tracker.stage("plan"):
                    pr = self.planner.plan(out, raw.situacion)
                b = _merge_guias(b, pr.guias)
                plan_info = pr.info
                out = out.model_copy(update={"plan": pr.plan})
                pre = pre.model_copy(update={"plan": pr.plan})
                if self.planner.llm is not None:
                    modelos = {**out.meta.modelos, "planner": self.planner.llm.model}
                    out = out.model_copy(update={"meta": out.meta.model_copy(update={"modelos": modelos})})  # fmt: skip
            if staged:
                with tracker.stage("verify"):
                    out, ev = v.check_plan(b, out, m)
                eventos += [e.to_dict() for e in ev]
            elif v is not None:
                with tracker.stage("verify"):
                    out = v.verify(m, b, out)
        except NotImplementedError:
            raise
        except Exception as e:  # el arnés registra el caso como fallido y sigue
            error = f"{type(e).__name__}: {e}"
            retries = getattr(e, "reintentos", retries)
            out = None
        finally:
            reset_tracker(token)

        usage = tracker.summary()
        if out is not None:
            out = out.model_copy(
                update={
                    "meta": out.meta.model_copy(
                        update={
                            "tokens": usage["tokens"],
                            "costo_usd": usage["costo_usd"],
                            "latencias_ms": usage["latencias_ms"],
                            "cache_hit": tracker.all_cache_hits(),
                        }
                    )
                }
            )
        return PipelineResult(
            out, m, b, usage, error, retries,
            output_pre_verify=pre if v is not None else None,
            verificacion=eventos,
            plan_info=plan_info,
        )  # fmt: skip
