"""Arma el pipeline desde el YAML de la versión (§1.5) y lo ejecuta con observabilidad.

extract → retrieve → reason → (verify, V3+) → SafeFishOutput
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from langchain_core.embeddings import Embeddings

from safefish.config import PipelineConfig
from safefish.extract.text import TextExtractor
from safefish.generation.reasoner import LLMClient, LLMReasoner
from safefish.observability import UsageTracker, reset_tracker, set_tracker
from safefish.schemas import (
    ExtractedMessage,
    Extractor,
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


class SafeFishPipeline:
    def __init__(
        self,
        cfg: PipelineConfig,
        extractor: Extractor,
        retriever: Retriever,
        reasoner: Reasoner,
        verifier: Verifier | None = None,
        kb_manifest: dict | None = None,
    ):
        self.cfg, self.extractor, self.retriever = cfg, extractor, retriever
        self.reasoner, self.verifier, self.kb_manifest = reasoner, verifier, kb_manifest or {}

    @classmethod
    def from_config(
        cls,
        cfg: PipelineConfig,
        *,
        llm: LLMClient | None = None,
        embeddings: Embeddings | None = None,
    ) -> SafeFishPipeline:
        from safefish.embeddings import make_embeddings
        from safefish.ingest.index import open_index, read_manifest
        from safefish.llm import StructuredLLM
        from safefish.retrieval.dense import DenseRetriever, NullRetriever

        r = cfg.retrieval
        manifest: dict = {}
        if not r.enabled:
            retriever: Retriever = NullRetriever()
        else:
            if r.query.strategy != "raw" or r.hybrid.enabled or r.rerank.enabled:
                raise NotImplementedError("multi-query, híbrido y rerank: Fase 3 (V4–V6)")
            if r.filter_canal != "off":
                raise NotImplementedError("filtro por canal: barrido de Fase 3")
            emb = embeddings or make_embeddings(cfg.embeddings)
            snap = cfg.snapshot_dir
            manifest = read_manifest(snap)
            store = open_index(snap, emb, getattr(emb, "model", cfg.embeddings.model))
            retriever = DenseRetriever(store, r.top_k, manifest["collection"])

        g = cfg.generation
        if g.planner:
            raise NotImplementedError("planner desde Guías: V2 (Track B)")
        if cfg.postprocess.verifier:
            raise NotImplementedError("verificador R1–R5: V3 (Track B)")
        llm = llm or StructuredLLM.from_cfg(g.reasoner, g.temperature, g.seed, g.timeout_s)
        reasoner = LLMReasoner(llm, prompt=g.prompt, max_retries=g.max_retries)
        return cls(cfg, TextExtractor(), retriever, reasoner, None, manifest)

    def run(self, raw: RawInput) -> PipelineResult:
        tracker = UsageTracker()
        token = set_tracker(tracker)
        m = b = out = None
        error, retries = None, 0
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
            if self.verifier is not None:
                with tracker.stage("verify"):
                    out = self.verifier.verify(m, b, out)
            retries = out.meta.reintentos_parseo
        except NotImplementedError:
            raise
        except Exception as e:  # el arnés registra el caso como fallido y sigue
            error = f"{type(e).__name__}: {e}"
            retries = getattr(e, "reintentos", 0)
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
        return PipelineResult(out, m, b, usage, error, retries)
