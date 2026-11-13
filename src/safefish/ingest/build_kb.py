"""Construcción de un snapshot inmutable de la KB (M-ING, §3.3).

    python -m safefish.ingest.build_kb --config configs/v1.yaml [--force] [--fake-embeddings]

fuentes → loaders → normalización Unicode → boilerplate → idioma → PII → dedupe exacto y
cercano → [V2+: normalización a CaseDoc/MessageDoc/SignalDoc/GuideDoc] → chunking →
embeddings + Chroma → `data/kb_snapshots/<kb>/`:

- `chroma/`                      índice persistente: 1 colección (single) o 4 (per_tipo)
- `chunks.jsonl`                 chunks con metadata (planner, BM25 en V5, auditorías)
- `normalizacion_revision.jsonl` V2+: muestra del 10 % para revisión manual (§2.3)
- `manifest.json`                hash, conteos, cobertura, filtros y fuentes (se versiona)

Ninguno de los archivos salvo `manifest.json` se versiona.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from safefish.config import PipelineConfig, load_config, resolve_path
from safefish.embeddings import make_embeddings
from safefish.ingest.chunk import chunk_documents, scalarize_metadata
from safefish.ingest.clean import (
    content_hash,
    dedupe_exact,
    dedupe_near,
    filter_language,
    strip_boilerplate,
)
from safefish.ingest.coverage import audit_injection, coverage
from safefish.ingest.index import build_indexes
from safefish.ingest.loaders import load_source, load_sources_file
from safefish.ingest.normalize import (
    LLMNormalizer,
    NormalizeReport,
    StructuredClient,
    kbdoc_to_document,
    normalize_documents,
)
from safefish.ingest.pii import redact_pii
from safefish.observability import UsageTracker, reset_tracker, set_tracker
from safefish.schemas import DocTipo, KBDoc
from safefish.textutils import normalize_unicode


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def kb_hash(chunks: list[Document]) -> str:
    h = hashlib.sha256()
    for d in sorted(chunks, key=lambda d: d.id):
        h.update(f"{d.id}\n{d.page_content}\n".encode())
    return h.hexdigest()[:16]


def _pii_enabled(doc: Document, spec_pii: bool | None, cfg: PipelineConfig) -> bool:
    if spec_pii is not None:
        return spec_pii
    return cfg.ingest.pii_por_tipo.get(DocTipo(doc.metadata["doc_tipo"]), True)


def prepare_documents(cfg: PipelineConfig) -> tuple[list[Document], dict]:
    """Carga y limpia todas las fuentes. Devuelve (documentos, reporte para el manifest)."""
    specs, base_dir = load_sources_file(resolve_path(cfg.ingest.sources))
    report: dict = {"fuentes": [], "filtros": Counter(), "pii": Counter()}
    docs: list[Document] = []
    for spec in specs:
        if not spec.enabled:
            continue
        loaded = load_source(spec, base_dir)
        cleaned = []
        for d in loaded:
            d.page_content = normalize_unicode(d.page_content)
            if _pii_enabled(d, spec.pii, cfg):
                r = redact_pii(d.page_content)
                d.page_content = r.text
                report["pii"].update(r.counts)
            cleaned.append(d)
        report["fuentes"].append(
            {
                "id": spec.id,
                "tipo": spec.tipo.value,
                "loader": spec.loader,
                "fuente_tipo": spec.fuente_tipo,
                "docs_cargados": len(loaded),
            }
        )
        docs.extend(cleaned)

    docs = strip_boilerplate(docs)
    empty = [d for d in docs if not d.page_content.strip()]
    docs = [d for d in docs if d.page_content.strip()]
    report["filtros"]["vacios"] += len(empty)

    docs, dropped_lang = filter_language(
        docs, cfg.ingest.languages, cfg.ingest.min_chars_lang_detect
    )
    report["filtros"]["idioma"] += len(dropped_lang)

    for d in docs:  # IDs determinísticos: hash del contenido normalizado (§5.8)
        d.metadata.setdefault(
            "doc_id", f"{d.metadata['source_id']}-{content_hash(d.page_content)[:10]}"
        )

    if cfg.ingest.dedupe.exact:
        docs, dropped = dedupe_exact(docs)
        report["filtros"]["dedupe_exacto"] += len(dropped)
    if cfg.ingest.dedupe.near_threshold is not None:
        vectorize = None
        if cfg.ingest.dedupe.near_method == "embeddings":
            emb = make_embeddings(cfg.embeddings)

            def vectorize(texts):
                import numpy as np

                return np.array(emb.embed_documents([t[:8000] for t in texts]))

        kwargs = {"vectorize": vectorize} if vectorize else {}
        docs, dropped = dedupe_near(docs, cfg.ingest.dedupe.near_threshold, **kwargs)
        report["filtros"]["dedupe_cercano"] += len(dropped)
        report["duplicados_cercanos"] = [
            {"doc_id": d.metadata["doc_id"], "de": d.metadata.get("duplicado_de"),
             "sim": d.metadata.get("duplicado_sim")}
            for d in dropped
        ]  # fmt: skip

    ids = Counter(d.metadata["doc_id"] for d in docs)
    if dup := [i for i, c in ids.items() if c > 1]:
        raise ValueError(f"doc_id repetidos entre fuentes: {dup[:5]}")
    return docs, report


def _counts(docs: list[Document], chunks: list[Document]) -> dict:
    def tipo(d):
        return d.metadata["doc_tipo"]

    casos = [c for c in chunks if tipo(c) == DocTipo.CASO]
    return {
        "docs_por_tipo": dict(Counter(tipo(d) for d in docs)),
        "chunks_por_tipo": dict(Counter(tipo(c) for c in chunks)),
        "docs_por_modalidad": dict(Counter(str(d.metadata.get("modalidad")) for d in docs if tipo(d) == DocTipo.CASO)),
        "chunks_casos_por_modalidad": dict(Counter(str(c.metadata.get("modalidad")) for c in casos)),
        "chunks_casos_por_modalidad_entidad": dict(
            Counter(f"{c.metadata.get('modalidad')}|{c.metadata.get('entidad_suplantada', c.metadata.get('entidad'))}" for c in casos)
        ),
        "mensajes_por_label": dict(Counter(str(d.metadata.get("label")) for d in docs if tipo(d) == DocTipo.MENSAJE)),
        "docs_por_fuente_tipo": dict(Counter(d.metadata.get("fuente_tipo") for d in docs)),
    }  # fmt: skip


def normalize_stage(
    cfg: PipelineConfig, docs: list[Document], llm: StructuredClient | None = None
) -> tuple[list[KBDoc], NormalizeReport, str | None]:
    """V2+: documentos limpios → documentos estructurados. El LLM se crea solo si hace falta
    (con `mode: auto` y fuentes ya estructuradas no se llama nunca)."""
    ncfg = cfg.ingest.normalize
    normalizer = None
    if ncfg.mode != "template":
        if llm is None:
            from safefish.llm import StructuredLLM

            llm = StructuredLLM.from_cfg(ncfg.model, 0.0, 7, ncfg.timeout_s)
        normalizer = LLMNormalizer(llm, ncfg)
    kbdocs, report = normalize_documents(docs, ncfg, normalizer)
    versions = sorted({d.normalizer_version for d in kbdocs})
    return kbdocs, report, ",".join(versions) or None


def review_sample(
    kbdocs: list[KBDoc], report: NormalizeReport, raw: dict[str, str], frac: float, seed: int
) -> list[dict]:
    """Muestra para revisión manual (§2.3). Prioriza lo normalizado por LLM: lo determinístico
    (plantillas, taxonomía, CSV etiquetado) no necesita revisión de contenido."""
    llm_ids = set(report.llm_doc_ids)
    pool = [d for d in kbdocs if d.doc_id in llm_ids] or [
        d for d in kbdocs if d.normalizer_version != "template"
    ]
    if not pool:
        return []
    n = min(len(pool), max(1, round(frac * len(pool))))
    sample = random.Random(seed).sample(sorted(pool, key=lambda d: d.doc_id), n)

    def fuente(doc_id: str) -> str:  # los casos múltiples de una fuente llevan sufijo -N
        return raw.get(doc_id) or raw.get(doc_id.rsplit("-", 1)[0], "")

    return [
        {
            "doc_id": d.doc_id,
            "tipo": type(d).__name__,
            "normalizer_version": d.normalizer_version,
            "fuente_extracto": fuente(d.doc_id)[:1500],
            "normalizado": d.model_dump(mode="json"),
            "revision": {"ok": None, "revisor": None, "notas": ""},
        }
        for d in sorted(sample, key=lambda d: d.doc_id)
    ]


def build_snapshot(
    cfg: PipelineConfig,
    *,
    embeddings: Embeddings | None = None,
    force: bool = False,
    normalizer_llm: StructuredClient | None = None,
) -> dict:
    if not cfg.kb_snapshot:
        raise ValueError(f"El config {cfg.version} no define kb_snapshot (¿es V0?)")
    out_dir = cfg.snapshot_dir
    if (out_dir / "manifest.json").exists() and not force:
        raise FileExistsError(
            f"El snapshot {cfg.kb_snapshot} ya existe y es inmutable. Usá otro nombre o --force."
        )
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    emb = embeddings or make_embeddings(cfg.embeddings)
    tracker = UsageTracker()
    token = set_tracker(tracker)
    norm_report: NormalizeReport | None = None
    normalizer_version = None
    revision: list[dict] = []
    try:
        docs, report = prepare_documents(cfg)
        if not any(d.metadata["doc_tipo"] == DocTipo.CASO for d in docs):
            raise ValueError(
                f"La KB no tiene documentos tipo `caso`: completá {cfg.ingest.sources} (Track A)"
            )
        if cfg.chunking.strategy == "per_case":
            raw = {d.metadata["doc_id"]: d.page_content for d in docs}
            with tracker.stage("normalize"):
                kbdocs, norm_report, normalizer_version = normalize_stage(cfg, docs, normalizer_llm)
            docs = [kbdoc_to_document(d) for d in kbdocs]
            ncfg = cfg.ingest.normalize
            revision = review_sample(
                kbdocs, norm_report, raw, ncfg.review_fraction, ncfg.review_seed
            )
        chunks = chunk_documents(docs, cfg.chunking)
        for c in chunks:
            c.metadata = scalarize_metadata(c.metadata)
        collection = cfg.kb_snapshot.replace("-", "_")
        with tracker.stage("index"):
            collections = build_indexes(
                chunks, emb, out_dir / "chroma", collection, cfg.index.layout
            )
    finally:
        reset_tracker(token)

    with open(out_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(
                json.dumps(
                    {"id": c.id, "texto": c.page_content, "metadata": c.metadata},
                    ensure_ascii=False,
                )
                + "\n"
            )
    if revision:
        with open(out_dir / "normalizacion_revision.jsonl", "w", encoding="utf-8") as f:
            for row in revision:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    campaigns = sorted(
        {str(d.metadata["campaign_id"]) for d in docs if d.metadata.get("campaign_id")}
    )
    manifest = {
        "snapshot": cfg.kb_snapshot,
        "kb_hash": kb_hash(chunks),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "config_version": cfg.version,
        "config_hash": cfg.config_hash(),
        "collection": collection,
        "index_layout": cfg.index.layout,
        "collections": collections,
        "ingest": cfg.ingest.model_dump(mode="json"),
        "chunking": cfg.chunking.model_dump(mode="json"),
        "embeddings": {"model": getattr(emb, "model", cfg.embeddings.model)},
        "normalizer_version": normalizer_version,
        "n_docs": len(docs),
        "n_chunks": len(chunks),
        "counts": _counts(docs, chunks),
        "cobertura": coverage([c.metadata for c in chunks], cfg.ingest.min_casos_por_modalidad),
        "normalizacion": norm_report.to_dict() if norm_report else None,
        "revision_muestra": [r["doc_id"] for r in revision],
        # Los mensajes pueden contener inyecciones a propósito (ejemplos de ataque) y la
        # taxonomía es curada por el equipo: se auditan los casos, guías y señales de fuentes
        # externas, que son lo que el modelo toma como referencia.
        "auditoria_inyeccion": audit_injection(
            (c.metadata["doc_id"], c.page_content)
            for c in chunks
            if c.metadata["doc_tipo"] != DocTipo.MENSAJE
            and c.metadata.get("fuente_tipo") != "taxonomia"
        ),
        "fuentes": report["fuentes"],
        "filtros_descartados": dict(report["filtros"]),
        "duplicados_cercanos": report.get("duplicados_cercanos", []),
        "pii_redactada": dict(report["pii"]),
        "campaign_ids": campaigns,
        "uso": tracker.summary(),
    }  # fmt: skip
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--force", action="store_true", help="sobrescribe un snapshot existente")
    ap.add_argument("--fake-embeddings", action="store_true", help="embeddings por hash (sin API)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    emb = make_embeddings(cfg.embeddings, fake=True) if args.fake_embeddings else None
    m = build_snapshot(cfg, embeddings=emb, force=args.force)
    print(f"Snapshot {m['snapshot']} ({m['kb_hash']}): {m['n_docs']} docs → {m['n_chunks']} chunks")
    print(f"  por tipo: {m['counts']['chunks_por_tipo']}")
    print(f"  descartados: {m['filtros_descartados']} · PII: {m['pii_redactada']}")
    if n := m.get("normalizacion"):
        print(f"  normalización: {n['metodo']} · descartados {n['descartados']} · errores {len(n['errores'])}")  # fmt: skip
        print(f"  revisión manual: {len(m['revision_muestra'])} docs en normalizacion_revision.jsonl")  # fmt: skip
    cob = m["cobertura"]
    if bajo := cob["modalidades_bajo_umbral"]:
        print(f"  ✗ cobertura: {len(bajo)} modalidad(es) con < {cob['min_casos_por_modalidad']} casos: {', '.join(bajo)}")  # fmt: skip
    if m["auditoria_inyeccion"]:
        print(f"  ⚠ auditoría de inyección: {len(m['auditoria_inyeccion'])} chunk(s) para revisar")  # fmt: skip
    print(f"  embeddings: {m['uso']['tokens']['embed']} tokens · USD {m['uso']['costo_usd']:.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
