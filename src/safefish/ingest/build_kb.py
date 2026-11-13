"""Construcción de un snapshot inmutable de la KB (M-ING, §3.3).

    python -m safefish.ingest.build_kb --config configs/v1.yaml [--force] [--fake-embeddings]

fuentes → loaders → normalización Unicode → boilerplate → idioma → PII → dedupe exacto y
cercano → chunking → embeddings + Chroma → `data/kb_snapshots/<kb>/`:

- `chroma/`        índice persistente (no se versiona)
- `chunks.jsonl`   chunks con metadata (insumo de BM25 en V5 y de auditorías; no se versiona)
- `manifest.json`  hash, conteos, filtros aplicados y fuentes (sí se versiona)
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
from safefish.ingest.index import build_index
from safefish.ingest.loaders import load_source, load_sources_file
from safefish.ingest.pii import redact_pii
from safefish.observability import UsageTracker, reset_tracker, set_tracker
from safefish.schemas import DocTipo
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


def build_snapshot(
    cfg: PipelineConfig, *, embeddings: Embeddings | None = None, force: bool = False
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
    try:
        docs, report = prepare_documents(cfg)
        if not any(d.metadata["doc_tipo"] == DocTipo.CASO for d in docs):
            raise ValueError(
                f"La KB no tiene documentos tipo `caso`: completá {cfg.ingest.sources} (Track A)"
            )
        chunks = chunk_documents(docs, cfg.chunking)
        for c in chunks:
            c.metadata = scalarize_metadata(c.metadata)
        collection = cfg.kb_snapshot.replace("-", "_")
        with tracker.stage("index"):
            build_index(chunks, emb, out_dir / "chroma", collection)
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
        "ingest": cfg.ingest.model_dump(mode="json"),
        "chunking": cfg.chunking.model_dump(mode="json"),
        "embeddings": {"model": getattr(emb, "model", cfg.embeddings.model)},
        "n_docs": len(docs),
        "n_chunks": len(chunks),
        "counts": _counts(docs, chunks),
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
    print(f"  embeddings: {m['uso']['tokens']['embed']} tokens · USD {m['uso']['costo_usd']:.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
