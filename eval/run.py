"""Arnés de evaluación (§4.2).

    python -m eval.run --config configs/v1.yaml --split dev [--golden-dir golden] [--limit N]
                       [--judge-plan]   # cobertura del plan con juez LLM (V2+, cuesta)
                       [--retrieval-only]  # barridos: sin generación, solo embeddings

Escribe `results/<run_id>/`: predictions.jsonl, retrieval.jsonl, metrics.json, costs.json y
config.yaml. `metrics.json` registra git_sha, config_hash, kb_snapshot/kb_hash,
golden_version, modelos, hash del prompt y fecha.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv

from eval.golden import golden_path, golden_version, load_golden
from eval.metrics_extraction import compute_extraction_metrics
from eval.metrics_retrieval import compute_retrieval_metrics
from eval.metrics_task import by_slice, compute_task_metrics
from safefish.config import REPO_ROOT, PipelineConfig, load_config
from safefish.pipeline import PipelineResult, SafeFishPipeline
from safefish.schemas import GoldenCase

KEY_METRICS = [
    "f1_binario", "recall_estafas", "fpr_hard_negative", "veredicto_aceptable",
    "acc_modalidad", "macro_f1_modalidad", "f1_senales", "grounding_citas",
    "grounding_citas_pre_r1", "soporte_modalidad", "contactos_no_respaldados",
    "cobertura_plan", "negative_rejection", "asr_inyeccion",
    "json_valido_sin_reintento", "n_errores",
]  # fmt: skip


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _item(it) -> dict:
    md = it.metadata
    keep = ("doc_id", "doc_tipo", "modalidad", "entidad", "entidad_suplantada", "label",
            "signal_id", "situacion", "campaign_id", "fuente_url", "canal")  # fmt: skip
    return {
        "id": it.id,
        "score_dense": it.score_dense,
        "score_bm25": it.score_bm25,
        "score_rerank": it.score_rerank,
        "metadata": {k: md[k] for k in keep if k in md},
    }


def prediction_row(case: GoldenCase, res: PipelineResult) -> dict:
    b = res.bundle
    return {
        "id": case.id,
        "split": case.split,
        "slices": case.slices,
        "canal": case.canal.value if case.canal else None,
        "origen": case.origen,
        "texto": res.extracted.texto if res.extracted else case.texto,
        "texto_gold": case.texto,
        "citas_bbox_gold": [c.model_dump(mode="json") for c in case.citas_bbox],
        "label": case.label,
        "veredicto_esperado": [v.value for v in case.veredicto_esperado],
        "modalidad_gold": case.modalidad.value if case.modalidad else None,
        "senales_gold": [s.value for s in case.senales_esperadas],
        "pred": res.output.model_dump(mode="json", exclude={"meta"}) if res.output else None,
        "pred_pre_verify": res.output_pre_verify.model_dump(mode="json", exclude={"meta"})
        if res.output_pre_verify else None,
        "verificacion": res.verificacion,
        "plan_info": res.plan_info,
        "error": res.error,
        "reintentos": res.reintentos_parseo,
        "contexto_modalidades_casos": sorted(
            {c.metadata.get("modalidad") for c in b.casos if c.metadata.get("modalidad")}
        ) if b else [],
        "contexto_guias": [g.texto for g in b.guias] if b else [],
        "contexto_guias_por_id": {str(g.metadata.get("doc_id", g.id)): g.texto for g in b.guias} if b else {},
        "usage": res.usage,
    }  # fmt: skip


def retrieval_row(case: GoldenCase, res: PipelineResult) -> dict:
    b = res.bundle
    if b is None:
        return {"id": case.id, "queries": []}
    return {
        "id": case.id,
        "queries": b.queries,
        "timings_ms": b.timings_ms,
        **{
            t: [_item(it) for it in getattr(b, t)]
            for t in ("casos", "mensajes", "senales", "guias")
        },
    }


def ops_metrics(pred_rows: list[dict]) -> dict:
    usages = [r["usage"] for r in pred_rows if r.get("usage")]
    if not usages:
        return {}
    stages = sorted({s for u in usages for s in u.get("latencias_ms", {})})
    lat = {}
    for s in [*stages, "total"]:
        vals = [
            sum(u["latencias_ms"].values()) if s == "total" else u["latencias_ms"].get(s, 0.0)
            for u in usages
        ]
        lat[s] = {
            "p50": round(float(np.percentile(vals, 50)), 1),
            "p95": round(float(np.percentile(vals, 95)), 1),
        }
    tok_in = [u["tokens"]["in"] for u in usages]
    tok_out = [u["tokens"]["out"] for u in usages]
    cost = [u["costo_usd"] for u in usages]
    return {
        "latencia_ms": lat,
        "tokens_in_promedio": round(float(np.mean(tok_in)), 1),
        "tokens_out_promedio": round(float(np.mean(tok_out)), 1),
        "costo_usd_por_analisis": round(float(np.mean(cost)), 6),
        "costo_usd_total": round(float(np.sum(cost)), 6),
        "costo_usd_pagado_total": round(float(sum(u["costo_usd_pagado"] for u in usages)), 6),
        "llm_calls": int(sum(u["llm_calls"] for u in usages)),
        "llm_cache_hits": int(sum(u["llm_cache_hits"] for u in usages)),
        "nota": "latencias con caché no representan la API; medir p95 con SAFEFISH_NO_CACHE=1",
    }


def run_eval(
    cfg: PipelineConfig,
    cases: list[GoldenCase],
    pipeline: SafeFishPipeline,
    out_root: Path,
    *,
    split: str,
    golden_file: Path,
    workers: int = 4,
    plan_judge=None,
    retrieval_only: bool = False,
) -> tuple[Path, dict]:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_id = f"{cfg.version}-{split}-{stamp}"
    out_dir = out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=False)

    def one(case: GoldenCase) -> PipelineResult:
        raw = case.to_raw_input()
        return pipeline.retrieve_only(raw) if retrieval_only else pipeline.run(raw)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        results = list(ex.map(one, cases))

    preds = [prediction_row(c, r) for c, r in zip(cases, results, strict=True)]
    retr = [retrieval_row(c, r) for c, r in zip(cases, results, strict=True)]
    juez = None
    if plan_judge is not None:
        from eval.plan_judge import judge_rows

        juez = judge_rows(
            preds, {c.id: c.pasos_clave_esperados for c in cases}, plan_judge, workers
        )
    for name, rows in (("predictions.jsonl", preds), ("retrieval.jsonl", retr)):
        with open(out_dir / name, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    gold = {
        c.id: {
            "label": c.label,
            "modalidad": c.modalidad.value if c.modalidad else None,
            "entidad": c.entidad,
            "senales": [s.value for s in c.senales_esperadas],
        }
        for c in cases
    }
    retrieval_metrics = (
        compute_retrieval_metrics(
            retr, gold, pipeline.kb_manifest.get("counts", {}), cfg.retrieval.k_casos()
        )
        if cfg.retrieval.enabled
        else {}
    )
    reasoner = pipeline.reasoner
    planner = pipeline.planner
    metrics = {
        "meta": {
            "run_id": run_id,
            "version": cfg.version,
            "config_hash": cfg.config_hash(),
            "kb_snapshot": cfg.kb_snapshot if cfg.retrieval.enabled else None,
            "kb_hash": pipeline.kb_manifest.get("kb_hash"),
            "split": split,
            "golden_path": str(golden_file),
            "golden_version": golden_version(golden_file),
            "n_casos": len(cases),
            "git_sha": _git("rev-parse", "--short", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "fecha": datetime.now(UTC).isoformat(timespec="seconds"),
            "modelos": {
                "reasoner": getattr(getattr(reasoner, "llm", None), "model", None),
                "embeddings": pipeline.kb_manifest.get("embeddings", {}).get("model"),
                "planner": getattr(getattr(planner, "llm", None), "model", None),
                "normalizer": pipeline.kb_manifest.get("normalizer_version"),
                "plan_judge": getattr(plan_judge, "model", None),
            },
            "prompt": {"name": getattr(reasoner, "prompt", None), "hash": getattr(reasoner, "prompt_hash", None)},
            "planner": {"mode": planner.mode, "prompt": planner.prompt, "hash": planner.prompt_hash} if planner else None,
            "verifier": sorted(pipeline.verifier.rules) if getattr(pipeline.verifier, "rules", None) else None,
        },
        "modo": "retrieval_only" if retrieval_only else "completo",
        "task": {} if retrieval_only else compute_task_metrics(preds),
        "extraccion": {} if retrieval_only else compute_extraction_metrics(preds),
        "retrieval": retrieval_metrics,
        "ops": ops_metrics(preds),
    }  # fmt: skip
    metrics["juez_plan"] = juez
    metrics["por_slice"] = by_slice(preds)
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "costs.json").write_text(
        json.dumps(
            {"por_caso": {r["id"]: r["usage"] for r in preds}, "totales": metrics["ops"]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return out_dir, metrics


def print_summary(out_dir: Path, metrics: dict) -> None:
    t, r, o = metrics["task"], metrics["retrieval"], metrics["ops"]
    if not t:  # retrieval-only: no hay métricas de tarea
        print(f"\n{metrics['meta']['run_id']} (solo retrieval) → {out_dir}")
        for k, v in r.items():
            print(f"  {k:<28} {v}")
        return
    print(f"\n{metrics['meta']['run_id']} → {out_dir}")
    for k in KEY_METRICS:
        print(f"  {k:<28} {t.get(k)}")
    for k in ("hit_modalidad@k", "mrr_casos", "ndcg@k_casos", "pureza_knn_mensajes"):
        if k in r:
            print(f"  {k:<28} {r[k]}")
    if t.get("eventos_verificador"):
        print(f"  {'eventos_verificador':<28} {t['eventos_verificador']}")
    for k, v in (metrics.get("extraccion") or {}).items():
        print(f"  {k:<28} {v}")
    if o:
        print(f"  {'costo_usd_por_analisis':<28} {o['costo_usd_por_analisis']}")
        print(f"  {'latencia_total_p95_ms':<28} {o['latencia_ms']['total']['p95']}")


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--golden-dir", default="golden")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "results"))
    ap.add_argument("--judge-plan", action="store_true", help="cobertura del plan con juez LLM")
    ap.add_argument("--judge-model", default="gpt-4.1-mini-2025-04-14")
    ap.add_argument(
        "--retrieval-only",
        action="store_true",
        help="barrido de solo retrieval: sin generación (§2.0)",
    )
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    gfile = golden_path(args.golden_dir, args.split)
    cases = load_golden(gfile)[: args.limit] if args.limit else load_golden(gfile)
    pipeline = SafeFishPipeline.from_config(cfg)
    judge = None
    if args.judge_plan:
        from safefish.llm import StructuredLLM

        judge = StructuredLLM.from_cfg(args.judge_model, 0.0, 7)
    out_dir, metrics = run_eval(
        cfg,
        cases,
        pipeline,
        Path(args.out_dir),
        split=args.split,
        golden_file=gfile,
        workers=args.workers,
        plan_judge=judge,
        retrieval_only=args.retrieval_only,
    )
    print_summary(out_dir, metrics)
    with open(out_dir / "predictions.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    for e in (x for x in rows if x["error"]):
        print(f"  ✗ {e['id']}: {e['error'][:200]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
