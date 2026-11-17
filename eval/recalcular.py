"""Recalcula las métricas de corridas ya hechas contra el golden actual, sin re-correr el pipeline.

    python -m eval.recalcular results/v9-test-... results/v6-dev-...   # o --todas

Sirve cuando el golden cambia sin que cambien los casos que quedan (por ejemplo, al sacar casos
repetidos): las predicciones de esos casos son las mismas, así que no hace falta volver a llamar a
la API. Se quedan solo las filas cuyos ids siguen en el golden del split, se recalculan las
métricas de tarea, extracción, retrieval, operación y por slice, y se reescriben
`predictions.jsonl`, `retrieval.jsonl`, `costs.json` y `metrics.json`. La cobertura del plan ya
juzgada viaja en cada fila, así que no se vuelve a juzgar. `metrics.json` registra en
`meta.recalculado` cuándo, contra qué versión del golden y cuántos casos salieron.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

from eval.golden import golden_path, golden_version, load_golden
from eval.metrics_extraction import compute_extraction_metrics
from eval.metrics_retrieval import compute_retrieval_metrics
from eval.metrics_task import by_slice, compute_task_metrics
from eval.run import ops_metrics
from safefish.config import REPO_ROOT, PipelineConfig


def _leer(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def _escribir(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def recalcular(run_dir: Path, golden_dir: str | Path = REPO_ROOT / "golden") -> dict:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    split = metrics["meta"]["split"]
    gpath = golden_path(golden_dir, split)
    cases = {c.id: c for c in load_golden(gpath)}

    preds_all = _leer(run_dir / "predictions.jsonl")
    preds = [r for r in preds_all if r["id"] in cases]
    retr = [r for r in _leer(run_dir / "retrieval.jsonl") if r["id"] in cases]
    faltan = sorted(set(cases) - {r["id"] for r in preds})
    if faltan:
        raise ValueError(
            f"{run_dir.name}: la corrida no tiene {len(faltan)} casos del golden: {faltan[:5]}"
        )

    gold = {
        c.id: {
            "label": c.label,
            "modalidad": c.modalidad.value if c.modalidad else None,
            "entidad": c.entidad,
            "senales": [s.value for s in c.senales_esperadas],
        }
        for c in cases.values()
    }
    cfg = PipelineConfig.model_validate(
        yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    )
    retrieval_only = metrics.get("modo") == "retrieval_only"
    if cfg.retrieval.enabled and retr:
        manifest_path = cfg.snapshot_dir / "manifest.json"
        counts = (
            json.loads(manifest_path.read_text(encoding="utf-8")).get("counts", {})
            if manifest_path.exists()
            else {}
        )
        metrics["retrieval"] = compute_retrieval_metrics(
            retr, gold, counts, cfg.retrieval.k_casos()
        )
    if not retrieval_only:
        metrics["task"] = compute_task_metrics(preds)
        metrics["extraccion"] = compute_extraction_metrics(preds)
    metrics["ops"] = ops_metrics(preds)
    metrics["por_slice"] = by_slice(preds)
    metrics["meta"]["n_casos"] = len(preds)
    metrics["meta"]["recalculado"] = {
        "fecha": datetime.now(UTC).isoformat(timespec="seconds"),
        "golden_version": golden_version(gpath),
        "golden_version_original": metrics["meta"].get("golden_version"),
        "casos_quitados": len(preds_all) - len(preds),
    }

    _escribir(run_dir / "predictions.jsonl", preds)
    _escribir(run_dir / "retrieval.jsonl", retr)
    (run_dir / "costs.json").write_text(
        json.dumps({"por_caso": {r["id"]: r["usage"] for r in preds}, "totales": metrics["ops"]},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )  # fmt: skip
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--todas", action="store_true", help="todas las corridas de results/")
    ap.add_argument("--golden-dir", default=str(REPO_ROOT / "golden"))
    args = ap.parse_args(argv)
    runs = [Path(r) for r in args.runs]
    if args.todas:
        runs += sorted(p.parent for p in (REPO_ROOT / "results").glob("*/metrics.json"))
    if not runs:
        ap.error("indicá corridas o --todas")
    for run in runs:
        m = recalcular(run, args.golden_dir)
        t = m.get("task") or {}
        print(f"{run.name}: n={m['meta']['n_casos']} (−{m['meta']['recalculado']['casos_quitados']}) "
              f"F1={t.get('f1_binario')} cobertura={t.get('cobertura_plan')}")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
