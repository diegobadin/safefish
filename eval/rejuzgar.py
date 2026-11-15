"""Re-juzga la cobertura del plan de corridas ya hechas con otro juez, sin re-correr el pipeline.

    python -m eval.rejuzgar results/v6-test-... results/v8-dev-... [--judge-model claude-sonnet-5-5]

Sirve para comparar corridas hechas con jueces distintos: todas se puntúan con el mismo juez.
Lee `predictions.jsonl` y el golden del split, y escribe en la carpeta de la corrida
`cobertura_<juez>.json` con la cobertura promedio, la de cada caso y el costo del juez.
No modifica `metrics.json` ni `predictions.jsonl`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from eval.golden import golden_path, load_golden
from eval.plan_judge import judge_rows


def rejuzgar(run_dir: Path, judge, golden_dir: str = "golden", workers: int = 4) -> dict:
    rows = [json.loads(ln) for ln in open(run_dir / "predictions.jsonl", encoding="utf-8")]
    for r in rows:
        r.pop("cobertura_plan", None)
    split = rows[0]["split"]
    gold = {c.id: c.pasos_clave_esperados for c in load_golden(golden_path(golden_dir, split))}
    resumen = judge_rows(rows, gold, judge, workers)
    por_caso = {r["id"]: r["cobertura_plan"] for r in rows if r.get("cobertura_plan") is not None}
    out = {
        "run": run_dir.name,
        "juez": resumen,
        "cobertura_plan": round(float(np.mean(list(por_caso.values()))), 4) if por_caso else None,
        "n_cobertura_plan": len(por_caso),
        "por_caso": por_caso,
    }
    nombre = f"cobertura_{judge.model}.json"
    with open(run_dir / nombre, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return out


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--judge-model", default="claude-sonnet-5-5")
    ap.add_argument("--golden-dir", default="golden")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)

    from safefish.llm import StructuredLLM

    judge = StructuredLLM.from_cfg(args.judge_model, 0.0, 7)
    for run in args.runs:
        out = rejuzgar(Path(run), judge, args.golden_dir, args.workers)
        print(f"{out['run']}: cobertura {out['cobertura_plan']} (n={out['n_cobertura_plan']}, "
              f"fallidos={out['juez']['n_fallidos']}, USD {out['juez']['costo_usd']})")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
