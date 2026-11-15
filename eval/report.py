"""Tabla de la ablación (§2.0): una fila por versión con la última corrida de cada split.

    python -m eval.report [--split test] [--results results] [--out docs/ablation_tabla.md]

Lee `results/*/metrics.json` (solo corridas completas, no las de solo retrieval), se queda con
la más reciente por (versión, split) y escribe una tabla Markdown con las métricas de
`eval.compare.HEADLINE`. Las significancias (IC bootstrap, McNemar) las da `make compare`
entre pares de versiones.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval.compare import HEADLINE
from safefish.config import REPO_ROOT

ORDEN = ["v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"]
COLUMNAS = [
    ("task", "f1_binario", "F1"), ("task", "recall_estafas", "Recall"),
    ("task", "fpr_hard_negative", "FPR neg. dif."), ("task", "veredicto_aceptable", "Veredicto ok"),
    ("task", "acc_modalidad", "Acc. modalidad"), ("task", "f1_senales", "F1 señales"),
    ("task", "grounding_citas", "Grounding"), ("task", "cobertura_plan", "Cobertura plan"),
    ("task", "asr_inyeccion", "ASR inyección"),
    ("retrieval", "hit_modalidad@k", "Hit-mod@k"), ("retrieval", "mrr_casos", "MRR casos"),
    ("ops", "costo_usd_por_analisis", "USD/análisis"),
]  # fmt: skip
assert all((g, k) in HEADLINE for g, k, _ in COLUMNAS)


def ultimas(results: Path, split: str) -> dict[str, tuple[Path, dict]]:
    por_version: dict[str, tuple[str, Path, dict]] = {}
    for f in results.glob("*/metrics.json"):
        m = json.loads(f.read_text(encoding="utf-8"))
        meta = m.get("meta", {})
        if m.get("modo") != "completo" or meta.get("split") != split:
            continue
        v, fecha = meta.get("version"), meta.get("fecha", "")
        if v not in por_version or fecha > por_version[v][0]:
            por_version[v] = (fecha, f.parent, m)
    return {v: (d, m) for v, (_, d, m) in por_version.items()}


def _fmt(x) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.3f}" if abs(x) < 10 else f"{x:.1f}"
    return str(x)


def tabla(runs: dict[str, tuple[Path, dict]]) -> str:
    head = "| Versión | n | " + " | ".join(c for _, _, c in COLUMNAS) + " | Corrida |"
    sep = "|" + "---|" * (len(COLUMNAS) + 3)
    filas = [head, sep]
    for v in sorted(runs, key=lambda v: (ORDEN.index(v) if v in ORDEN else 99, v)):
        d, m = runs[v]
        vals = [_fmt((m.get(g) or {}).get(k)) for g, k, _ in COLUMNAS]
        filas.append(
            f"| {v} | {m['meta'].get('n_casos')} | " + " | ".join(vals) + f" | `{d.name}` |"
        )
    return "\n".join(filas)


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--results", default=str(REPO_ROOT / "results"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    runs = ultimas(Path(args.results), args.split)
    if not runs:
        print(f"No hay corridas completas de {args.split} en {args.results}")
        return 1
    md = f"### Ablación · split {args.split}\n\n{tabla(runs)}\n"
    print(md)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"→ {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
