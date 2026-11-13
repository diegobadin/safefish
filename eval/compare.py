"""Comparación pareada entre dos corridas (§4.1): deltas, IC 95 % bootstrap y McNemar.

    python -m eval.compare --a results/<run_v0> --b results/<run_v1>

Escribe `results/compare_<a>_<b>.md`. Con ~120 casos las diferencias chicas son ruido: la
decisión de adoptar una versión se apoya en el IC del delta y en el p-valor de McNemar.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from eval.metrics_task import binary_metrics, by_slice, modalidad_metrics, y_pred, y_true

HEADLINE = [
    ("task", "f1_binario"), ("task", "recall_estafas"), ("task", "fpr_hard_negative"),
    ("task", "veredicto_aceptable"), ("task", "kappa_ponderado"), ("task", "acc_modalidad"),
    ("task", "macro_f1_modalidad"), ("task", "f1_senales"), ("task", "grounding_citas"),
    ("task", "grounding_citas_pre_r1"), ("task", "soporte_modalidad"),
    ("task", "contactos_no_respaldados"), ("task", "cobertura_plan"),
    ("task", "negative_rejection"), ("task", "asr_inyeccion"),
    ("task", "json_valido_sin_reintento"), ("task", "n_errores"),
    ("retrieval", "hit_modalidad@k"), ("retrieval", "mrr_casos"), ("retrieval", "ndcg@k_casos"),
    ("retrieval", "pureza_knn_mensajes"),
    ("ops", "costo_usd_por_analisis"), ("ops", "tokens_in_promedio"),
]  # fmt: skip


def load_run(run_dir: str | Path) -> tuple[dict, dict[str, dict]]:
    run_dir = Path(run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    with open(run_dir / "predictions.jsonl", encoding="utf-8") as f:
        rows = {r["id"]: r for r in map(json.loads, f)}
    return metrics, rows


def mcnemar(a_ok: np.ndarray, b_ok: np.ndarray) -> dict:
    """McNemar exacto (binomial sobre pares discordantes)."""
    b = int((a_ok & ~b_ok).sum())  # A acierta, B falla
    c = int((~a_ok & b_ok).sum())  # A falla, B acierta
    p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    return {"a_ok_b_mal": b, "a_mal_b_ok": c, "p_value": round(float(p), 4)}


def paired_bootstrap(
    rows_a: list[dict], rows_b: list[dict], metric: Callable[[list[dict]], float | None],
    n: int = 2000, seed: int = 0,
) -> dict:  # fmt: skip
    rng = np.random.default_rng(seed)
    idx = np.arange(len(rows_a))
    deltas = []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        ma, mb = metric([rows_a[i] for i in s]), metric([rows_b[i] for i in s])
        if ma is not None and mb is not None:
            deltas.append(mb - ma)
    if not deltas:
        return {"delta": None, "ic95": None}
    point_a, point_b = metric(rows_a), metric(rows_b)
    return {
        "delta": None if point_a is None or point_b is None else round(point_b - point_a, 4),
        "ic95": [
            round(float(np.percentile(deltas, 2.5)), 4),
            round(float(np.percentile(deltas, 97.5)), 4),
        ],
    }


def compare(run_a: str | Path, run_b: str | Path) -> tuple[str, dict]:
    ma, ra = load_run(run_a)
    mb, rb = load_run(run_b)
    warnings = []
    if ma["meta"]["golden_version"] != mb["meta"]["golden_version"]:
        warnings.append(
            "Las corridas usan versiones distintas del golden: la comparación no es válida."
        )
    ids = sorted(set(ra) & set(rb))
    if len(ids) < max(len(ra), len(rb)):
        warnings.append(f"Solo {len(ids)} casos en común (A={len(ra)}, B={len(rb)}).")
    A, B = [ra[i] for i in ids], [rb[i] for i in ids]

    a_ok = np.array([y_pred(r) == y_true(r) for r in A])
    b_ok = np.array([y_pred(r) == y_true(r) for r in B])
    mod_ids = [i for i, r in enumerate(A) if r["label"] == "estafa" and r.get("modalidad_gold")]

    def _mod_ok(rows):
        return np.array([bool(rows[i]["pred"]) and rows[i]["pred"]["modalidad"] == rows[i]["modalidad_gold"] for i in mod_ids])  # fmt: skip

    stats = {
        "mcnemar_binario": mcnemar(a_ok, b_ok),
        "mcnemar_modalidad": mcnemar(_mod_ok(A), _mod_ok(B)) if mod_ids else None,
        "bootstrap": {
            "f1_binario": paired_bootstrap(A, B, lambda rs: binary_metrics(rs)["f1_binario"]),
            "recall_estafas": paired_bootstrap(
                A, B, lambda rs: binary_metrics(rs)["recall_estafas"]
            ),
            "acc_modalidad": paired_bootstrap(
                A, B, lambda rs: modalidad_metrics(rs)["acc_modalidad"]
            ),
        },
    }

    name_a, name_b = ma["meta"]["version"], mb["meta"]["version"]
    lines = [
        f"# Comparación {name_a} → {name_b}",
        "",
        f"- A: `{ma['meta']['run_id']}` (config `{ma['meta']['config_hash']}`, kb `{ma['meta']['kb_snapshot']}`)",
        f"- B: `{mb['meta']['run_id']}` (config `{mb['meta']['config_hash']}`, kb `{mb['meta']['kb_snapshot']}`)",
        f"- Golden: `{ma['meta']['golden_path']}` v`{ma['meta']['golden_version']}` · n = {len(ids)}",
        *[f"- ⚠️ {w}" for w in warnings],
        "",
        "## Métricas",
        "",
        f"| Métrica | {name_a} | {name_b} | Δ |",
        "|---|---|---|---|",
    ]
    for sec, key in HEADLINE:
        va, vb = ma.get(sec, {}).get(key), mb.get(sec, {}).get(key)
        d = (
            round(vb - va, 4)
            if isinstance(va, int | float) and isinstance(vb, int | float)
            else "—"
        )
        lines.append(f"| {sec}.{key} | {va} | {vb} | {d} |")

    lines += ["", "## Significancia", ""]
    for k, v in stats["bootstrap"].items():
        lines.append(f"- Δ {k}: {v['delta']} · IC 95 % bootstrap pareado {v['ic95']}")
    m = stats["mcnemar_binario"]
    lines.append(
        f"- McNemar (acierto binario): A✓B✗ = {m['a_ok_b_mal']}, A✗B✓ = {m['a_mal_b_ok']}, p = {m['p_value']}"
    )
    if stats["mcnemar_modalidad"]:
        m = stats["mcnemar_modalidad"]
        lines.append(
            f"- McNemar (modalidad): A✓B✗ = {m['a_ok_b_mal']}, A✗B✓ = {m['a_mal_b_ok']}, p = {m['p_value']}"
        )

    sa, sb = by_slice(A), by_slice(B)
    lines += ["", "## Por slice", "", f"| Slice | n | F1 {name_a} | F1 {name_b} | FPR {name_a} | FPR {name_b} |", "|---|---|---|---|---|---|"]  # fmt: skip
    for k in sorted(set(sa) | set(sb)):
        xa, xb = sa.get(k, {}), sb.get(k, {})
        lines.append(f"| {k} | {xa.get('n', xb.get('n'))} | {xa.get('f1_binario')} | {xb.get('f1_binario')} | {xa.get('fpr')} | {xb.get('fpr')} |")  # fmt: skip

    flips = [
        (r_a["id"], y_pred(r_a), y_pred(r_b), r_a["label"])
        for r_a, r_b in zip(A, B, strict=True)
        if y_pred(r_a) != y_pred(r_b)
    ]
    if flips:
        lines += [
            "",
            "## Casos que cambian de decisión binaria",
            "",
            "| id | label | A | B |",
            "|---|---|---|---|",
        ]
        lines += [f"| {i} | {lab} | {pa} | {pb} |" for i, pa, pb, lab in flips]
    return "\n".join(lines) + "\n", stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="corrida base (p. ej. V0)")
    ap.add_argument("--b", required=True, help="corrida nueva (p. ej. V1)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    md, _ = compare(args.a, args.b)
    out = (
        Path(args.out)
        if args.out
        else Path(args.b).parent / f"compare_{Path(args.a).name}_{Path(args.b).name}.md"
    )
    out.write_text(md, encoding="utf-8")
    print(md)
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
