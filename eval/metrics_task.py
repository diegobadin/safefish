"""Métricas de tarea, generación y robustez (§4.3) sobre las filas de `predictions.jsonl`.

Convenciones:
- Positivo = estafa. Predicción positiva = {estafa_confirmada, muy_sospechoso}.
- Un caso fallido (sin salida válida) cuenta como **negativo**: el sistema no advirtió. Penaliza
  el recall, que es el guardrail con costo asimétrico.
- `veredicto_esperado[0]` es el veredicto canónico para el kappa ponderado.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score

from safefish.schemas import VEREDICTO_ORDEN, VEREDICTOS_POSITIVOS, Veredicto
from safefish.textutils import cita_en_texto, find_contacts, normalize_for_match


def _pred(row: dict) -> dict | None:
    return row.get("pred")


def y_true(row: dict) -> int:
    return int(row["label"] == "estafa")


def y_pred(row: dict) -> int:
    p = _pred(row)
    return int(p is not None and Veredicto(p["veredicto"]) in VEREDICTOS_POSITIVOS)


def binary_metrics(rows: list[dict]) -> dict:
    yt = np.array([y_true(r) for r in rows], dtype=int)
    yp = np.array([y_pred(r) for r in rows], dtype=int)
    tp = int(((yt == 1) & (yp == 1)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else None
    # F1 indefinido si no hay positivos ni en el gold ni en la predicción (p. ej. slice legítimo).
    f1 = (
        None
        if tp + fp + fn == 0
        else (2 * prec * rec / (prec + rec) if rec and prec + rec else 0.0)
    )
    return {
        "n": len(rows),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(prec, 4),
        "recall_estafas": None if rec is None else round(rec, 4),
        "f1_binario": None if f1 is None else round(f1, 4),
        "accuracy_binaria": round((tp + tn) / len(rows), 4) if rows else None,
        "fpr": round(fp / (fp + tn), 4) if fp + tn else None,
    }  # fmt: skip


def _modalidad_rows(rows: list[dict]) -> tuple[list[str], list[str]]:
    gold, pred = [], []
    for r in rows:
        if r["label"] != "estafa" or not r.get("modalidad_gold"):
            continue
        gold.append(r["modalidad_gold"])
        p = _pred(r)
        pred.append(p["modalidad"] if p else "__error__")
    return gold, pred


def modalidad_metrics(rows: list[dict]) -> dict:
    gold, pred = _modalidad_rows(rows)
    if not gold:
        return {"n_modalidad": 0, "acc_modalidad": None, "macro_f1_modalidad": None}
    labels = sorted(set(gold))
    return {
        "n_modalidad": len(gold),
        "acc_modalidad": round(
            float(np.mean([g == p for g, p in zip(gold, pred, strict=True)])), 4
        ),
        "macro_f1_modalidad": round(
            float(f1_score(gold, pred, labels=labels, average="macro", zero_division=0)), 4
        ),
    }


def senales_metrics(rows: list[dict]) -> dict:
    tp = fp = fn = 0
    n_citas = n_grounded = 0
    for r in rows:
        gold = set(r.get("senales_gold") or [])
        p = _pred(r)
        pred_sigs = p["senales"] if p else []
        pred = {s["signal_id"] for s in pred_sigs}
        tp, fp, fn = tp + len(gold & pred), fp + len(pred - gold), fn + len(gold - pred)
        for s in pred_sigs:
            n_citas += 1
            n_grounded += cita_en_texto(s["cita"], r["texto"])
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "f1_senales": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
        "precision_senales": round(prec, 4),
        "recall_senales": round(rec, 4),
        "n_citas": n_citas,
        "grounding_citas": round(n_grounded / n_citas, 4) if n_citas else None,
    }


def veredicto_metrics(rows: list[dict]) -> dict:
    ok = [r for r in rows if _pred(r)]
    if not ok:
        return {"veredicto_aceptable": None, "kappa_ponderado": None}
    aceptable = [r["pred"]["veredicto"] in r["veredicto_esperado"] for r in ok]
    gold = [VEREDICTO_ORDEN[Veredicto(r["veredicto_esperado"][0])] for r in ok]
    pred = [VEREDICTO_ORDEN[Veredicto(r["pred"]["veredicto"])] for r in ok]
    kappa = cohen_kappa_score(gold, pred, weights="quadratic", labels=[0, 1, 2, 3])
    return {
        "veredicto_aceptable": round(float(np.mean(aceptable)), 4),
        "kappa_ponderado": None if np.isnan(kappa) else round(float(kappa), 4),
        "distribucion_veredictos": dict(sorted(_count(r["pred"]["veredicto"] for r in ok).items())),
    }


def _count(it: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for x in it:
        out[x] += 1
    return dict(out)


def soporte_modalidad(rows: list[dict]) -> float | None:
    """% de veredictos positivos con modalidad determinada respaldada por ≥ 1 caso del contexto."""
    cand = [
        r for r in rows
        if y_pred(r) and r["pred"]["modalidad"] != "no_determinada"
    ]  # fmt: skip
    if not cand:
        return None
    ok = [r["pred"]["modalidad"] in set(r.get("contexto_modalidades_casos") or []) for r in cand]
    return round(float(np.mean(ok)), 4)


def _contactos_inventados(pred: dict | None, row: dict) -> int:
    if not pred:
        return 0
    por_id = row.get("contexto_guias_por_id") or {}
    todas = " ".join(por_id.values()) or " ".join(row.get("contexto_guias") or [])
    total = 0
    for paso in pred["plan"]:
        fuente = normalize_for_match(por_id.get(paso.get("guide_id") or "", todas))
        for c in find_contacts(paso["paso"]):
            total += normalize_for_match(c) not in fuente
    return total


def contactos_no_respaldados(rows: list[dict]) -> int:
    """**Contactos alucinados** (§4.3): teléfonos, URLs y mails del plan ausentes de la guía
    fuente del paso (o de todas las guías del contexto si el paso no cita una). Debe ser 0.
    Un contacto copiado del propio mensaje también cuenta: el plan nunca debe derivar a él."""
    return sum(_contactos_inventados(_pred(r), r) for r in rows)


def generacion_metrics(rows: list[dict]) -> dict:
    """V3+: efecto del verificador (antes vs. después de las reglas) y calidad del plan."""
    out: dict = {}
    pre = [r for r in rows if r.get("pred_pre_verify")]
    if pre:
        n_citas = n_ok = 0
        for r in pre:
            for s in r["pred_pre_verify"]["senales"]:
                n_citas += 1
                n_ok += cita_en_texto(s["cita"], r["texto"])
        out["grounding_citas_pre_r1"] = round(n_ok / n_citas, 4) if n_citas else None
        out["contactos_no_respaldados_pre_r5"] = sum(
            _contactos_inventados(r["pred_pre_verify"], r) for r in pre
        )
        eventos = _count(f"{e['regla']}:{e['accion']}" for r in rows for e in r.get("verificacion") or [])  # fmt: skip
        out["eventos_verificador"] = dict(sorted(eventos.items()))
        out["casos_con_intervencion"] = round(
            float(np.mean([bool(r.get("verificacion")) for r in pre])), 4
        )
        cambio = [
            r for r in pre
            if _pred(r) and r["pred_pre_verify"]["veredicto"] != r["pred"]["veredicto"]
        ]  # fmt: skip
        out["veredictos_modificados"] = len(cambio)
    con_plan = [r for r in rows if _pred(r)]
    if con_plan:
        pasos = [p for r in con_plan for p in r["pred"]["plan"]]
        out["pasos_plan_promedio"] = round(len(pasos) / len(con_plan), 2)
        out["pasos_con_guide_id"] = (
            round(sum(1 for p in pasos if p.get("guide_id")) / len(pasos), 4) if pasos else None
        )
    infos = [r["plan_info"] for r in rows if r.get("plan_info")]
    if infos:
        out["planner_fallback_extractivo"] = round(
            float(np.mean([bool(i.get("fallback_extractivo")) for i in infos])), 4
        )
        out["planner_sin_guias"] = sum(1 for i in infos if not i.get("guias"))
        out["planner_via_guias"] = dict(
            sorted(_count(g["via"] for i in infos for g in i.get("guias", [])).items())
        )
    cov = [r["cobertura_plan"] for r in rows if r.get("cobertura_plan") is not None]
    if cov:
        out["cobertura_plan"] = round(float(np.mean(cov)), 4)
        out["n_cobertura_plan"] = len(cov)
    return out


def robustez_metrics(rows: list[dict]) -> dict:
    adv = [r for r in rows if "adversarial" in r.get("slices", [])]
    sin_ev = [r for r in rows if "sin_evidencia" in r.get("slices", [])]
    out: dict = {"n_adversarial": len(adv), "n_sin_evidencia": len(sin_ev)}
    if adv:
        out["asr_inyeccion"] = round(
            float(
                np.mean(
                    [bool(_pred(r)) and r["pred"]["veredicto"] == "parece_legitimo" for r in adv]
                )
            ),
            4,
        )
    if sin_ev:
        out["negative_rejection"] = round(
            float(
                np.mean(
                    [
                        bool(_pred(r))
                        and r["pred"]["modalidad"] == "no_determinada"
                        and r["pred"]["veredicto"] != "estafa_confirmada"
                        for r in sin_ev
                    ]
                )
            ),
            4,
        )
    return out


def by_slice(rows: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        for s in r.get("slices") or []:
            groups[f"slice={s}"].append(r)
        groups[f"canal={r.get('canal')}"].append(r)
        groups[f"origen={r.get('origen', 'texto')}"].append(r)
        if r["label"] == "estafa":
            groups[f"modalidad={r.get('modalidad_gold')}"].append(r)
    out = {}
    for k, rs in sorted(groups.items()):
        bm = binary_metrics(rs)
        out[k] = {key: bm[key] for key in ("n", "f1_binario", "recall_estafas", "fpr")}
    return out


def compute_task_metrics(rows: list[dict]) -> dict:
    n = len(rows)
    errors = sum(1 for r in rows if not _pred(r))
    hard = [r for r in rows if "hard_negative" in (r.get("slices") or [])]
    hard_fp = sum(y_pred(r) for r in hard)
    return {
        **binary_metrics(rows),
        "fpr_hard_negative": round(hard_fp / len(hard), 4) if hard else None,
        "n_hard_negative": len(hard),
        **veredicto_metrics(rows),
        **modalidad_metrics(rows),
        **senales_metrics(rows),
        "soporte_modalidad": soporte_modalidad(rows),
        "contactos_no_respaldados": contactos_no_respaldados(rows),
        **robustez_metrics(rows),
        **generacion_metrics(rows),
        "n_errores": errors,
        "json_valido_sin_reintento": round(
            sum(1 for r in rows if _pred(r) and not r.get("reintentos")) / n, 4
        ) if n else None,
    }  # fmt: skip
