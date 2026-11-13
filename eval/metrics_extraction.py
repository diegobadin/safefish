"""Métricas de extracción para el slice de capturas (§4.3, capa Extracción).

- CER/WER de la transcripción contra el texto de referencia del golden (distancia de
  Levenshtein sobre caracteres / palabras normalizadas, como `jiwer`).
- F1 de entidades (links, teléfonos, montos, CBU/alias, mails) extraídas del texto predicho
  vs. del texto de referencia.
- Localización: % de citas de referencia (`citas_bbox` del golden) cubiertas por una señal
  verificada con la misma cita (fuzzy ≥ 0.8) y bbox con IoU ≥ 0.5; y % de señales
  verificadas que tienen bbox.
"""

from __future__ import annotations

import numpy as np

from safefish.extract.align import iou
from safefish.extract.text import extract_entities
from safefish.textutils import cita_score, normalize_for_match, normalize_unicode


def levenshtein(a: list | str, b: list | str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred: str, ref: str) -> float:
    p, r = normalize_unicode(pred), normalize_unicode(ref)
    return levenshtein(p, r) / max(1, len(r))


def wer(pred: str, ref: str) -> float:
    p, r = normalize_for_match(pred).split(), normalize_for_match(ref).split()
    return levenshtein(p, r) / max(1, len(r))


def _entity_set(texto: str) -> set[tuple[str, str]]:
    e = extract_entities(texto)
    return {
        (k, normalize_for_match(v))
        for k in ("links", "telefonos", "montos", "cbu_alias", "emails")
        for v in getattr(e, k)
    }


def compute_extraction_metrics(rows: list[dict]) -> dict:
    caps = [r for r in rows if r.get("origen") == "captura" and r.get("texto_gold")]
    if not caps:
        return {}
    cers = [cer(r["texto"], r["texto_gold"]) for r in caps]
    wers = [wer(r["texto"], r["texto_gold"]) for r in caps]
    tp = fp = fn = 0
    for r in caps:
        pred, gold = _entity_set(r["texto"]), _entity_set(r["texto_gold"])
        tp, fp, fn = tp + len(pred & gold), fp + len(pred - gold), fn + len(gold - pred)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    senales = [s for r in caps if r.get("pred") for s in r["pred"]["senales"] if s["verificada"]]
    loc = []
    for r in caps:
        pred = (
            [s for s in (r["pred"] or {}).get("senales", []) if s["verificada"] and s.get("bbox")]
            if r.get("pred")
            else []
        )
        for g in r.get("citas_bbox_gold") or []:
            loc.append(
                any(
                    max(cita_score(g["cita"], s["cita"]), cita_score(s["cita"], g["cita"])) >= 0.8
                    and iou(tuple(s["bbox"]), tuple(g["bbox"])) >= 0.5
                    for s in pred
                )
            )
    return {
        "n_capturas": len(caps),
        "cer": round(float(np.mean(cers)), 4),
        "wer": round(float(np.mean(wers)), 4),
        "f1_entidades": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
        "localizacion_iou50": round(float(np.mean(loc)), 4) if loc else None,
        "n_citas_bbox_gold": len(loc),
        "senales_con_bbox": round(sum(1 for s in senales if s.get("bbox")) / len(senales), 4)
        if senales
        else None,
    }
