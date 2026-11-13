"""Métricas determinísticas de retrieval (§4.3) sobre `retrieval.jsonl`.

La relevancia de un caso se **deriva de la metadata** (no hay que etiquetar ids a mano):
2 = misma modalidad y entidad que el gold; 1 = misma modalidad; 0 = otra. Los totales de
relevantes por modalidad salen del manifest del snapshot (para Recall@k y el nDCG ideal).

En V1 (una colección) la lista de `casos` son los chunks de documentos tipo caso dentro del
top-k total; el ranking se mide dentro de esa lista, igual que en V2+ con colecciones propias.
"""

from __future__ import annotations

import math

import numpy as np


def relevance(item_md: dict, gold_mod: str, gold_ent: str | None) -> int:
    if item_md.get("modalidad") != gold_mod:
        return 0
    ent = item_md.get("entidad_suplantada", item_md.get("entidad"))
    return 2 if gold_ent and ent == gold_ent else 1


def dcg(rels: list[int]) -> float:
    return sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg(rels: list[int], n_rel2: int, n_rel1: int, k: int) -> float | None:
    ideal = ([2] * n_rel2 + [1] * n_rel1)[:k]
    idcg = dcg(ideal)
    return dcg(rels[:k]) / idcg if idcg > 0 else None


def _mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(float(np.mean(xs)), 4) if xs else None


def compute_retrieval_metrics(
    rows: list[dict], gold: dict[str, dict], kb_counts: dict, k: int
) -> dict:
    """rows: una fila por caso con `id` y listas `casos`, `mensajes`, `senales`, `guias` (cada
    ítem con su metadata). gold: id → {label, modalidad, entidad, senales}."""
    if not rows:
        return {}
    por_mod = kb_counts.get("chunks_casos_por_modalidad", {})
    por_mod_ent = kb_counts.get("chunks_casos_por_modalidad_entidad", {})

    hits, mrrs, recalls, ndcgs, pureza, rec_sen = [], [], [], [], [], []
    sin_cobertura = 0
    tipos = {"casos": [], "mensajes": [], "senales": [], "guias": []}
    for r in rows:
        g = gold[r["id"]]
        for t in tipos:
            tipos[t].append(len(r.get(t, [])))

        mensajes = r.get("mensajes", [])
        if mensajes:
            pureza.append(np.mean([m["metadata"].get("label") == g["label"] for m in mensajes]))

        if g.get("senales"):
            got = {s["metadata"].get("signal_id") for s in r.get("senales", [])}
            rec_sen.append(len(set(g["senales"]) & got) / len(g["senales"]))

        mod = g.get("modalidad")
        if g["label"] != "estafa" or not mod or mod == "no_determinada":
            continue
        n_rel = int(por_mod.get(mod, 0))
        if n_rel == 0:
            sin_cobertura += 1
            continue
        n_rel2 = int(por_mod_ent.get(f"{mod}|{g.get('entidad')}", 0)) if g.get("entidad") else 0
        rels = [relevance(c["metadata"], mod, g.get("entidad")) for c in r.get("casos", [])][:k]
        hits.append(float(any(rels)))
        first = next((i for i, x in enumerate(rels) if x), None)
        mrrs.append(1.0 / (first + 1) if first is not None else 0.0)
        recalls.append(sum(1 for x in rels if x) / min(k, n_rel))
        ndcgs.append(ndcg(rels, n_rel2, n_rel - n_rel2, k))

    return {
        "k": k,
        "n_eval_casos": len(hits),
        "n_sin_cobertura_kb": sin_cobertura,
        "hit_modalidad@k": _mean(hits),
        "mrr_casos": _mean(mrrs),
        "recall@k_casos": _mean(recalls),
        "ndcg@k_casos": _mean(ndcgs),
        "pureza_knn_mensajes": _mean(pureza),
        "recall@k_senales": _mean(rec_sen),
        "items_promedio_por_tipo": {t: round(float(np.mean(v)), 2) for t, v in tipos.items()},
    }
