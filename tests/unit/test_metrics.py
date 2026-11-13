import numpy as np
import pytest

from eval.compare import mcnemar
from eval.metrics_retrieval import compute_retrieval_metrics, ndcg
from eval.metrics_task import compute_task_metrics


def _row(id_, label, ver, mod_gold=None, mod_pred="no_determinada", sig_gold=(), sigs=(), slices=(), texto="pasame el código ya", esperado=None):  # fmt: skip
    pred = None if ver is None else {
        "veredicto": ver, "modalidad": mod_pred,
        "senales": [{"signal_id": s, "cita": c} for s, c in sigs], "casos_similares": [], "plan": [],
    }  # fmt: skip
    return {
        "id": id_, "label": label, "slices": list(slices), "canal": "sms", "texto": texto,
        "veredicto_esperado": esperado or (["muy_sospechoso"] if label == "estafa" else ["parece_legitimo"]),
        "modalidad_gold": mod_gold, "senales_gold": list(sig_gold), "pred": pred, "reintentos": 0,
    }  # fmt: skip


ROWS = [
    _row("1", "estafa", "estafa_confirmada", "robo_cuenta_whatsapp", "robo_cuenta_whatsapp",
         ["pedido_codigo_verificacion"], [("pedido_codigo_verificacion", "pasame el código")]),
    _row("2", "estafa", "levemente_sospechoso", "phishing_bancario", "no_determinada"),  # FN
    _row("3", "legitimo", "muy_sospechoso", slices=["hard_negative"],
         sigs=[("urgencia_temporal", "texto inventado que no está")]),  # FP + cita alucinada
    _row("4", "legitimo", "parece_legitimo", slices=["hard_negative"]),
    _row("5", "estafa", None, "inversion_falsa"),  # fallo → negativo
]  # fmt: skip


def test_metricas_de_tarea():
    m = compute_task_metrics(ROWS)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 2, 1)
    assert m["recall_estafas"] == pytest.approx(1 / 3, abs=1e-4)
    assert m["f1_binario"] == pytest.approx(0.4, abs=1e-4)
    assert m["fpr_hard_negative"] == 0.5
    assert m["acc_modalidad"] == pytest.approx(1 / 3, abs=1e-4)
    assert m["grounding_citas"] == 0.5
    assert m["n_errores"] == 1
    assert m["json_valido_sin_reintento"] == 0.8


def test_ndcg_perfecto_y_parcial():
    assert ndcg([2, 1, 0], n_rel2=1, n_rel1=1, k=3) == pytest.approx(1.0)
    assert ndcg([0, 1, 2], n_rel2=1, n_rel1=1, k=3) < 1.0
    assert ndcg([0, 0], 0, 0, 2) is None


def test_metricas_de_retrieval_derivadas_de_metadata():
    rows = [{
        "id": "1",
        "casos": [{"metadata": {"modalidad": "phishing_bancario"}},
                  {"metadata": {"modalidad": "robo_cuenta_whatsapp", "entidad_suplantada": "contacto"}}],
        "mensajes": [{"metadata": {"label": "estafa"}}, {"metadata": {"label": "legitimo"}}],
        "senales": [{"metadata": {"signal_id": "pedido_codigo_verificacion"}}],
    }]  # fmt: skip
    gold = {"1": {"label": "estafa", "modalidad": "robo_cuenta_whatsapp", "entidad": "contacto",
                  "senales": ["pedido_codigo_verificacion", "urgencia_temporal"]}}  # fmt: skip
    counts = {"chunks_casos_por_modalidad": {"robo_cuenta_whatsapp": 2},
              "chunks_casos_por_modalidad_entidad": {"robo_cuenta_whatsapp|contacto": 1}}  # fmt: skip
    m = compute_retrieval_metrics(rows, gold, counts, k=6)
    assert m["hit_modalidad@k"] == 1.0
    assert m["mrr_casos"] == 0.5
    assert m["recall@k_casos"] == 0.5
    assert m["pureza_knn_mensajes"] == 0.5
    assert m["recall@k_senales"] == 0.5


def test_mcnemar():
    a = np.array([True] * 10 + [False] * 10)
    b = np.array([True] * 10 + [True] * 9 + [False])
    r = mcnemar(a, b)
    assert (r["a_ok_b_mal"], r["a_mal_b_ok"]) == (0, 9)
    assert r["p_value"] < 0.01
