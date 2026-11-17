from safefish.schemas import EvidenceItem
from safefish.trace import activa, item, traza, trazando


def test_sin_traza_activa_no_hace_nada():
    assert not activa()
    traza("busqueda", coleccion="casos")  # no falla ni guarda nada


def test_trazando_registra_pasos_en_orden():
    with trazando() as t:
        assert activa()
        traza("extraccion", origen="texto")
        traza("busqueda", coleccion="casos")
    assert not activa()
    assert [p["etapa"] for p in t.pasos] == ["extraccion", "busqueda"]
    assert t.pasos[1]["coleccion"] == "casos" and "t_ms" in t.pasos[0]


def test_item_resume_documento_y_puntajes():
    it = EvidenceItem(
        id="c1#0", coleccion="casos", texto="x " * 300,
        metadata={"doc_id": "caso-1", "modalidad": "phishing_bancario", "score_rrf": 0.03},
        score_dense=0.81, score_rerank=3.0,
    )  # fmt: skip
    d = item(it, 2)
    assert d["doc"] == "caso-1" and d["rank"] == 2 and d["rrf"] == 0.03 and d["rerank"] == 3.0
    assert len(d["texto"]) <= 240 and "bm25" not in d
