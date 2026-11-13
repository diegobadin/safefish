"""Regresiones de la revisión de la Fase 2 (un test por hallazgo)."""

import json

from langchain_core.documents import Document

from eval.metrics_extraction import compute_extraction_metrics
from eval.plan_judge import Juicio, PlanJudgeOutput, judge_rows
from safefish.config import NormalizeCfg, RetrievalCfg
from safefish.extract.text import TextExtractor
from safefish.ingest.build_kb import kb_hash, review_sample
from safefish.ingest.normalize import (
    CONTACTO_REDACTADO,
    GuiaNorm,
    GuiasNorm,
    LLMNormalizer,
    NormalizeReport,
    normalize_documents,
    parse_guide_template,
)
from safefish.llm import FakeStructuredLLM
from safefish.postprocess.verifier import CONTACTO_ELIMINADO, RuleVerifier
from safefish.schemas import (
    CaseDoc,
    CasoRef,
    EvidenceItem,
    PasoPlan,
    RawInput,
    RetrievalBundle,
    RunMeta,
    SafeFishOutput,
    SenalDetectada,
)
from safefish.textutils import cita_score, find_contacts, normalize_unicode


def _out(**kw):
    base = {"veredicto": "muy_sospechoso", "modalidad": "no_determinada", "meta": RunMeta(run_id="r", config_hash="h")}  # fmt: skip
    return SafeFishOutput(**{**base, **kw})


def _m(texto):
    return TextExtractor().extract(RawInput(texto=texto))


GUIA = RetrievalBundle(
    casos=[
        EvidenceItem(
            id="C-1",
            coleccion="casos",
            texto="...",
            metadata={"doc_id": "C-1", "modalidad": "falso_familiar_cambio_numero"},
        )
    ],  # fmt: skip
    guias=[
        EvidenceItem(
            id="G-1",
            coleccion="guias",
            texto="Si transferiste:\nPasos:\n1. Llamá a tu banco y pedí la reversión.",
            metadata={
                "doc_id": "G-1",
                "organismo": "BCRA",
                "pasos_json": json.dumps(["Llamá a tu banco."]),
            },  # fmt: skip
        )
    ],
)


# 1. Detector de contactos amplio -------------------------------------------------------


def test_contactos_con_tld_no_listado_puntos_y_numeros_cortos():
    assert find_contacts("Entrá a galicia-validacion.help ya") == ["galicia-validacion.help"]
    assert find_contacts("Escribí a soporte.bna.digital") == ["soporte.bna.digital"]
    assert find_contacts("Llamá al 11.5555.4444") == ["11.5555.4444"]
    assert find_contacts("Llamá al 134 o marcá *611") == ["134", "*611"]
    assert find_contacts("Transferí $12.500.000 hoy") == []  # montos no son teléfonos
    plan = [PasoPlan(paso="Validá en galicia-validacion.help o llamá al 11.5555.4444", guide_id="G-1")]  # fmt: skip
    res, ev = RuleVerifier(["R5"]).check_plan(GUIA, _out(plan=plan))
    assert "galicia-validacion" not in res.plan[0].paso and "5555" not in res.plan[0].paso
    assert len(ev) == 2


# 2. Parser de guías sobre texto ya normalizado ----------------------------------------


def test_guia_por_plantilla_no_pierde_continuaciones_ni_lineas_finales():
    raw = (
        "Si transferiste dinero:\n1. Llamá a tu banco y pedí la reversión\n"
        "   de la transferencia al 0800-999-2727.\n2. Hacé la denuncia.\n"
        "Más información en www.bcra.gob.ar"
    )
    g = parse_guide_template(normalize_unicode(raw), {"doc_id": "G"})
    assert g.pasos[0].endswith("de la transferencia al 0800-999-2727.")
    assert g.pasos[1] == "Hacé la denuncia. Más información en www.bcra.gob.ar"


def test_pagina_larga_con_una_vinieta_va_al_llm():
    raw = "Nota. " * 100 + "\nLlamá a tu banco al 0810-555-1111.\n- Guardá capturas."
    assert parse_guide_template(normalize_unicode(raw), {"doc_id": "G"}) is None


# 3 y 4. R1/R3 ----------------------------------------------------------------------------


def test_r3_cuenta_senales_distintas():
    texto = "Hola ma, cambié de número. Agendame este y transferime hoy"
    senales = [
        SenalDetectada(signal_id="cambio_de_numero", cita="cambié de número", explicacion="x"),
        SenalDetectada(signal_id="cambio_de_numero", cita="Agendame este", explicacion="x"),
    ]
    out = _out(veredicto="estafa_confirmada", modalidad="falso_familiar_cambio_numero", senales=senales)  # fmt: skip
    res, ev = RuleVerifier(["R1", "R2", "R3"]).check_analysis(_m(texto), GUIA, out)
    assert res.veredicto.value == "muy_sospechoso"
    assert ev[-1].detalle["senales_verificadas"] == 1


def test_r1_rechaza_citas_triviales_y_dentro_de_otra_palabra():
    assert cita_score("cambio", "hacemos un intercambio") < 0.9
    assert cita_score("pin", "compré pintura") < 0.9
    senales = [
        SenalDetectada(signal_id="cambio_de_numero", cita="a", explicacion="x"),
        SenalDetectada(signal_id="urgencia_temporal", cita="e", explicacion="x"),
    ]
    res, ev = RuleVerifier(["R1"]).check_analysis(_m("Hola ma, cambié de número"), GUIA, _out(senales=senales))  # fmt: skip
    assert res.senales == [] and [e.accion for e in ev] == ["cita_trivial", "cita_trivial"]


# 5 y 13. R5 ------------------------------------------------------------------------------


def test_r5_acepta_el_organismo_de_la_metadata_de_la_guia():
    plan = [PasoPlan(paso="Si el banco no responde, reclamá ante el BCRA.", guide_id="G-1")]
    res, ev = RuleVerifier(["R5"]).check_plan(GUIA, _out(plan=plan))
    assert res.plan == plan and not ev


def test_r5_limpia_contactos_inventados_en_explicaciones():
    m = _m("Tu cuenta será bloqueada, entrá a bit.ly/abc")
    out = _out(
        senales=[
            SenalDetectada(
                signal_id="amenaza_bloqueo",
                cita="Tu cuenta será bloqueada",
                explicacion="El link bit.ly/abc no es oficial; el oficial es banco-seguro.help",
            )
        ],  # fmt: skip
        casos_similares=[CasoRef(case_id="C-1", por_que="Mismo guion; llamá al 0800-123-4567")],
    )
    res, ev = RuleVerifier(["R5"]).check_plan(GUIA, out, m)
    assert "bit.ly/abc" in res.senales[0].explicacion  # está en el mensaje
    assert "banco-seguro.help" not in res.senales[0].explicacion
    assert CONTACTO_ELIMINADO in res.casos_similares[0].por_que
    assert {e.accion for e in ev} == {"contacto_inventado_en_explicacion"}


# 7 y 8. Normalizador ---------------------------------------------------------------------


def _doc(tipo, texto, **md):
    return Document(page_content=texto, metadata={"doc_tipo": tipo, "doc_id": md.pop("doc_id", "D1"), "source_id": "s", **md})  # fmt: skip


def test_metadata_fuera_del_enum_se_descarta_sin_abortar_el_build():
    docs = [
        _doc("mensaje", "hola", label="estafa", canal="telegram"),
        _doc("mensaje", "chau", doc_id="D2", label="legitimo", canal="sms"),
    ]
    out, rep = normalize_documents(docs, NormalizeCfg(mode="template"))
    assert [d.doc_id for d in out] == ["D2"]
    assert rep.descartados["metadata_invalida"] == 1 and rep.errores[0]["doc_id"] == "D1"


def test_normalizador_redacta_contactos_inventados_en_el_contexto():
    resp = GuiasNorm(
        es_procedimiento=True,
        guias=[
            GuiaNorm(
                modalidades=[],
                situaciones=[],
                organismo=None,
                contexto="Ante dudas llamá al 0800-000-1111.",
                pasos=["Bloqueá la tarjeta."],
            )
        ],  # fmt: skip
    )
    llm = LLMNormalizer(FakeStructuredLLM(lambda m, s: resp), NormalizeCfg())
    (g,), rep = normalize_documents([_doc("guia", "Bloqueá la tarjeta desde la app.")], NormalizeCfg(), llm)  # fmt: skip
    assert CONTACTO_REDACTADO in g.contexto and rep.contactos_redactados == 1


# 9 y 12. Snapshot ------------------------------------------------------------------------


def test_kb_hash_cambia_con_la_metadata():
    a = Document(id="x", page_content="t", metadata={"modalidad": "phishing_bancario"})
    b = Document(id="x", page_content="t", metadata={"modalidad": "deuda_multa_falsa"})
    assert kb_hash([a]) != kb_hash([b])


def test_muestra_de_revision_es_el_10_por_ciento_de_casos_y_guias():
    casos = [CaseDoc(doc_id=f"C{i:02d}", modalidad="phishing_bancario", descripcion="d") for i in range(30)]  # fmt: skip
    rep = NormalizeReport(llm_doc_ids=["C00"])
    sample = review_sample(casos, rep, {}, 0.1, 7)
    assert len(sample) == 3 and "C00" in {r["doc_id"] for r in sample}


# 6. k común y solo retrieval ---------------------------------------------------------------


def test_k_eval_fija_el_k_de_las_metricas_de_casos():
    assert RetrievalCfg(top_k=6).k_casos() == 6
    assert RetrievalCfg(top_n={"casos": 4}).k_casos() == 4
    assert RetrievalCfg(top_k=6, k_eval=4).k_casos() == 4


# 10 y 11. Métricas -----------------------------------------------------------------------


def test_localizacion_iou_contra_citas_de_referencia():
    row = {
        "origen": "captura",
        "texto": "pasame el código",
        "texto_gold": "pasame el código",
        "citas_bbox_gold": [
            {"cita": "pasame el código", "bbox": [10, 10, 100, 30]},
            {"cita": "otra", "bbox": [0, 0, 5, 5]},
        ],  # fmt: skip
        "pred": {
            "senales": [{"cita": "pasame el código", "verificada": True, "bbox": [12, 10, 100, 31]}]
        },  # fmt: skip
    }
    m = compute_extraction_metrics([row])
    assert m["localizacion_iou50"] == 0.5 and m["n_citas_bbox_gold"] == 2


def test_juez_reporta_fallos_y_uso():
    calls = {"n": 0}

    def responder(messages, schema):
        calls["n"] += 1
        return "no es json" if calls["n"] == 1 else PlanJudgeOutput(juicios=[Juicio(paso_clave=1, cubierto=True, razon="")])  # fmt: skip

    rows = [{"id": "a", "pred": {"plan": [{"paso": "x"}]}}, {"id": "b", "pred": {"plan": [{"paso": "y"}]}}]  # fmt: skip
    res = judge_rows(rows, {"a": ["x"], "b": ["y"]}, FakeStructuredLLM(responder), workers=1)
    assert res["n_fallidos"] == 1 and res["n_juzgados"] == 1
    assert res["tokens"]["in"] > 0
    assert [r["cobertura_plan"] for r in rows] == [None, 1.0]
