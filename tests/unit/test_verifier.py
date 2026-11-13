"""Verificador R1–R5 (M-POST.1, §2.3). Invariante del gate G2: grounding de citas = 100 %
después de R1."""

import json

import pytest

from safefish.extract.text import TextExtractor
from safefish.postprocess.verifier import CONTACTO_ELIMINADO, RuleVerifier
from safefish.schemas import (
    CasoRef,
    EvidenceItem,
    OcrWord,
    PasoPlan,
    RawInput,
    RetrievalBundle,
    RunMeta,
    SafeFishOutput,
    SenalDetectada,
)
from safefish.textutils import cita_en_texto

TEXTO = (
    "Hola! Te mandé un código de 6 dígitos por error, pasame el código porfa. "
    "Es urgente, en 5 minutos se vence."
)
GUIA_TXT = (
    "Si compartiste el código:\nPasos:\n1. Volvé a registrar WhatsApp con tu número.\n"
    "2. Activá la verificación en dos pasos.\n3. Denunciá en la UFECI: (+54 11) 5071-0040."
)


def _m(texto=TEXTO, ocr=None):
    m = TextExtractor().extract(RawInput(texto=texto))
    return m.model_copy(update={"ocr_words": ocr, "origen": "captura" if ocr else "texto"})


def _bundle(modalidades=("robo_cuenta_whatsapp",), guias=True):
    casos = [
        EvidenceItem(
            id=f"C-{i}",
            coleccion="casos",
            texto="...",
            metadata={"doc_id": f"C-{i}", "modalidad": mod},
        )  # fmt: skip
        for i, mod in enumerate(modalidades)
    ]
    g = [
        EvidenceItem(
            id="G-1",
            coleccion="guias",
            texto=GUIA_TXT,
            metadata={"doc_id": "G-1", "pasos_json": json.dumps(["Volvé a registrar WhatsApp con tu número."])},
        )
    ]  # fmt: skip
    return RetrievalBundle(casos=casos, guias=g if guias else [])


def _out(veredicto="estafa_confirmada", modalidad="robo_cuenta_whatsapp", senales=None, plan=None, casos=None):  # fmt: skip
    return SafeFishOutput(
        veredicto=veredicto,
        modalidad=modalidad,
        senales=senales
        if senales is not None
        else [
            SenalDetectada(signal_id="pedido_codigo_verificacion", cita="pasame el código porfa", explicacion="x"),
            SenalDetectada(signal_id="urgencia_temporal", cita="Es urgente, en 5 minutos se vence", explicacion="x"),
        ],
        casos_similares=casos or [CasoRef(case_id="C-0", por_que="igual")],
        plan=plan or [PasoPlan(paso="Volvé a registrar WhatsApp con tu número.", guide_id="G-1")],
        meta=RunMeta(run_id="r", config_hash="h"),
    )  # fmt: skip


def _acciones(ev):
    return [(e.regla, e.accion) for e in ev]


# --- R1 -----------------------------------------------------------------------------------


def test_r1_descarta_cita_alucinada_y_marca_verificadas():
    out = _out(
        senales=[
            SenalDetectada(signal_id="pedido_codigo_verificacion", cita="pasame el codigo porfa", explicacion="x"),  # sin tilde: ok
            SenalDetectada(signal_id="pedido_credenciales", cita="ingresá tu clave del banco", explicacion="x"),  # inventada
        ]
    )  # fmt: skip
    res, ev = RuleVerifier(["R1"]).check_analysis(_m(), _bundle(), out)
    assert [s.signal_id.value for s in res.senales] == ["pedido_codigo_verificacion"]
    assert res.senales[0].verificada
    assert ("R1", "cita_no_literal") in _acciones(ev)


def test_r1_invariante_grounding_100():
    citas = ["pasame el código", "Te mandé un código de 6 dígitos", "dame tu CBU", "en 5 minutos se vence", "esto no está"]  # fmt: skip
    out = _out(senales=[SenalDetectada(signal_id="urgencia_temporal", cita=c, explicacion="x") for c in citas])  # fmt: skip
    res, _ = RuleVerifier(["R1"]).check_analysis(_m(), _bundle(), out)
    assert res.senales and all(cita_en_texto(s.cita, TEXTO) for s in res.senales)


def test_r1_deduplica_y_alinea_bbox_en_capturas():
    ocr = [OcrWord(text=w, bbox=(10 * i, 5, 10 * i + 9, 15)) for i, w in enumerate(TEXTO.split())]
    dup = SenalDetectada(signal_id="pedido_codigo_verificacion", cita="pasame el código porfa", explicacion="x")  # fmt: skip
    res, _ = RuleVerifier(["R1"]).check_analysis(_m(ocr=ocr), _bundle(), _out(senales=[dup, dup]))
    assert len(res.senales) == 1
    assert res.senales[0].bbox is not None


# --- R2 / R3 / R4 -------------------------------------------------------------------------


def test_r2_modalidad_sin_caso_pasa_a_no_determinada_y_filtra_casos_inexistentes():
    out = _out(casos=[CasoRef(case_id="C-0", por_que="a"), CasoRef(case_id="C-999", por_que="b")])
    res, ev = RuleVerifier(["R2"]).check_analysis(_m(), _bundle(("phishing_bancario",)), out)
    assert res.modalidad.value == "no_determinada"
    assert [c.case_id for c in res.casos_similares] == ["C-0"]
    assert {"modalidad_sin_respaldo", "caso_similar_inexistente"} <= {a for _, a in _acciones(ev)}


def test_r3_estafa_confirmada_sin_respaldo_baja_a_muy_sospechoso():
    res, ev = RuleVerifier(["R1", "R2", "R3"]).check_analysis(_m(), _bundle(("phishing_bancario",)), _out())  # fmt: skip
    assert res.veredicto.value == "muy_sospechoso"
    assert ("R3", "baja_a_muy_sospechoso") in _acciones(ev)


def test_r3_respeta_estafa_confirmada_con_respaldo_y_dos_senales():
    res, ev = RuleVerifier().check_analysis(_m(), _bundle(), _out())
    assert res.veredicto.value == "estafa_confirmada" and not ev


def test_r3_una_sola_senal_verificada_no_alcanza():
    out = _out(senales=[SenalDetectada(signal_id="pedido_codigo_verificacion", cita="pasame el código porfa", explicacion="x")])  # fmt: skip
    res, _ = RuleVerifier(["R1", "R2", "R3"]).check_analysis(_m(), _bundle(), out)
    assert res.veredicto.value == "muy_sospechoso"


def test_r4_parece_legitimo_con_senal_critica_sube():
    out = _out(veredicto="parece_legitimo", modalidad="no_determinada")
    res, ev = RuleVerifier(["R1", "R4"]).check_analysis(_m(), _bundle(), out)
    assert res.veredicto.value == "levemente_sospechoso"
    assert ev[-1].detalle["senales_criticas"] == ["pedido_codigo_verificacion"]


def test_r4_transferencia_sin_urgencia_no_es_critica():
    texto = "Hola! Para el regalo de Juan transferí a este alias: regalo.juan.mp"
    out = _out(
        veredicto="parece_legitimo",
        modalidad="no_determinada",
        senales=[SenalDetectada(signal_id="transferencia_a_tercero", cita="transferí a este alias", explicacion="x")],
    )  # fmt: skip
    res, ev = RuleVerifier(["R1", "R4"]).check_analysis(_m(texto), _bundle(), out)
    assert res.veredicto.value == "parece_legitimo" and not ev


def test_reglas_idempotentes():
    v = RuleVerifier()
    once = v.verify(_m(), _bundle(("phishing_bancario",)), _out())
    twice = v.verify(_m(), _bundle(("phishing_bancario",)), once)
    assert once == twice


# --- R5 -----------------------------------------------------------------------------------


def test_r5_reemplaza_contactos_que_no_estan_en_la_guia():
    plan = [
        PasoPlan(
            paso="Denunciá en la UFECI: (+54 11) 5071-0040.", guide_id="G-1"
        ),  # está en la guía
        PasoPlan(paso="Llamá al 0800-999-1234 o entrá a www.recupero-wa.com", guide_id="G-1"),
    ]
    res, ev = RuleVerifier(["R5"]).check_plan(_bundle(), _out(plan=plan))
    assert res.plan[0].paso == plan[0].paso
    assert "0800-999-1234" not in res.plan[1].paso and "recupero-wa" not in res.plan[1].paso
    assert CONTACTO_ELIMINADO in res.plan[1].paso
    assert _acciones(ev).count(("R5", "contacto_inventado")) == 2


def test_r5_elimina_paso_con_organismo_inventado_y_guide_id_inexistente():
    plan = [
        PasoPlan(paso="Hacé el reclamo ante el BCRA.", guide_id="G-1"),  # el BCRA no está en G-1
        PasoPlan(paso="Activá la verificación en dos pasos.", guide_id="G-404"),
    ]
    res, ev = RuleVerifier(["R5"]).check_plan(_bundle(), _out(plan=plan))
    assert [p.paso for p in res.plan] == ["Activá la verificación en dos pasos."]
    assert res.plan[0].guide_id is None
    assert {"paso_con_organismo_inventado", "guide_id_inexistente"} <= {a for _, a in _acciones(ev)}


def test_r5_si_no_queda_plan_usa_los_pasos_literales_de_la_guia():
    plan = [PasoPlan(paso="Presentá un reclamo en ANSES.", guide_id="G-1")]
    res, ev = RuleVerifier(["R5"]).check_plan(_bundle(), _out(plan=plan))
    assert res.plan == [PasoPlan(paso="Volvé a registrar WhatsApp con tu número.", guide_id="G-1")]
    assert ("R5", "plan_reemplazado_por_guia") in _acciones(ev)


def test_reglas_desconocidas_fallan():
    with pytest.raises(ValueError):
        RuleVerifier(["R9"])


def test_r5_no_acepta_contactos_por_substring_de_la_guia():
    # Comparación estructurada: teléfonos por dígitos (tolera el prefijo +54), URLs por host.
    plan = [
        PasoPlan(paso="Llamá al 11 5071-0040.", guide_id="G-1"),  # mismo número sin +54: ok
        PasoPlan(paso="Llamá al 351 5071-0040.", guide_id="G-1"),  # otra característica
        PasoPlan(paso="Entrá a whatsapp-recupero.com/ayuda", guide_id="G-1"),
    ]
    res, ev = RuleVerifier(["R5"]).check_plan(_bundle(), _out(plan=plan))
    assert res.plan[0].paso == plan[0].paso
    assert CONTACTO_ELIMINADO in res.plan[1].paso and CONTACTO_ELIMINADO in res.plan[2].paso
    assert _acciones(ev).count(("R5", "contacto_inventado")) == 2
