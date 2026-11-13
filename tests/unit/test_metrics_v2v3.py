"""Métricas nuevas de Fase 2: efecto del verificador, plan, extracción y juez del plan."""

from eval.metrics_extraction import cer, compute_extraction_metrics, wer
from eval.metrics_task import contactos_no_respaldados, generacion_metrics
from eval.plan_judge import Juicio, PlanJudgeOutput, coverage_from_judgement, judge_rows
from safefish.llm import FakeStructuredLLM
from safefish.postprocess.verifier import RuleVerifier
from safefish.schemas import EvidenceItem, PasoPlan, RetrievalBundle, RunMeta, SafeFishOutput


def _pred(plan, senales=(), veredicto="muy_sospechoso"):
    return {"veredicto": veredicto, "modalidad": "no_determinada", "senales": list(senales), "plan": plan}  # fmt: skip


def test_contactos_se_miden_contra_la_guia_del_paso_no_contra_el_mensaje():
    row = {
        "texto": "Llamá al 11-5555-1234 urgente",
        "contexto_guias_por_id": {
            "G-1": "Llamá al 0800-111-2222",
            "G-2": "Escribí a ayuda@banco.com",
        },
        "pred": _pred(
            [
                {"paso": "Llamá al 0800-111-2222", "guide_id": "G-1"},  # ok
                {"paso": "Escribí a ayuda@banco.com", "guide_id": "G-1"},  # está en G-2, no en G-1
                {"paso": "Llamá al 11-5555-1234", "guide_id": None},  # sale del mensaje: alucinado
            ]
        ),
    }
    assert contactos_no_respaldados([row]) == 2


def test_generacion_metrics_antes_y_despues_del_verificador():
    texto = "pasame el código"
    s_ok = {"signal_id": "pedido_codigo_verificacion", "cita": "pasame el código", "verificada": True}  # fmt: skip
    s_bad = {"signal_id": "pedido_credenciales", "cita": "dame tu clave", "verificada": False}
    rows = [
        {
            "texto": texto,
            "pred_pre_verify": _pred([], [s_ok, s_bad], "estafa_confirmada"),
            "pred": _pred([{"paso": "No respondas.", "guide_id": "G-1"}], [s_ok]),
            "verificacion": [
                {"regla": "R1", "accion": "cita_no_literal"},
                {"regla": "R3", "accion": "baja_a_muy_sospechoso"},
            ],  # fmt: skip
            "plan_info": {"guias": [{"id": "G-1", "via": "exacta"}], "fallback_extractivo": False},
            "cobertura_plan": 0.5,
        }
    ]
    m = generacion_metrics(rows)
    assert m["grounding_citas_pre_r1"] == 0.5
    assert m["eventos_verificador"] == {"R1:cita_no_literal": 1, "R3:baja_a_muy_sospechoso": 1}
    assert m["veredictos_modificados"] == 1 and m["casos_con_intervencion"] == 1.0
    assert m["pasos_con_guide_id"] == 1.0 and m["planner_via_guias"] == {"exacta": 1}
    assert m["cobertura_plan"] == 0.5


def test_r5_no_cuenta_dos_veces_el_dominio_de_un_mail():
    b = RetrievalBundle(
        guias=[EvidenceItem(id="G-1", coleccion="guias", texto="Bloqueá la tarjeta.")]
    )
    out = SafeFishOutput(
        veredicto="muy_sospechoso",
        modalidad="no_determinada",
        plan=[PasoPlan(paso="Escribí a ayuda@falso-banco.com", guide_id="G-1")],
        meta=RunMeta(run_id="r", config_hash="h"),
    )
    _, ev = RuleVerifier(["R5"]).check_plan(b, out)
    assert len(ev) == 1 and ev[0].detalle["contacto"] == "ayuda@falso-banco.com"


def test_cer_wer_y_entidades():
    assert cer("hola", "hola") == 0.0 and wer("hola che", "hola") == 1.0
    rows = [
        {"origen": "captura", "texto": "Entrá a bit.ly/x ya", "texto_gold": "Entrá a bit.ly/x ya!",
         "pred": {"senales": [{"verificada": True, "bbox": [1, 2, 3, 4]}, {"verificada": True, "bbox": None}]}},
        {"origen": "texto", "texto": "x", "texto_gold": "x", "pred": None},
    ]  # fmt: skip
    m = compute_extraction_metrics(rows)
    assert m["n_capturas"] == 1 and m["wer"] == 0.0 and m["f1_entidades"] == 1.0
    assert m["senales_con_bbox"] == 0.5


def test_juez_del_plan():
    out = PlanJudgeOutput(
        juicios=[
            Juicio(paso_clave=1, cubierto=True, razon=""),
            Juicio(paso_clave=2, cubierto=False, razon=""),
            Juicio(paso_clave=9, cubierto=True, razon="fuera de rango"),
        ]
    )
    assert coverage_from_judgement(out, 2) == 0.5
    rows = [
        {"id": "a", "pred": _pred([{"paso": "Bloqueá la tarjeta", "guide_id": "G"}])},
        {"id": "b", "pred": None},
        {"id": "c", "pred": _pred([])},
    ]
    judge_rows(rows, {"a": ["bloquear tarjeta", "denunciar"], "b": ["x"]}, FakeStructuredLLM(lambda m, s: out))  # fmt: skip
    assert [r["cobertura_plan"] for r in rows] == [0.5, 0.0, None]
