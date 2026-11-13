"""Fixtures compartidas: LLM falso determinístico y rutas de la mini-KB."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from safefish.generation.reasoner import CasoLLM, PasoLLM, ReasonerLLMOutput, SenalLLM
from safefish.llm import FakeStructuredLLM

FIXTURES = Path(__file__).parent / "fixtures"

_MSG = re.compile(r"<mensaje_no_confiable[^>]*>\n(.*?)\n</mensaje_no_confiable>", re.S)
_CASO = re.compile(r'<caso id="([^"]+)" modalidad="([^"]+)"')

# palabra clave → (señal, modalidad sugerida)
_REGLAS = [
    ("código", "pedido_codigo_verificacion", "robo_cuenta_whatsapp"),
    ("alias", "transferencia_a_tercero", "falso_familiar_cambio_numero"),
    ("anydesk", "instalacion_app_acceso_remoto", "falso_soporte_acceso_remoto"),
    ("bloqueo", "amenaza_bloqueo", "phishing_bancario"),
    ("asistente:", "instrucciones_dirigidas_a_ia", None),
]


def _cita(texto: str, kw: str) -> str:
    i = texto.lower().find(kw)
    start = texto.rfind(" ", 0, max(0, i - 15)) + 1
    end = texto.find(" ", i + len(kw) + 15)
    return texto[start : end if end != -1 else len(texto)]


_GUIA = re.compile(r'<guia id="([^"]+)"[^>]*>.*?^1\. ([^\n]+)', re.S | re.M)


def planner_responder(messages, schema):
    """Planner falso: el primer paso de cada guía, redactado igual, con su guide_id."""
    from safefish.generation.planner import PasoPlanLLM, PlanLLMOutput

    human = messages[-1].content
    return PlanLLMOutput(pasos=[PasoPlanLLM(paso=p, guide_id=g) for g, p in _GUIA.findall(human)])


def heuristic_responder(messages, schema):
    """Clasificador por palabras clave con citas literales: estable y sin red. Despacha por
    esquema para que un único LLM falso sirva al razonador y al planner."""
    if schema.__name__ == "PlanLLMOutput":
        return planner_responder(messages, schema)
    human = messages[-1].content if len(messages) == 2 else messages[1].content
    texto = _MSG.search(human).group(1)
    low = texto.lower()
    senales, modalidad = [], "no_determinada"
    for kw, sid, mod in _REGLAS:
        if kw in low and not ("no lo compartas" in low and kw == "código"):
            senales.append(
                SenalLLM(signal_id=sid, cita=_cita(texto, kw), explicacion="regla de prueba")
            )
            modalidad = mod if modalidad == "no_determinada" and mod else modalidad
    casos = [CasoLLM(case_id=cid, por_que="mismo patrón") for cid, mod in _CASO.findall(human) if mod == modalidad][:2]  # fmt: skip
    veredicto = (
        "estafa_confirmada"
        if len(senales) >= 2
        else "muy_sospechoso"
        if senales
        else "parece_legitimo"
    )
    return ReasonerLLMOutput(
        veredicto=veredicto,
        modalidad=modalidad if senales else "no_determinada",
        senales=senales,
        casos_similares=casos,
        plan=[PasoLLM(paso="No respondas y verificá por un canal oficial.", guide_id=None)],
    )


@pytest.fixture
def fake_llm() -> FakeStructuredLLM:
    return FakeStructuredLLM(heuristic_responder)


@pytest.fixture
def tmp_snapshots(tmp_path, monkeypatch) -> Path:
    """Redirige data/kb_snapshots a un directorio temporal."""
    import safefish.config as config

    d = tmp_path / "kb_snapshots"
    monkeypatch.setattr(config, "KB_SNAPSHOTS_DIR", d)
    return d


@pytest.fixture(autouse=True)
def _sin_cache(monkeypatch):
    monkeypatch.setenv("SAFEFISH_NO_CACHE", "1")
