"""Contratos Pydantic: el ejemplo del Anexo A, el golden y los configs."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from safefish.config import load_config
from safefish.schemas import GoldenCase, RawInput, SafeFishOutput

REPO = Path(__file__).resolve().parents[2]

ANEXO_A = {
    "veredicto": "estafa_confirmada",
    "modalidad": "robo_cuenta_whatsapp",
    "senales": [
        {
            "signal_id": "pedido_codigo_verificacion",
            "cita": "pasame el código de 6 dígitos que te llegó",
            "explicacion": "Ninguna entidad legítima pide reenviar un código de verificación.",
            "bbox": [112, 640, 590, 688],
            "verificada": True,
        },
        {
            "signal_id": "urgencia_temporal",
            "cita": "es urgente, en 5 minutos se vence",
            "explicacion": "La presión de tiempo busca que no verifiques.",
            "bbox": [112, 700, 470, 742],
            "verificada": True,
        },
    ],
    "casos_similares": [
        {
            "case_id": "C-017",
            "fuente_url": "https://example.org",
            "por_que": "Mismo guion: pide el código de WhatsApp con una excusa.",
        }
    ],
    "plan": [{"paso": "No compartas el código.", "guide_id": "G-004"}],
    "meta": {
        "run_id": "r",
        "config_hash": "h",
        "kb_snapshot": "kb-v2",
        "tokens": {"in": 3120, "out": 410},
        "costo_usd": 0.0,
        "latencias_ms": {"extract": 0, "retrieve": 0, "reason": 0},
    },
}


def test_anexo_a_valida():
    out = SafeFishOutput.model_validate(ANEXO_A)
    assert out.senales[0].bbox == (112, 640, 590, 688)


def test_enums_cerrados_rechazan_valores_inventados():
    bad = {**ANEXO_A, "modalidad": "estafa_nueva_inventada"}
    with pytest.raises(ValidationError):
        SafeFishOutput.model_validate(bad)


def test_raw_input_requiere_texto_o_imagen():
    with pytest.raises(ValidationError):
        RawInput()


def _golden(**kw):
    base = {
        "id": "T-1",
        "split": "test",
        "texto": "hola",
        "label": "legitimo",
        "veredicto_esperado": ["parece_legitimo"],
    }
    return GoldenCase.model_validate({**base, **kw})


def test_golden_estafa_exige_modalidad():
    with pytest.raises(ValidationError):
        _golden(label="estafa", veredicto_esperado=["muy_sospechoso"])


def test_golden_test_rechaza_sinteticos():
    with pytest.raises(ValidationError):
        _golden(fuente="sintetico")
    assert _golden(split="dev", fuente="sintetico").fuente == "sintetico"


@pytest.mark.parametrize("name", ["v0", "v1", "v1-fixture"])
def test_configs_cargan_y_hash_estable(name):
    a = load_config(REPO / "configs" / f"{name}.yaml")
    b = load_config(REPO / "configs" / f"{name}.yaml")
    assert a.config_hash() == b.config_hash()
    assert a.retrieval.filter_canal == "off"


def test_v0_y_v1_difieren_solo_en_retrieval_y_kb():
    v0 = load_config(REPO / "configs" / "v0.yaml")
    v1 = load_config(REPO / "configs" / "v1.yaml")
    assert v0.generation == v1.generation
    assert v0.postprocess == v1.postprocess
    assert not v0.retrieval.enabled and v1.retrieval.enabled
