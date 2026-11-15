"""Cliente LLM: despacho por proveedor y conversión de mensajes a Claude (sin llamar a la API)."""

import pytest

from safefish.llm import _bloques_claude, es_claude, esfuerzo_por_defecto


def test_esfuerzo_por_modelo():
    assert esfuerzo_por_defecto("claude-sonnet-5-5") == "low"
    assert esfuerzo_por_defecto("claude-haiku-5-5") == "low"
    assert esfuerzo_por_defecto("gpt-5-2025-08-07") == "minimal"
    assert esfuerzo_por_defecto("gpt-4.1-mini-2025-04-14") is None  # la clave de caché no cambia
    assert es_claude("claude-haiku-5-5") and not es_claude("gpt-4.1-2025-04-14")


def test_imagen_en_base64_pasa_a_bloque_de_claude_antes_del_texto():
    partes = [
        {"type": "text", "text": "Transcribí la captura."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD", "detail": "high"}},
    ]
    assert _bloques_claude(partes) == [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"}},
        {"type": "text", "text": "Transcribí la captura."},
    ]


def test_imagen_por_url_remota_no_se_acepta():
    with pytest.raises(ValueError):
        _bloques_claude([{"type": "image_url", "image_url": {"url": "https://ejemplo.com/a.png"}}])
