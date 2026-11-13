import pytest

from safefish.ingest.pii import redact_pii


@pytest.mark.parametrize(
    ("texto", "esperado", "tipo"),
    [
        ("transferí al CBU 0170099220000067797370 ya", "transferí al CBU <CBU> ya", "CBU"),
        ("mi alias es alias: pepe.gato.luna", "mi alias es alias: <ALIAS>", "ALIAS"),
        ("escribime a juan.perez@gmail.com", "escribime a <EMAIL>", "EMAIL"),
        ("entrá a bit.ly/3xYz9 urgente", "entrá a <LINK_ACORTADO> urgente", "LINK_ACORTADO"),
        ("llamá al +54 9 11 4567-8901", "llamá al <TEL>", "TEL"),
        ("llamá al 011 4567-8901.", "llamá al <TEL>.", "TEL"),
        ("tarjeta 4111 1111 1111 1111", "tarjeta <TARJETA>", "TARJETA"),
        ("DNI 30.123.456", "<DNI>", "DNI"),
        ("tu código es 482913", "tu código es <CODIGO>", "CODIGO"),
        ("Hola Juan, soy Martina Gómez", "Hola <NOMBRE>, soy <NOMBRE>", "NOMBRE"),
    ],
)
def test_redacta_con_placeholder_tipado(texto, esperado, tipo):
    r = redact_pii(texto)
    assert esperado in r.text
    assert r.counts[tipo] >= 1


def test_link_no_acortado_conserva_dominio_y_pierde_ruta():
    r = redact_pii("ingresá a https://banco-seguridad-ar.com/login?u=123")
    assert r.text == "ingresá a https://banco-seguridad-ar.com/<RUTA>"


def test_no_redacta_parentesco_ni_entidades():
    # "Hola ma" y "soy de Mercado Pago" son señal, no PII.
    r = redact_pii("Hola Mamá, soy Mercado Pago")
    assert r.text == "Hola Mamá, soy Mercado Pago"
    assert r.counts["NOMBRE"] == 0


def test_no_confunde_montos_ni_fechas():
    texto = "pagá $ 15.000 antes del 12/11/2026 o 3 cuotas de 5000"
    assert redact_pii(texto).text == texto


def test_numero_que_no_pasa_luhn_no_es_tarjeta():
    assert "<TARJETA>" not in redact_pii("pedido 1234 5678 9012 3456").text
