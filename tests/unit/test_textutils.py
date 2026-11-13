from safefish.textutils import (
    cita_en_texto,
    cita_score,
    find_cbu_alias,
    find_links,
    find_phones,
    normalize_for_match,
    normalize_unicode,
)

MSG = "Hola!! Pasame el código de 6 dígitos que te llegó, es URGENTE 🙏"


def test_normalize_unicode_quita_ancho_cero_y_espacios():
    assert normalize_unicode("a\u200bb   c\r\n\n\n\nd") == "ab c\n\nd"


def test_normalize_for_match_es_laxa():
    assert normalize_for_match("  CÓDIGO   “ya” ") == "codigo ya"


def test_cita_literal_con_tildes_y_mayusculas():
    assert cita_en_texto("pasame el codigo de 6 digitos", MSG)
    assert cita_score("es urgente", MSG) == 1.0


def test_cita_casi_literal_pasa_y_alucinada_no():
    assert cita_en_texto("pasame el código de 6 digitos que te llego", MSG)
    assert not cita_en_texto("transferime 5000 pesos al alias", MSG)


def test_entidades():
    t = "Entrá a www.mp-seguro.com/x o bit.ly/abc, llamá al 11 4567-8901 y depositá al alias foo.bar.baz"
    assert "www.mp-seguro.com/x" in find_links(t)
    assert "bit.ly/abc" in find_links(t)
    assert any("4567-8901" in p for p in find_phones(t))
    assert "foo.bar.baz" in find_cbu_alias(t)


def test_contacto_respaldado_es_estructurado_no_por_substring():
    from safefish.textutils import contacto_respaldado

    guia = "Reclamá en https://www.bcra.gob.ar/reclamos/fraude o al (+54 11) 4348-3500."
    assert contacto_respaldado("bcra.gob.ar", guia)
    assert contacto_respaldado("11 4348-3500", guia)
    assert not contacto_respaldado("gob.ar", guia)  # substring de la prosa, otro host
    assert not contacto_respaldado("bcra.gob.ar/reclamos/fraude-express", guia)
    assert not contacto_respaldado("351 4348-3500", guia)
