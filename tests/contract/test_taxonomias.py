"""Paridad entre los Enum de schemas.py y taxonomias.yaml (contrato congelado en F0)."""

from safefish.schemas import Modalidad, Severidad, SignalId
from safefish.taxonomia import load_taxonomia, senales_criticas


def test_modalidades_en_paridad():
    assert set(load_taxonomia()["modalidades"]) == {m.value for m in Modalidad}


def test_senales_en_paridad():
    assert set(load_taxonomia()["senales"]) == {s.value for s in SignalId}


def test_senales_tienen_severidad_y_tipo_validos():
    for sid, s in load_taxonomia()["senales"].items():
        Severidad(s["severidad"])
        assert s["tipo"] in {"riesgo", "legitimidad"}, sid
        assert s["definicion"].strip(), sid


def test_criticas_son_las_del_anexo_c():
    assert senales_criticas() == {
        SignalId.PEDIDO_CODIGO_VERIFICACION,
        SignalId.PEDIDO_CREDENCIALES,
        SignalId.PEDIDO_DATOS_TARJETA,
        SignalId.TRANSFERENCIA_A_TERCERO,
        SignalId.INSTALACION_APP_ACCESO_REMOTO,
    }


def test_contra_senales_no_estan_en_el_enum():
    contra = set(load_taxonomia()["contra_senales"])
    assert contra and not contra & {s.value for s in SignalId}
