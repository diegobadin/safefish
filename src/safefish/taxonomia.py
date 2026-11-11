"""Acceso a `taxonomias.yaml` (definiciones, severidades) para prompts, KB y reglas."""

from __future__ import annotations

from functools import cache
from pathlib import Path

import yaml

from safefish.schemas import Modalidad, Severidad, SignalId

TAXONOMIA_PATH = Path(__file__).with_name("taxonomias.yaml")


@cache
def load_taxonomia(path: Path = TAXONOMIA_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def severidad(signal_id: SignalId | str) -> Severidad:
    return Severidad(load_taxonomia()["senales"][str(signal_id)]["severidad"])


def senales_criticas() -> frozenset[SignalId]:
    return frozenset(s for s in SignalId if severidad(s) is Severidad.CRITICA)


def descripcion_modalidad(m: Modalidad | str) -> str:
    return load_taxonomia()["modalidades"][str(m)]["descripcion"]


def render_modalidades() -> str:
    """Lista `- id: descripción` para el system prompt."""
    tax = load_taxonomia()["modalidades"]
    return "\n".join(f"- {k}: {v['descripcion']}" for k, v in tax.items())


def render_senales() -> str:
    """Lista `- id (severidad): definición` para el system prompt."""
    tax = load_taxonomia()["senales"]
    return "\n".join(f"- {k} ({v['severidad']}): {v['definicion']}" for k, v in tax.items())


def senal_como_documento(signal_id: str) -> str:
    """Texto que se embebe para la colección `senales` (§1.4): nombre + definición + ejemplos."""
    s = load_taxonomia()["senales"][signal_id]
    partes = [f"Señal: {signal_id.replace('_', ' ')}", f"Definición: {s['definicion']}"]
    if s.get("ejemplos"):
        partes.append("Ejemplos: " + " | ".join(s["ejemplos"]))
    if s.get("contraejemplos"):
        partes.append("Contraejemplos (no es la señal): " + " | ".join(s["contraejemplos"]))
    return "\n".join(partes)
