"""Redacción de PII con placeholders tipados (§1.4, §5.5).

Los placeholders preservan la señal ("CBU de un tercero", "link acortado") sin persistir el
dato. Los links no acortados conservan el dominio (es evidencia: `banco-seguridad.com`) pero
pierden la ruta, que puede contener tokens o identificadores personales.

Limitación conocida: la detección de nombres es heurística (disparadores como "hola", "soy",
"a nombre de" + palabra capitalizada). No reemplaza un NER; se documenta en la data card.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from safefish.textutils import (
    NO_NOMBRES,
    RE_0800,
    RE_ALIAS,
    RE_CARD,
    RE_CBU,
    RE_CODE,
    RE_CUIT,
    RE_DNI,
    RE_EMAIL,
    RE_NOMBRE,
    RE_PHONE,
    RE_URL,
    is_shortener,
    luhn_ok,
)


@dataclass
class RedactionResult:
    text: str
    counts: Counter = field(default_factory=Counter)


def _redact_url(m: re.Match, counts: Counter) -> str:
    url = m.group(0)
    trail = ""
    while url and url[-1] in ".,;:!?":
        trail, url = url[-1] + trail, url[:-1]
    if is_shortener(url):
        counts["LINK_ACORTADO"] += 1
        return "<LINK_ACORTADO>" + trail
    prefix = re.match(r"(?i)^(?:https?://)?(?:www\.)?", url).group(0)
    rest = url[len(prefix) :]
    host, sep, path = rest.partition("/")
    if sep and path:
        counts["RUTA"] += 1
        return f"{prefix}{host}/<RUTA>{trail}"
    return url + trail


def _redact_phone(m: re.Match, counts: Counter) -> str:
    digits = re.sub(r"\D", "", m.group(0))
    if not 8 <= len(digits) <= 13:
        return m.group(0)
    counts["TEL"] += 1
    return "<TEL>"


def _redact_card(m: re.Match, counts: Counter) -> str:
    if not luhn_ok(m.group(0)):
        return m.group(0)
    counts["TARJETA"] += 1
    return "<TARJETA>"


def _redact_name(m: re.Match, counts: Counter) -> str:
    trigger, name = m.group(1), m.group(2)
    first = name.split()[0].lower()
    if first in NO_NOMBRES:
        return m.group(0)
    counts["NOMBRE"] += 1
    return f"{trigger}<NOMBRE>"


def redact_pii(text: str) -> RedactionResult:
    counts: Counter = Counter()

    def sub(pattern: re.Pattern, label: str, s: str) -> str:
        def repl(_m: re.Match) -> str:
            counts[label] += 1
            return f"<{label}>"

        return pattern.sub(repl, s)

    text = sub(RE_EMAIL, "EMAIL", text)
    text = RE_URL.sub(lambda m: _redact_url(m, counts), text)
    text = RE_CODE.sub(
        lambda m: (counts.update(["CODIGO"]), f"{m.group(1)}{m.group(2)}<CODIGO>")[1], text
    )
    text = RE_ALIAS.sub(lambda m: (counts.update(["ALIAS"]), f"{m.group(1)}<ALIAS>")[1], text)
    text = sub(RE_CBU, "CBU", text)
    text = sub(RE_CUIT, "CUIT", text)
    text = RE_CARD.sub(lambda m: _redact_card(m, counts), text)
    text = sub(RE_DNI, "DNI", text)
    text = sub(RE_0800, "TEL", text)
    text = RE_PHONE.sub(lambda m: _redact_phone(m, counts), text)
    text = RE_NOMBRE.sub(lambda m: _redact_name(m, counts), text)
    return RedactionResult(text=text, counts=counts)
