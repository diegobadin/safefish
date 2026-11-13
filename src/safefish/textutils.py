"""Utilidades de texto compartidas: normalización Unicode, patrones de entidades y citas.

Lo usan la ingesta (limpieza, PII), la extracción (entidades) y la evaluación/verificador
(grounding de citas, regla R1). Mantenerlas en un solo lugar evita que la KB y el
verificador normalicen distinto.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

# ---------------------------------------------------------------------------
# Normalización
# ---------------------------------------------------------------------------

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)
_SPACES = re.compile(r"[ \t  -   　]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")


def normalize_unicode(text: str) -> str:
    """Normalización canónica (KB y entrada): NFKC, sin caracteres de ancho cero,
    espacios colapsados y como mucho una línea en blanco seguida."""
    text = unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(_SPACES.sub(" ", line).strip() for line in text.split("\n"))
    return _MANY_NEWLINES.sub("\n\n", text).strip()


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


_NON_WORD = re.compile(r"[^\w$%<>@./:-]+")


def normalize_for_match(text: str) -> str:
    """Forma laxa para comparar citas (R1): minúsculas, sin tildes, sin comillas ni
    puntuación suelta y con espacios colapsados."""
    text = strip_accents(normalize_unicode(text).lower())
    return " ".join(_NON_WORD.sub(" ", text).split())


def cita_score(cita: str, texto: str) -> float:
    """Similitud [0, 1] entre la cita y la mejor ventana del texto (sobre formas normalizadas).

    1.0 si la cita es subcadena exacta tras normalizar; si no, el mejor `ratio` de difflib
    sobre ventanas de palabras de largo parecido al de la cita.
    """
    c, t = normalize_for_match(cita), normalize_for_match(texto)
    if not c:
        return 0.0
    if c in t:
        return 1.0
    cw, tw = c.split(), t.split()
    n = len(cw)
    best = 0.0
    for size in {max(1, n - 1), n, n + 1}:
        for i in range(0, max(1, len(tw) - size + 1)):
            window = " ".join(tw[i : i + size])
            best = max(best, SequenceMatcher(None, c, window, autojunk=False).ratio())
            if best >= 0.999:
                return best
    return best


def cita_en_texto(cita: str, texto: str, threshold: float = 0.9) -> bool:
    """Regla R1: la cita aparece (de forma literal o casi literal) en el texto."""
    return cita_score(cita, texto) >= threshold


# ---------------------------------------------------------------------------
# Patrones (Argentina). Orden de aplicación importa: de más a menos específico.
# ---------------------------------------------------------------------------

SHORTENER_DOMAINS = (
    "bit.ly", "bitly.com", "tinyurl.com", "acortar.link", "cutt.ly", "t.co", "goo.gl",
    "is.gd", "ow.ly", "rb.gy", "shorturl.at", "s.id", "tiny.cc", "rebrand.ly", "wa.link",
    "qrco.de", "t.ly", "short.gy", "linktr.ee",
)  # fmt: skip

_TLDS = r"(?:com|net|org|ar|info|xyz|online|site|top|ly|me|io|co|app|link|shop|click|lat|live|store|cc|gl|gd|at|de|id|gy|ee|sh)"
RE_URL = re.compile(
    rf"""(?ix)
    (?:https?://|www\.)[^\s<>"'()]+            # con esquema o www
    | \b(?:[a-z0-9](?:[a-z0-9-]{{0,61}}[a-z0-9])?\.)+{_TLDS}(?:\.ar)?\b(?:/[^\s<>"'()]*)?   # dominio suelto
    """
)
RE_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
RE_CBU = re.compile(r"(?<![\d.])(?:\d[ -]?){21}\d(?![\d.])")  # CBU/CVU: 22 dígitos
RE_CUIT = re.compile(r"\b(?:20|23|24|27|30|33|34)-?\d{8}-?\d\b")
RE_CARD = re.compile(r"(?<![\d.])(?:\d[ -]?){12,18}\d(?![\d.])")
RE_DNI = re.compile(
    r"(?i)\b(?:dni|documento)\s*(?:n[°ºo.]?\s*)?:?\s*\d{1,2}\.?\d{3}\.?\d{3}\b|\b\d{1,2}\.\d{3}\.\d{3}\b"
)
RE_PHONE = re.compile(
    r"""(?x)
    (?<![\w$.])
    (?:\+?54[\s-]?(?:9[\s-]?)?)?          # prefijo país (+54 9)
    (?:\(?0?\d{2,4}\)?[\s-]?)?            # característica (011, 0351, (11))
    (?:15[\s-]?)?                         # prefijo celular
    \d{3,4}[\s-]?\d{4}
    (?!\w)(?!\.\d)                       # no seguido de dígitos (montos)
    """
)
RE_0800 = re.compile(r"\b0[86]00[\s-]?\d{3}[\s-]?\d{4}\b")
RE_CODE = re.compile(r"(?i)\b(c[oó]digo|token|clave|pin)\b([^\d\n]{0,30}?)((?:\d[\s-]?){3,7}\d)\b")
RE_ALIAS = re.compile(
    r"(?i)\b(alias(?:\s+(?:cbu|cvu))?\s*[:\-]?\s*)([a-z0-9]+(?:\.[a-z0-9]+){1,3})\b"
)
RE_MONTO = re.compile(
    r"(?i)(?:\$|u\$s|usd|ars)\s?\d{1,3}(?:[.\s]\d{3})*(?:,\d{1,2})?|\b\d{1,3}(?:\.\d{3})+(?:,\d{1,2})?\s?(?:pesos|d[oó]lares)\b"
)

_NAME_TRIGGER = (
    r"(?i:hola|soy|sr\.?|sra\.?|señor|señora|estimad[oa]|querid[oa]|a nombre de|titular:?)"
)
_CAP = r"[A-ZÁÉÍÓÚÑ][a-záéíóúñü]+"
RE_NOMBRE = re.compile(rf"\b({_NAME_TRIGGER}\s+)({_CAP}(?:\s+{_CAP})?)")
# Palabras capitalizadas tras "hola/soy" que NO son nombres de persona (son señal o entidad).
NO_NOMBRES = {
    "mamá", "mama", "ma", "papá", "papa", "pa", "abuela", "abuelo", "tía", "tia", "tío", "tio",
    "hijo", "hija", "amigo", "amiga", "cliente", "usuario", "usuaria", "vecino", "vecina",
    "banco", "mercado", "pago", "correo", "andreani", "oca", "afip", "arca", "anses",
    "galicia", "santander", "bbva", "macro", "naranja", "ualá", "uala", "brubank", "nación",
    "nacion", "provincia", "whatsapp", "netflix", "movistar", "claro", "personal", "edenor",
    "edesur", "metrogas", "soporte", "atención", "atencion", "equipo", "buenas", "buen",
    "buenos", "como", "cómo", "que", "qué", "te", "tu",
}  # fmt: skip


def luhn_ok(digits: str) -> bool:
    ds = [int(d) for d in re.sub(r"\D", "", digits)]
    if len(ds) < 13:
        return False
    total = 0
    for i, d in enumerate(reversed(ds)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def is_shortener(url: str) -> bool:
    host = re.sub(r"(?i)^(?:https?://)?(?:www\.)?", "", url).split("/")[0].lower()
    return host in SHORTENER_DOMAINS


def domain_of(url: str) -> str:
    return re.sub(r"(?i)^(?:https?://)?(?:www\.)?", "", url).split("/")[0].lower()


def find_links(text: str) -> list[str]:
    return [m.group(0).rstrip(".,;:!?") for m in RE_URL.finditer(text) if "@" not in m.group(0)]


def find_phones(text: str) -> list[str]:
    out = [m.group(0) for m in RE_0800.finditer(text)]
    for m in RE_PHONE.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if 8 <= len(digits) <= 13 and m.group(0) not in out:
            out.append(m.group(0).strip())
    return out


def find_cbu_alias(text: str) -> list[str]:
    return [m.group(0) for m in RE_CBU.finditer(text)] + [
        m.group(2) for m in RE_ALIAS.finditer(text)
    ]


# ---------------------------------------------------------------------------
# Organismos (regla R5): un nombre de organismo en el plan tiene que estar en la guía fuente.
# Cada entrada agrupa alias equivalentes; matchea sobre la forma normalizada (sin tildes).
# ---------------------------------------------------------------------------

ORGANISMOS: dict[str, tuple[str, ...]] = {
    "bcra": ("bcra", "banco central"),
    "anses": ("anses",),
    "afip": ("afip",),
    "arca": ("arca", "agencia de recaudacion y control aduanero"),
    "enacom": ("enacom",),
    "cnv": ("cnv", "comision nacional de valores"),
    "ufeci": ("ufeci", "unidad fiscal especializada en ciberdelincuencia"),
    "mpf": ("ministerio publico fiscal", "mpf"),
    "pfa": ("policia federal", "pfa"),
    "policia_ciudad": ("policia de la ciudad",),
    "defensa_consumidor": ("defensa del consumidor", "defensa de las y los consumidores"),
    "cert": ("cert.ar", "certar", "icic", "direccion nacional de ciberseguridad"),
    "aaip": (
        "aaip",
        "agencia de acceso a la informacion publica",
        "proteccion de datos personales",
    ),
    "red_link": ("red link",),
    "banelco": ("banelco",),
}


def find_organismos(text: str) -> set[str]:
    """Claves de `ORGANISMOS` mencionadas en el texto (por cualquiera de sus alias)."""
    t = re.sub(r"[.:,;/-]+(?=\s|$)", "", f" {normalize_for_match(text)} ") + " "
    return {
        key
        for key, aliases in ORGANISMOS.items()
        if any(f" {normalize_for_match(a)} " in t for a in aliases)
    }


def find_contacts(text: str) -> list[str]:
    """Teléfonos, mails y links del texto, sin contar dos veces el dominio de un mail."""
    emails = RE_EMAIL.findall(text)
    links = [u for u in find_links(text) if not any(u in e for e in emails)]
    return list(dict.fromkeys([*find_phones(text), *emails, *links]))
