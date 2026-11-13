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
    # Subcadena exacta, pero respetando límites de palabra: "cambio" no está en "intercambio".
    if re.search(rf"(?<!\w){re.escape(c)}(?!\w)", t):
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


# Detector **amplio** para los controles de contactos (R5, normalizador, métrica): cualquier
# TLD, teléfonos con puntos y números cortos de servicio. Es más agresivo que `find_links` /
# `find_phones` (que usan la PII y las entidades) a propósito: un falso positivo acá solo
# hace que se redacte de más; un falso negativo deja pasar un contacto inventado.
RE_URL_LAX = re.compile(
    r"""(?ix)
    (?<![@\w.-])
    (?:https?://)?(?:www\.)?
    (?:[^\W_](?:[\w-]{0,61}[^\W_])?\.)+[^\W\d_]{2,24}   # etiquetas y TLD Unicode (IDN)
    (?![\w-])
    (?:/[^\s<>"'()]*)?
    """
)
RE_URL_SCHEME = re.compile(r"(?i)\b(?:https?|hxxps?)://[^\s<>\"'()]+")  # cualquier cosa con esquema
RE_IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?::\d{1,5})?(?:/[^\s<>\"'()]*)?(?![\d.])")
RE_PHONE_DOTS = re.compile(
    r"(?<![\d.])(?:\+?54[ .]?(?:9[ .]?)?)?(?:\(?0?\d{2,4}\)?[ .])?\d{2,4}\.\d{3,4}\.\d{4}(?![\d.])"
)  # el último grupo de 4 dígitos excluye montos (grupos de 3)
RE_SHORT_CODE = re.compile(
    r"(?i)(?<![\w*])\*\d{2,5}(?!\d)|(?<!\w)(?:al|l[ií]nea|n[uú]mero|marc[aá]\w*)\s+(\d{3,4})(?![.,\s-]?\d)"
)


def find_contacts(text: str) -> list[str]:
    """Teléfonos, mails y links del texto (detector amplio), sin contar dos veces el dominio
    de un mail."""
    emails = RE_EMAIL.findall(text)
    links = [
        m.group(0).rstrip(".,;:!?")
        for rx in (RE_URL_SCHEME, RE_URL_LAX, RE_IPV4)
        for m in rx.finditer(text)
    ] + find_links(text)
    links = [u for u in links if u and not any(u in e for e in emails)]
    phones = find_phones(text) + [m.group(0) for m in RE_PHONE_DOTS.finditer(text)]
    phones = [p for p in phones if not any(p in u for u in links)]  # dígitos de una IP/URL
    shorts = [m.group(1) or m.group(0) for m in RE_SHORT_CODE.finditer(text)]
    found = list(dict.fromkeys([*phones, *shorts, *emails, *links]))
    # Un contacto contenido en otro más largo (bit.ly/x dentro de https://bit.ly/x) no se
    # cuenta aparte: R5 lo reemplaza junto con el largo y la métrica no lo duplica.
    return [c for c in found if not any(c != o and c in o for o in found)]


# ---------------------------------------------------------------------------
# Delimitadores de prompt: el texto no confiable (mensaje, fuentes, KB) nunca debe poder
# abrir o cerrar un tag del prompt. Se reemplazan los ángulos (no se borran: borrar en una
# pasada permite reconstruir un tag anidado) y las comillas de los atributos.
# ---------------------------------------------------------------------------


def escape_tags(text: str) -> str:
    return str(text).replace("<", "‹").replace(">", "›")


def escape_attr(value: object) -> str:
    return escape_tags(str(value)).replace('"', "'").replace("\n", " ")


# ---------------------------------------------------------------------------
# Contactos respaldados (R5, normalizador, métrica de contactos alucinados): comparación
# estructurada contra los contactos extraídos de la fuente, no por substring de la prosa.
# ---------------------------------------------------------------------------

_PHONE_ONLY = re.compile(r"[\d\s()+.*#-]+")


def _url_parts(u: str) -> tuple[str, str]:
    u = re.sub(r"(?i)^(?:https?://)?(?:www\.)?", "", u.strip()).rstrip("/.,;:")
    host, _, path = u.partition("/")
    return host.lower(), path


def canonical_phone(text: str) -> str:
    """Dígitos de un teléfono argentino sin los prefijos que no cambian el destino:
    internacional (`00`), país (`54`), el `9` de celular tras el `54` y el `0` troncal.
    La característica de área se conserva: `351 4348-3500` y `4348-3500` son números
    distintos."""
    d = re.sub(r"\D", "", text)
    if d.startswith("00"):
        d = d[2:]
    if d.startswith("54") and len(d) >= 12:
        d = d[2:]
        if d.startswith("9") and len(d) == 11:
            d = d[1:]
    if d.startswith("0"):
        d = d[1:]
    return d


def contacto_respaldado(c: str, fuente: str) -> bool:
    """`c` (teléfono, mail o URL) figura en `fuente`:

    - mail: igualdad exacta (sin distinguir mayúsculas);
    - teléfono: igualdad exacta de `canonical_phone` (solo se ignoran prefijos de país y
      troncales, nunca una característica de área agregada u omitida);
    - URL: mismo host y la ruta de `c` es prefijo de la de la fuente (acortar está bien,
      agregar una ruta no).
    """
    contactos = find_contacts(fuente)
    if "@" in c:
        return c.lower() in {x.lower() for x in contactos if "@" in x}
    if _PHONE_ONLY.fullmatch(c):
        d = canonical_phone(c)
        return any(canonical_phone(x) == d for x in contactos if _PHONE_ONLY.fullmatch(x))
    host, path = _url_parts(c)
    for x in contactos:
        if "@" in x or _PHONE_ONLY.fullmatch(x):
            continue
        h, p = _url_parts(x)
        if h == host and (not path or p == path or p.startswith(path.rstrip("/") + "/")):
            return True
    return False
