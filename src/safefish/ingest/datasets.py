"""Datasets de mensajes → un CSV común para el loader `csv` (colección `mensajes`).

Cada conversor lee lo que dejó `safefish.ingest.fetch` en `data/raw/datasets/<id>/` y escribe
`data/interim/mensajes/<id>.csv` con las columnas de `COLUMNAS`.

Etiquetas: `label` es `estafa` solo cuando el dataset marca phishing/smishing/fraude. El "spam"
genérico (publicidad, cadenas) **no** es una estafa: queda con `label` vacío y el build lo
descarta (un mensaje sin etiqueta no aporta al voto kNN) hasta que se re-etiquete. La etiqueta
original queda en `label_origen`.

Golden: de SpaPhish (emails reales en español nativo) se reserva una fracción fija por hash como
candidatos al golden de email (`data/interim/golden_candidatos/spaphish.csv`); esos mensajes no
van a la KB (evita leakage KB ↔ test, §4.5).
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

COLUMNAS = ["id", "texto", "label", "label_origen", "canal", "idioma", "fuente_pais", "fecha", "dataset"]  # fmt: skip
MAX_CHARS = 6000  # emails larguísimos: el embedding no necesita más
GOLDEN_FRACCION = 10  # 1 de cada N mensajes de SpaPhish va a candidatos del golden

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


@dataclass
class Fila:
    id: str
    texto: str
    label: str  # "estafa" | "legitimo" | "" (sin etiqueta utilizable)
    label_origen: str
    canal: str
    idioma: str
    fuente_pais: str = ""
    fecha: str = ""
    dataset: str = ""


def _texto(*partes: str | None) -> str:
    t = "\n\n".join(p.strip() for p in partes if p and p.strip())
    return t[:MAX_CHARS]


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _dict_rows(path: Path, delimiter: str) -> Iterator[dict]:
    yield from csv.DictReader(_read_text(path).splitlines(keepends=True), delimiter=delimiter)


def _asunto(s: str | None) -> str | None:
    return f"Asunto: {s.strip()}" if s and s.strip() else None


def _hash_id(*partes: str) -> str:
    return hashlib.sha256("\x1f".join(partes).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Conversores
# ---------------------------------------------------------------------------


def spaphish(d: Path) -> Iterator[Fila]:
    for r in _dict_rows(d / "Spaphish dataset - DiB.csv", ","):
        lab = str(r.get("Label", "")).strip()
        yield Fila(
            id=f"spaphish-{r['hash'][:16]}",
            texto=_texto(_asunto(r.get("subject")), r.get("body")),
            label={"1": "estafa", "0": "legitimo"}.get(lab, ""),
            label_origen={"1": "phishing", "0": "legitimo"}.get(lab, lab),
            canal="email",
            idioma="es",
            fecha=(r.get("date") or "").strip(),
            dataset="spaphish",
        )


def spearphishmx(d: Path) -> Iterator[Fila]:
    for r in _dict_rows(d / "SpearPhishMX_dataset_publico.csv", ";"):
        lab = str(r.get("etiqueta", "")).strip()
        yield Fila(
            id=f"spearphishmx-{r['hash'][:16]}",
            texto=_texto(_asunto(r.get("asunto")), r.get("cuerpo_sanitizado")),
            label={"1": "estafa", "0": "legitimo"}.get(lab, ""),
            label_origen={"1": "spear_phishing", "0": "legitimo"}.get(lab, lab),
            canal="email",
            idioma="es",
            fuente_pais="MX",
            dataset="spearphishmx",
        )


def spamshield(d: Path) -> Iterator[Fila]:
    path = d / "Spanish__clean-00001.jsonl"
    for line in _read_text(path).splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        texto = _texto(str(r.get("text") or r.get("message") or ""))
        lab = str(r.get("label", r.get("labels", ""))).strip().lower()
        yield Fila(
            id=f"spamshield-{_hash_id(texto)}",
            texto=texto,
            label={"ham": "legitimo", "0": "legitimo", "not_spam": "legitimo"}.get(lab, ""),
            label_origen=lab,
            canal="",
            idioma="es",
            dataset="spamshield",
        )


def uci_sms(d: Path) -> Iterator[Fila]:
    for line in _read_text(d / "SMSSpamCollection").splitlines():
        lab, _, texto = line.partition("\t")
        if not texto:
            continue
        yield Fila(
            id=f"uci-{_hash_id(texto)}",
            texto=_texto(texto),
            label={"ham": "legitimo"}.get(lab.strip(), ""),  # "spam" no es necesariamente estafa
            label_origen=lab.strip(),
            canal="sms",
            idioma="en",
            fuente_pais="UK",
            dataset="uci-sms-spam",
        )


CONVERSORES: dict[str, tuple[str, Callable[[Path], Iterator[Fila]]]] = {
    "spaphish": ("datasets/spaphish", spaphish),
    "spearphishmx": ("datasets/spearphishmx", spearphishmx),
    "spamshield-es": ("datasets/spamshield", spamshield),
    "uci-sms-spam": ("datasets/uci-sms-spam", uci_sms),
}
CON_GOLDEN = {"spaphish"}


def es_golden(fila_id: str) -> bool:
    return int(hashlib.sha256(fila_id.encode()).hexdigest(), 16) % GOLDEN_FRACCION == 0


def write_csv(path: Path, filas: list[Fila]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNAS)
        w.writeheader()
        w.writerows(asdict(x) for x in filas)


def prepare(id_: str, raw: Path, interim: Path) -> str:
    sub, fn = CONVERSORES[id_]
    d = raw / sub
    if not d.exists():
        return f"– {id_:22s} no está bajado"
    vistos: set[str] = set()
    kb, golden = [], []
    for x in fn(d):
        if not x.texto or x.id in vistos:  # filas vacías o duplicadas exactas
            continue
        vistos.add(x.id)
        (golden if id_ in CON_GOLDEN and es_golden(x.id) else kb).append(x)
    write_csv(interim / "mensajes" / f"{id_}.csv", kb)
    if golden:
        write_csv(interim / "golden_candidatos" / f"{id_}.csv", golden)
    c = Counter(x.label or "sin_etiqueta" for x in kb)
    resumen = ", ".join(f"{k} {v}" for k, v in sorted(c.items()))
    extra = f" · golden {len(golden)}" if golden else ""
    return f"✓ {id_:22s} {len(kb):5d} filas ({resumen}){extra}"


def prepare_all(raw: Path, interim: Path) -> list[str]:
    out = []
    for id_ in CONVERSORES:
        try:
            out.append(prepare(id_, raw, interim))
        except (OSError, KeyError, ValueError, csv.Error) as e:
            out.append(f"✗ {id_:22s} {type(e).__name__}: {e}")
    return out
