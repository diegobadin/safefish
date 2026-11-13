"""Normalización de fuentes a documentos estructurados (M-ING.3, V2, §2.3).

Cada documento limpio se convierte en `CaseDoc | MessageDoc | SignalDoc | GuideDoc`:

- **Mensajes** y **señales** ya vienen estructurados (CSV con etiqueta, taxonomía): se
  convierten de forma determinística.
- **Casos** y **guías** se parsean por plantilla si ya vienen estructurados (secciones
  `Descripción:/Frases típicas:/Mecanismo:` o pasos numerados) y, si no, con un LLM *mini* con
  salida estructurada (`mode: auto`).

La normalización también actúa como **sanitizador** (§5.4): del texto de la fuente solo se
conservan campos del esquema. Controles sobre la salida del LLM:

- las `frases_tipicas` deben aparecer en la fuente (fuzzy ≥ `min_frase_score`), si no se
  descartan: una frase inventada contaminaría el retrieval por analogía;
- los teléfonos, URLs y mails de los pasos de una guía deben estar literalmente en la fuente,
  si no se reemplazan (mismo criterio que la regla R5 del verificador);
- la metadata curada en `sources.yaml` o en el front matter (modalidad, canal, situación) manda
  sobre la del LLM; las discrepancias se registran en el manifest.
"""

from __future__ import annotations

import contextvars
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from safefish.config import NormalizeCfg
from safefish.prompting import load_prompt_file
from safefish.schemas import (
    Canal,
    CaseDoc,
    DocTipo,
    GuideDoc,
    KBDoc,
    MessageDoc,
    Modalidad,
    Severidad,
    SignalDoc,
    SignalId,
    Situacion,
)
from safefish.taxonomia import load_taxonomia, render_modalidades, render_senales
from safefish.textutils import (
    cita_score,
    contacto_respaldado,
    escape_attr,
    escape_tags,
    find_contacts,
    strip_accents,
)

PROMPTS_DIR = Path(__file__).with_name("prompts")
MAX_CHARS_LLM = 12_000
CONTACTO_REDACTADO = "[contacto no verificado: usá el canal oficial que ya conozcas]"


# ---------------------------------------------------------------------------
# Coerción de metadata curada
# ---------------------------------------------------------------------------


def _as_list(v) -> list[str]:
    if v is None or v == "":
        return []
    if isinstance(v, str):
        return [x.strip() for x in re.split(r"[,|]", v) if x.strip()]
    return [str(x).strip() for x in v if str(x).strip()]


def _canales(v) -> list[Canal]:
    vals = [x.lower() for x in _as_list(v)]
    if "multi" in vals:
        return []
    return [Canal(x) for x in vals]


def _enum_list(enum, v) -> list:
    return [enum(x) for x in _as_list(v)]


def _provenance(md: dict, normalizer_version: str, doc_id: str | None = None) -> dict:
    return {
        "doc_id": doc_id or md["doc_id"],
        "fuente_url": md.get("fuente_url"),
        "fuente_tipo": md.get("fuente_tipo", "desconocido"),
        "source_id": md.get("source_id", ""),
        "fecha": str(md["fecha"]) if md.get("fecha") else None,
        "campaign_id": str(md["campaign_id"]) if md.get("campaign_id") else None,
        "normalizer_version": normalizer_version,
    }


# ---------------------------------------------------------------------------
# Parsers de plantilla (sin LLM)
# ---------------------------------------------------------------------------

_SECCION = re.compile(r"(?im)^\s*(descripci[oó]n|frases t[ií]picas|mecanismo)\s*:\s*")
_QUOTED = re.compile(r"[\"“«]([^\"”»]{3,})[\"”»]")
_PASO = re.compile(r"^\s*(?:\d+[.)]|[-•*])\s+(.+)$")


def _key(header: str) -> str:
    return strip_accents(header.lower()).split()[0]  # descripcion | frases | mecanismo


def split_frases(raw: str) -> list[str]:
    quoted = [q.strip() for q in _QUOTED.findall(raw)]
    if quoted:
        return quoted
    return [x.strip(" .-•*") for x in re.split(r"[;\n]", raw) if x.strip(" .-•*")]


def parse_case_template(text: str, md: dict) -> CaseDoc | None:
    """Casos curados con secciones fijas y `modalidad` en la metadata."""
    if not md.get("modalidad"):
        return None
    marks = list(_SECCION.finditer(text))
    if not marks:
        return None
    sections: dict[str, str] = {}
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        sections[_key(m.group(1))] = text[m.end() : end].strip()
    if not sections.get("descripcion"):
        return None
    return CaseDoc(
        **_provenance(md, "template"),
        modalidad=Modalidad(md["modalidad"]),
        canales=_canales(md.get("canal")),
        entidad_suplantada=md.get("entidad_suplantada") or md.get("entidad"),
        descripcion=sections["descripcion"],
        frases_tipicas=split_frases(sections.get("frases", "")),
        mecanismo=sections.get("mecanismo", ""),
        signal_ids=_enum_list(SignalId, md.get("signal_ids")),
    )


def parse_guide_template(text: str, md: dict) -> GuideDoc | None:
    """Guías curadas: contexto + lista numerada o con viñetas de pasos."""
    contexto, pasos = [], []
    for line in text.splitlines():
        if m := _PASO.match(line):
            pasos.append(m.group(1).strip())
        elif pasos and line.strip() and line.startswith((" ", "\t")):
            pasos[-1] += " " + line.strip()  # continuación del paso anterior
        elif not pasos and line.strip():
            contexto.append(line.strip())
    if not pasos:
        return None
    return GuideDoc(
        **_provenance(md, "template"),
        modalidades=_enum_list(Modalidad, md.get("modalidad")),
        situaciones=_enum_list(Situacion, md.get("situacion")),
        organismo=md.get("organismo"),
        contexto=" ".join(contexto),
        pasos=pasos,
    )


def message_from_document(doc: Document) -> MessageDoc | None:
    md = doc.metadata
    label = str(md.get("label", "")).strip().lower()
    if label not in {"estafa", "legitimo"}:
        return None
    canales = _canales(md.get("canal"))
    return MessageDoc(
        **_provenance(md, "template"),
        texto=doc.page_content,
        label=label,
        modalidad=Modalidad(md["modalidad"]) if md.get("modalidad") else None,
        canal=canales[0] if len(canales) == 1 else None,
        entidad=md.get("entidad") or md.get("entidad_suplantada"),
    )


def signal_from_document(doc: Document) -> SignalDoc:
    md = doc.metadata
    sid = md["signal_id"]
    s = load_taxonomia()["senales"][sid]
    return SignalDoc(
        **_provenance(md, "template"),
        signal_id=SignalId(sid),
        severidad=Severidad(s["severidad"]),
        tipo_senal=s.get("tipo", "riesgo"),
        definicion=s["definicion"].strip(),
        ejemplos=list(s.get("ejemplos") or []),
        contraejemplos=list(s.get("contraejemplos") or []),
    )


# ---------------------------------------------------------------------------
# Normalizador LLM
# ---------------------------------------------------------------------------


class _LLM(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CasoNorm(_LLM):
    modalidad: Modalidad
    canales: list[Canal]
    entidad_suplantada: str | None = Field(
        description="categoría en snake_case de a quién se suplanta, o null"
    )
    descripcion: str = Field(description="2 a 4 oraciones: qué pasa y qué pide el estafador")
    frases_tipicas: list[str] = Field(description="frases copiadas LITERALMENTE de la fuente")
    mecanismo: str = Field(description="cómo se concreta el daño")
    signal_ids: list[SignalId]


class CasosNorm(_LLM):
    describe_estafa: bool = Field(description="false si la fuente no describe ninguna modalidad")
    casos: list[CasoNorm]


class GuiaNorm(_LLM):
    modalidades: list[Modalidad] = Field(description="vacío si aplica a cualquier modalidad")
    situaciones: list[Situacion] = Field(description="vacío si aplica a cualquier situación")
    organismo: str | None = Field(description="quién publica la guía, tal como figura, o null")
    contexto: str = Field(description="una oración: a quién y cuándo aplica")
    pasos: list[str] = Field(description="pasos en orden, fieles a la fuente")


class GuiasNorm(_LLM):
    es_procedimiento: bool = Field(description="false si la fuente no indica qué hacer")
    guias: list[GuiaNorm]


class StructuredClient(Protocol):
    model: str

    def invoke(self, messages, schema, stage: str = "normalize"): ...


@dataclass
class NormalizeReport:
    metodo: Counter = field(default_factory=Counter)  # template | llm | crudo
    descartados: Counter = field(default_factory=Counter)
    frases_descartadas: int = 0
    contactos_redactados: int = 0
    discrepancias: list[dict] = field(default_factory=list)
    errores: list[dict] = field(default_factory=list)
    llm_doc_ids: list[str] = field(default_factory=list)

    def merge(self, other: NormalizeReport) -> None:
        self.metodo.update(other.metodo)
        self.descartados.update(other.descartados)
        self.frases_descartadas += other.frases_descartadas
        self.contactos_redactados += other.contactos_redactados
        self.discrepancias.extend(other.discrepancias)
        self.errores.extend(other.errores)
        self.llm_doc_ids.extend(other.llm_doc_ids)

    def to_dict(self) -> dict:
        return {
            "metodo": dict(self.metodo),
            "descartados": dict(self.descartados),
            "frases_descartadas": self.frases_descartadas,
            "contactos_redactados": self.contactos_redactados,
            "discrepancias_metadata": self.discrepancias,
            "errores": self.errores,
        }


def _sanitize_fuente(text: str) -> str:
    """La fuente va dentro de <fuente_no_confiable>: se escapan los ángulos para que no pueda
    abrir ni cerrar ningún delimitador (un borrado de una pasada se evade anidando tags)."""
    return escape_tags(text[:MAX_CHARS_LLM])


def redact_unsupported_contacts(paso: str, fuente: str) -> tuple[str, int]:
    """Reemplaza teléfonos, URLs y mails que no están literalmente en `fuente`."""
    n = 0
    for c in sorted(find_contacts(paso), key=len, reverse=True):
        # `c in paso`: un link puede haberse ido junto con el mail que lo contenía
        if c in paso and not contacto_respaldado(c, fuente):
            paso = paso.replace(c, CONTACTO_REDACTADO)
            n += 1
    return paso, n


class LLMNormalizer:
    def __init__(self, llm: StructuredClient, cfg: NormalizeCfg):
        self.llm, self.cfg = llm, cfg
        self.prompts = {
            t: load_prompt_file(PROMPTS_DIR / f"{cfg.prompt}_{t}.md") for t in ("caso", "guia")
        }

    def version(self, tipo: str) -> str:
        return f"{self.llm.model}:{self.cfg.prompt}_{tipo}:{self.prompts[tipo][1]}"

    def _messages(self, tipo: str, doc: Document):
        template, _ = self.prompts[tipo]
        md = doc.metadata
        return template.format_messages(
            modalidades=render_modalidades(),
            senales=render_senales(),
            situaciones=", ".join(s.value for s in Situacion),
            fuente_url=escape_attr(md.get("fuente_url") or "-"),
            titulo=escape_attr(md.get("titulo") or "-"),
            fuente=_sanitize_fuente(doc.page_content),
        )

    def cases(self, doc: Document, report: NormalizeReport) -> list[CaseDoc] | None:
        r = self.llm.invoke(self._messages("caso", doc), CasosNorm, stage="normalize")
        if r.parsed is None:
            report.errores.append({"doc_id": doc.metadata["doc_id"], "error": (r.error or "")[:300]})  # fmt: skip
            return None
        out: CasosNorm = r.parsed
        md = doc.metadata
        if not out.describe_estafa or not out.casos:
            report.descartados["no_describe_estafa"] += 1
            return []
        docs = []
        for i, c in enumerate(out.casos):
            frases = [f for f in c.frases_tipicas if cita_score(f, doc.page_content) >= self.cfg.min_frase_score]  # fmt: skip
            report.frases_descartadas += len(c.frases_tipicas) - len(frases)
            modalidad = c.modalidad
            if md.get("modalidad") and Modalidad(md["modalidad"]) != c.modalidad:
                report.discrepancias.append({"doc_id": md["doc_id"], "campo": "modalidad", "curado": md["modalidad"], "llm": c.modalidad.value})  # fmt: skip
                modalidad = Modalidad(md["modalidad"])
            if modalidad is Modalidad.NO_DETERMINADA:
                report.descartados["caso_sin_modalidad"] += 1
                continue
            doc_id = md["doc_id"] if len(out.casos) == 1 else f"{md['doc_id']}-{i + 1}"
            docs.append(
                CaseDoc(
                    **_provenance(md, self.version("caso"), doc_id),
                    modalidad=modalidad,
                    canales=_canales(md["canal"]) if md.get("canal") else c.canales,
                    entidad_suplantada=md.get("entidad_suplantada") or c.entidad_suplantada,
                    descripcion=c.descripcion,
                    frases_tipicas=frases,
                    mecanismo=c.mecanismo,
                    signal_ids=c.signal_ids,
                )
            )
        return docs

    def guides(self, doc: Document, report: NormalizeReport) -> list[GuideDoc] | None:
        r = self.llm.invoke(self._messages("guia", doc), GuiasNorm, stage="normalize")
        if r.parsed is None:
            report.errores.append({"doc_id": doc.metadata["doc_id"], "error": (r.error or "")[:300]})  # fmt: skip
            return None
        out: GuiasNorm = r.parsed
        md = doc.metadata
        if not out.es_procedimiento or not out.guias:
            report.descartados["no_es_procedimiento"] += 1
            return []
        docs = []
        for i, g in enumerate(out.guias):
            pasos = []
            for p in g.pasos:
                p, n = redact_unsupported_contacts(p, doc.page_content)
                report.contactos_redactados += n
                pasos.append(p)
            if not pasos:
                continue
            doc_id = md["doc_id"] if len(out.guias) == 1 else f"{md['doc_id']}-{i + 1}"
            docs.append(
                GuideDoc(
                    **_provenance(md, self.version("guia"), doc_id),
                    modalidades=_enum_list(Modalidad, md["modalidad"])
                    if md.get("modalidad")
                    else g.modalidades,  # fmt: skip
                    situaciones=_enum_list(Situacion, md["situacion"])
                    if md.get("situacion")
                    else g.situaciones,  # fmt: skip
                    organismo=md.get("organismo") or g.organismo,
                    contexto=g.contexto,
                    pasos=pasos,
                )
            )
        return docs


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------


def _crudo(doc: Document) -> CaseDoc | GuideDoc | None:
    """Sin plantilla ni LLM: el caso solo se conserva si la modalidad está curada."""
    md = doc.metadata
    try:
        if md["doc_tipo"] == DocTipo.CASO and md.get("modalidad"):
            return CaseDoc(
                **_provenance(md, "crudo"),
                modalidad=Modalidad(md["modalidad"]),
                canales=_canales(md.get("canal")),
                entidad_suplantada=md.get("entidad_suplantada"),
                descripcion=doc.page_content[:2000],
            )
        if md["doc_tipo"] == DocTipo.GUIA:
            return GuideDoc(
                **_provenance(md, "crudo"),
                modalidades=_enum_list(Modalidad, md.get("modalidad")),
                situaciones=_enum_list(Situacion, md.get("situacion")),
                organismo=md.get("organismo"),
                pasos=[doc.page_content[:2000]],
            )
    except (ValidationError, ValueError):
        return None
    return None


def normalize_one(
    doc: Document, cfg: NormalizeCfg, llm: LLMNormalizer | None, report: NormalizeReport
) -> list[KBDoc]:
    md = doc.metadata
    tipo = DocTipo(md["doc_tipo"])
    if tipo is DocTipo.MENSAJE:
        m = message_from_document(doc)
        if m is None:
            report.descartados["mensaje_sin_label"] += 1
            return []
        report.metodo["template"] += 1
        return [m]
    if tipo is DocTipo.SENAL:
        report.metodo["template"] += 1
        return [signal_from_document(doc)]

    if cfg.mode != "llm":
        parsed = (
            parse_case_template(doc.page_content, md)
            if tipo is DocTipo.CASO
            else parse_guide_template(doc.page_content, md)
        )
        if parsed is not None:
            report.metodo["template"] += 1
            return [parsed]
    if cfg.mode != "template" and llm is not None:
        out = llm.cases(doc, report) if tipo is DocTipo.CASO else llm.guides(doc, report)
        if out is not None:
            report.metodo["llm"] += len(out)
            report.llm_doc_ids.extend(d.doc_id for d in out)
            return out
    crudo = _crudo(doc)
    if crudo is None:
        report.descartados[f"{tipo.value}_no_normalizable"] += 1
        return []
    report.metodo["crudo"] += 1
    return [crudo]


def normalize_documents(
    docs: list[Document],
    cfg: NormalizeCfg,
    llm: LLMNormalizer | None = None,
    workers: int = 4,
) -> tuple[list[KBDoc], NormalizeReport]:
    def one(d: Document) -> tuple[list[KBDoc], NormalizeReport]:
        rep = NormalizeReport()  # un reporte por tarea: sin estado compartido entre hilos
        return normalize_one(d, cfg, llm, rep), rep

    # Cada tarea corre en una copia del contexto: el UsageTracker (ContextVar) sigue contando.
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = [ex.submit(contextvars.copy_context().run, one, d) for d in docs]
        results = [f.result() for f in futures]
    report = NormalizeReport()
    out: list[KBDoc] = []
    for kds, rep in results:  # orden de entrada: determinístico
        out.extend(kds)
        report.merge(rep)
    ids = Counter(d.doc_id for d in out)
    if dup := [i for i, c in ids.items() if c > 1]:
        raise ValueError(f"doc_id repetidos tras normalizar: {dup[:5]}")
    return out, report


# ---------------------------------------------------------------------------
# Plantillas de texto por colección (lo que se embebe, §1.4) y metadata
# ---------------------------------------------------------------------------


def _citar(frases: list[str]) -> str:
    return "; ".join(f'"{f}"' for f in frases)


def render_kbdoc(d: KBDoc) -> str:
    match d:
        case CaseDoc():
            partes = [f"Descripción: {d.descripcion}"]
            if d.frases_tipicas:
                partes.append(f"Frases típicas: {_citar(d.frases_tipicas)}")
            if d.mecanismo:
                partes.append(f"Mecanismo: {d.mecanismo}")
            return "\n".join(partes)
        case MessageDoc():
            return d.texto
        case SignalDoc():
            partes = [f"Señal: {d.signal_id.value.replace('_', ' ')}", f"Definición: {d.definicion}"]  # fmt: skip
            if d.ejemplos:
                partes.append("Ejemplos: " + " | ".join(d.ejemplos))
            if d.contraejemplos:
                partes.append("Contraejemplos (no es la señal): " + " | ".join(d.contraejemplos))
            return "\n".join(partes)
        case GuideDoc():
            pasos = "\n".join(f"{i}. {p}" for i, p in enumerate(d.pasos, 1))
            return f"{d.contexto}\nPasos:\n{pasos}".strip()
    raise TypeError(type(d))


DOC_TIPO = {CaseDoc: DocTipo.CASO, MessageDoc: DocTipo.MENSAJE, SignalDoc: DocTipo.SENAL, GuideDoc: DocTipo.GUIA}  # fmt: skip


def kbdoc_metadata(d: KBDoc) -> dict:
    """Metadata del chunk (se escalariza después para Chroma)."""
    base = d.model_dump(
        mode="json",
        include={"doc_id", "fuente_url", "fuente_tipo", "source_id", "fecha", "campaign_id", "normalizer_version"},
    )  # fmt: skip
    base["doc_tipo"] = DOC_TIPO[type(d)].value
    match d:
        case CaseDoc():
            base |= {
                "modalidad": d.modalidad.value,
                "canal": [c.value for c in d.canales] or ["multi"],
                "entidad_suplantada": d.entidad_suplantada,
                "signal_ids": [s.value for s in d.signal_ids],
                "frases_tipicas": " | ".join(d.frases_tipicas),
            }
        case MessageDoc():
            base |= {
                "label": d.label,
                "modalidad": d.modalidad.value if d.modalidad else None,
                "canal": [d.canal.value] if d.canal else None,
                "entidad": d.entidad,
            }
        case SignalDoc():
            base |= {"signal_id": d.signal_id.value, "severidad": d.severidad.value, "tipo_senal": d.tipo_senal}  # fmt: skip
        case GuideDoc():
            base |= {
                "modalidad": [m.value for m in d.modalidades] or ["general"],
                "situacion": [s.value for s in d.situaciones] or ["general"],
                "organismo": d.organismo,
                "pasos_json": json.dumps(d.pasos, ensure_ascii=False),
            }
    return base


def kbdoc_to_document(d: KBDoc) -> Document:
    return Document(id=d.doc_id, page_content=render_kbdoc(d), metadata=kbdoc_metadata(d))
