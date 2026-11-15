"""Planner desde Guías (M-GEN.2, V2+).

1. **Selección determinística** de guías por metadata `(modalidad, situación)` sobre las guías
   del snapshot (`chunks.jsonl`), con niveles de *fallback* y, si no hay ninguna por metadata,
   búsqueda densa en la colección `guias`.
2. **Redacción**: el LLM *solo reordena y redacta* pasos de las guías seleccionadas; cada paso
   referencia su `guide_id`. Los `guide_id` que no son de las guías seleccionadas se descartan.
   Si el LLM falla o no deja pasos válidos, se usa el plan **extractivo** (pasos literales).

`planner_v1` ve solo campos estructurados (veredicto, modalidad, situación, ids de señales).
`planner_v2` ve además el mensaje, como dato, para adaptar los pasos: las guías dan la
estructura de cómo proceder y, si una guía es de otra entidad, su nombre y sus contactos se
reemplazan por "la entidad oficial" del mensaje.

3. **Adaptación determinística** (todas las versiones con guías, con o sin verificador): en cada
   paso se reemplazan los contactos que no están en las guías seleccionadas y los contactos y
   nombres de una entidad suplantable (bancos, ARCA, ANSES, Correo Argentino) que el mensaje
   no nombra. Así un contacto de ARCA no aparece en el plan de un mail de otra empresa.
   La regla R5 del verificador controla después contactos y organismos inventados.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field

from safefish.llm import LLMResult
from safefish.prompting import load_prompt_file
from safefish.schemas import (
    EvidenceItem,
    ExtractedMessage,
    Modalidad,
    PasoPlan,
    SafeFishOutput,
    Situacion,
)
from safefish.taxonomia import load_taxonomia
from safefish.textutils import (
    contacto_respaldado,
    escape_attr,
    escape_tags,
    normalize_for_match,
    redact_contacts,
    strip_accents,
)

PROMPTS_DIR = Path(__file__).with_name("prompts")
GENERALES = {"general", "no_determinada", ""}
SITUACION_POR_DEFECTO = Situacion.NO_RESPONDIO
SIN_DANO = {Situacion.NO_RESPONDIO.value, Situacion.RESPONDIO.value}  # no pagó ni entregó datos
MAX_CHARS_MENSAJE = 2000  # planner_v2: alcanza para un mail largo sin inflar el costo


def es_extranjera(g: GuideRecord) -> bool:
    pais = str(g.metadata.get("fuente_pais") or "").upper()
    host = urlparse(str(g.metadata.get("fuente_url") or "")).hostname or ""
    return (bool(pais) and pais != "AR") or host.endswith(".es")


Via = Literal["exacta", "modalidad", "general_situacion", "general", "modalidad_otra_situacion", "densa"]  # fmt: skip


def _split(v) -> set[str]:
    if v is None:
        return set()
    if isinstance(v, list | tuple | set):
        return {str(x).strip() for x in v if str(x).strip()}
    return {x.strip() for x in str(v).split(",") if x.strip()}


_PASO_LINEA = re.compile(r"^\s*\d+[.)]\s+(.+)$")


@dataclass
class GuideRecord:
    id: str
    texto: str
    metadata: dict
    pasos: list[str]
    modalidades: set[str] = field(default_factory=set)
    situaciones: set[str] = field(default_factory=set)

    @classmethod
    def from_chunk(cls, chunk_id: str, texto: str, md: dict) -> GuideRecord:
        if md.get("pasos_json"):
            pasos = json.loads(md["pasos_json"])
        else:  # snapshot V1 (sin normalizar): pasos numerados del texto, o el texto entero
            pasos = [m.group(1).strip() for ln in texto.splitlines() if (m := _PASO_LINEA.match(ln))]  # fmt: skip
            pasos = pasos or [texto.strip()]
        return cls(
            id=str(md.get("doc_id", chunk_id)),
            texto=texto,
            metadata=md,
            pasos=pasos,
            modalidades=_split(md.get("modalidad")) - GENERALES,
            situaciones=_split(md.get("situacion")) - GENERALES,
        )

    @property
    def es_general(self) -> bool:
        return not self.modalidades

    def aplica_a_situacion(self, s: str) -> bool:
        return not self.situaciones or s in self.situaciones

    def to_evidence(self, via: str) -> EvidenceItem:
        return EvidenceItem(
            id=self.id, coleccion="guias", texto=self.texto, metadata={**self.metadata, "via": via}
        )


class GuideStore:
    """Guías del snapshot en memoria. `dense` es un buscador opcional `(query, k) -> [ids]`."""

    def __init__(
        self,
        guides: list[GuideRecord],
        dense: Callable[[str, int], list[str]] | None = None,
    ):
        self.guides = sorted(guides, key=lambda g: g.id)
        self.by_id = {g.id: g for g in self.guides}
        self.dense = dense

    @classmethod
    def from_snapshot(cls, snapshot_dir: Path, dense=None) -> GuideStore:
        path = snapshot_dir / "chunks.jsonl"
        guides: dict[str, GuideRecord] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                md = row["metadata"]
                if md.get("doc_tipo") != "guia":
                    continue
                g = GuideRecord.from_chunk(row["id"], row["texto"], md)
                if g.id in guides:  # V1: una guía partida en varios chunks → se unen
                    prev = guides[g.id]
                    prev.texto += "\n" + g.texto
                    prev.pasos += [p for p in g.pasos if p not in prev.pasos]
                else:
                    guides[g.id] = g
        return cls(list(guides.values()), dense)

    def select(
        self, modalidad: Modalidad, situacion: Situacion, k: int
    ) -> list[tuple[GuideRecord, Via]]:
        mod, sit = modalidad.value, situacion.value
        especifica = mod not in GENERALES
        tiers: list[tuple[Via, Callable[[GuideRecord], bool]]] = []
        if especifica and sit in SIN_DANO:
            # Primero las guías solo de prevención; las que mezclan "respondió" con "pagó" traen
            # pasos para víctimas (cancelar la operación, denunciar) y van después.
            tiers += [
                (
                    "exacta",
                    lambda g: (
                        mod in g.modalidades and sit in g.situaciones and g.situaciones <= SIN_DANO
                    ),
                ),  # fmt: skip
                ("modalidad", lambda g: mod in g.modalidades and not g.situaciones),
                ("exacta", lambda g: mod in g.modalidades and sit in g.situaciones),
            ]
        elif especifica:
            tiers += [
                ("exacta", lambda g: mod in g.modalidades and sit in g.situaciones),
                ("modalidad", lambda g: mod in g.modalidades and not g.situaciones),
            ]
        if sit in SIN_DANO:
            # Sin pago ni datos entregados, el plan es de prevención: nunca los protocolos para
            # víctimas (bloquear cuentas, denuncia penal), que son las guías generales sin
            # situación y las de otras situaciones de la misma modalidad.
            tiers.append(
                ("general_situacion", lambda g: g.es_general and bool(g.situaciones & SIN_DANO))
            )
            if especifica:
                tiers.append(
                    (
                        "modalidad_otra_situacion",
                        lambda g: mod in g.modalidades and bool(g.situaciones & SIN_DANO),
                    )  # fmt: skip
                )
        else:
            tiers += [
                ("general_situacion", lambda g: g.es_general and sit in g.situaciones),
                ("general", lambda g: g.es_general and not g.situaciones),
            ]
            if especifica:
                tiers.append(("modalidad_otra_situacion", lambda g: mod in g.modalidades))

        out: list[tuple[GuideRecord, Via]] = []
        seen: set[str] = set()

        # Dentro de cada nivel, las guías de fuentes argentinas antes que las extranjeras
        # (INCIBE es de España: sus pasos nombran organismos y medios de pago de allá).
        guias = sorted(self.guides, key=es_extranjera)

        def recorrer(niveles) -> None:
            for via, pred in niveles:
                for g in guias:
                    if len(out) >= k:
                        return
                    if g.id not in seen and pred(g):
                        out.append((g, via))
                        seen.add(g.id)

        recorrer(tiers)
        if not out and sit in SIN_DANO:
            # Sin ninguna guía de prevención: antes que un plan vacío, las de la modalidad en
            # otra situación y las generales.
            recorrer(
                [
                    ("modalidad_otra_situacion", lambda g: especifica and mod in g.modalidades),
                    ("general", lambda g: g.es_general and not g.situaciones),
                ]  # fmt: skip
            )
        if len(out) >= k:
            return out
        if not out and self.dense is not None:
            nombre = load_taxonomia()["modalidades"].get(mod, {}).get("nombre", mod)
            for gid in self.dense(f"Qué hacer: {nombre}. Situación: {sit.replace('_', ' ')}", k):
                if gid in self.by_id and gid not in seen:
                    out.append((self.by_id[gid], "densa"))
                    seen.add(gid)
        return out[:k]


# ---------------------------------------------------------------------------
# Redacción con LLM
# ---------------------------------------------------------------------------


class _LLM(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PasoPlanLLM(_LLM):
    paso: str = Field(description="Un paso concreto, en segunda persona (vos)")
    guide_id: str = Field(description="id de la <guia> de la que sale el paso")


class PlanLLMOutput(_LLM):
    pasos: list[PasoPlanLLM]


@dataclass
class PlanResult:
    plan: list[PasoPlan]
    guias: list[EvidenceItem]
    info: dict


def render_guias(sel: list[tuple[GuideRecord, str]]) -> str:
    blocks = []
    for g, _ in sel:
        # Contexto + pasos numerados: el mismo texto contra el que R5 verifica el plan. Viene de
        # la KB (fuentes externas): no puede abrir ni cerrar delimitadores.
        org = escape_attr(g.metadata.get("organismo") or "-")
        texto = escape_tags(g.texto)
        blocks.append(f'<guia id="{escape_attr(g.id)}" organismo="{org}">\n{texto}\n</guia>')
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Adaptación a la entidad del mensaje
# ---------------------------------------------------------------------------

# Entidades que las estafas suplantan y que aparecen en las guías con nombre o contacto propio.
# Solo organizaciones a las que la persona contactaría; no plataformas que usa (WhatsApp) ni
# organismos de denuncia generales (BCRA, UFECI, Defensa del Consumidor), que valen siempre.
ENTIDADES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    # clave: (alias en texto, dominios)
    "banco_nacion": (("banco nacion", "banco de la nacion argentina", "bna"), ("bna.com.ar",)),
    "banco_provincia": (("banco provincia", "banco de la provincia", "bapro"), ("bancoprovincia.com.ar",)),  # fmt: skip
    "credicoop": (("credicoop", "banco credicoop"), ("bancocredicoop.coop",)),
    "arca": (("arca", "afip", "domicilio fiscal electronico"), ("arca.gob.ar", "afip.gob.ar")),
    "anses": (("anses",), ("anses.gob.ar",)),
    "correo_argentino": (("correo argentino",), ("correoargentino.com.ar",)),
}
ENTIDAD_GENERICA = "la entidad oficial"
CANALES_GENERICOS = "los canales oficiales de la entidad (su app o la web que ya conocés)"
CONTACTO_SIN_GUIA = "[contacto eliminado: usá el canal oficial que ya conozcas]"


def _palabras(texto: str) -> str:
    """Minúsculas, sin tildes y sin puntuación: 'AFIP: multa' -> 'afip multa'."""
    return " ".join(re.sub(r"[^\w]+", " ", normalize_for_match(texto)).split())


def _menciona(texto_norm: str, alias: str) -> bool:
    return f" {_palabras(alias)} " in f" {texto_norm} "


def entidades_mencionadas(texto: str) -> set[str]:
    t = _palabras(texto)
    return {k for k, (alias, _) in ENTIDADES.items() if any(_menciona(t, a) for a in alias)}


def _host(contacto: str) -> str:
    c = contacto.lower()
    if "@" in c:
        return c.rsplit("@", 1)[1]
    c = re.sub(r"^[a-z]+://", "", c)
    return c.split("/", 1)[0].split(":", 1)[0]


def _entidad_de_contacto(contacto: str) -> str | None:
    h = _host(contacto)
    for k, (_, dominios) in ENTIDADES.items():
        if any(h == d or h.endswith("." + d) for d in dominios):
            return k
    return None


def _reemplazar_nombre(texto: str, alias: str) -> str:
    """Cambia el nombre de una entidad por "la entidad oficial", con el artículo que corresponda."""
    patron = r"\b(?:(del|al|el|de la|la)\s+)?" + r"\s+".join(map(re.escape, alias.split())) + r"\b"

    def sub(m: re.Match) -> str:
        art = (m.group(1) or "").lower()
        nuevo = {"del": "de ", "al": "a ", "de la": "de "}.get(art, "") + ENTIDAD_GENERICA
        return nuevo[0].upper() + nuevo[1:] if m.start() == 0 else nuevo

    # El alias se busca sin tildes y se reemplaza sobre el original en las mismas posiciones
    # (en NFC, quitar tildes no cambia el largo; si cambiara, se busca sobre el original).
    plano = strip_accents(texto)
    if len(plano) != len(texto):
        return re.sub(patron, sub, texto, flags=re.I)
    out, ultimo = [], 0
    for m in re.finditer(patron, plano, flags=re.I):
        out += [texto[ultimo : m.start()], sub(m)]
        ultimo = m.end()
    return "".join(out + [texto[ultimo:]])


def adaptar_paso(paso: str, fuente_guias: str, mencionadas: set[str]) -> tuple[str, list[dict]]:
    """Deja un paso apto para este mensaje. Devuelve (paso, cambios):

    - contacto de una entidad que el mensaje no nombra → canales oficiales genéricos;
    - contacto que no está en las guías seleccionadas → aviso de contacto eliminado;
    - nombre de una entidad que el mensaje no nombra → "la entidad oficial"."""
    cambios: list[dict] = []

    def ok_entidad(c: str) -> bool:
        e = _entidad_de_contacto(c)
        return e is None or e in mencionadas

    paso, otros = redact_contacts(paso, ok_entidad, CANALES_GENERICOS)
    cambios += [{"accion": "contacto_de_otra_entidad", "contacto": c} for c in otros]
    paso, sin_guia = redact_contacts(paso, lambda c: contacto_respaldado(c, fuente_guias), CONTACTO_SIN_GUIA)  # fmt: skip
    cambios += [{"accion": "contacto_sin_guia", "contacto": c} for c in sin_guia]
    for k, (alias, _) in ENTIDADES.items():
        if k in mencionadas:
            continue
        for a in sorted(alias, key=len, reverse=True):
            nuevo = _reemplazar_nombre(paso, a)
            if nuevo != paso:
                cambios.append({"accion": "nombre_de_otra_entidad", "entidad": k})
                paso = nuevo
    return paso, cambios


def extractive_plan(sel: list[tuple[GuideRecord, str]], max_pasos: int) -> list[PasoPlan]:
    """Pasos literales de las guías, en orden, intercalando guías si hay más de una."""
    plan: list[PasoPlan] = []
    seen: set[str] = set()
    longest = max((len(g.pasos) for g, _ in sel), default=0)
    for i in range(longest):
        for g, _ in sel:
            if i < len(g.pasos) and g.pasos[i] not in seen and len(plan) < max_pasos:
                plan.append(PasoPlan(paso=g.pasos[i], guide_id=g.id))
                seen.add(g.pasos[i])
    return plan


class GuidePlanner:
    def __init__(
        self,
        store: GuideStore,
        llm=None,
        *,
        mode: Literal["llm", "extractivo"] = "llm",
        prompt: str = "planner_v1",
        top_guias: int = 2,
        max_pasos: int = 5,
    ):
        if mode == "llm" and llm is None:
            raise ValueError("planner_mode=llm necesita un cliente LLM")
        self.store, self.llm, self.mode = store, llm, mode
        self.prompt, self.top_guias, self.max_pasos = prompt, top_guias, max_pasos

    @property
    def prompt_hash(self) -> str | None:
        return load_prompt_file(PROMPTS_DIR / f"{self.prompt}.md")[1] if self.mode == "llm" else None  # fmt: skip

    def plan(
        self,
        out: SafeFishOutput,
        situacion: Situacion | None,
        m: ExtractedMessage | None = None,
    ) -> PlanResult:
        res = self._plan(out, situacion, m)
        # Adaptación determinística a la entidad del mensaje (también para el plan extractivo).
        fuente = "\n".join(g.texto for g in res.guias)
        mencionadas = entidades_mencionadas(m.texto) if m is not None else set()
        plan, cambios = [], []
        for i, p in enumerate(res.plan):
            texto, c = adaptar_paso(p.paso, fuente, mencionadas)
            cambios += [{**x, "paso": i} for x in c]
            plan.append(p.model_copy(update={"paso": texto}))
        res.info["entidades_mensaje"] = sorted(mencionadas)
        res.info["adaptaciones"] = cambios
        return PlanResult(plan, res.guias, res.info)

    def _plan(
        self, out: SafeFishOutput, situacion: Situacion | None, m: ExtractedMessage | None
    ) -> PlanResult:
        sit = situacion or SITUACION_POR_DEFECTO
        sel = self.store.select(out.modalidad, sit, self.top_guias)
        info: dict = {
            "situacion": sit.value,
            "situacion_asumida": situacion is None,
            "guias": [{"id": g.id, "via": via} for g, via in sel],
            "modo": self.mode,
            "fallback_extractivo": False,
            "pasos_descartados": 0,
        }
        guias = [g.to_evidence(via) for g, via in sel]
        if not sel:
            return PlanResult([], guias, info)
        if self.mode == "extractivo":
            return PlanResult(extractive_plan(sel, self.max_pasos), guias, info)

        template, _ = load_prompt_file(PROMPTS_DIR / f"{self.prompt}.md")
        campos = {
            "veredicto": out.veredicto.value,
            "modalidad": out.modalidad.value,
            "situacion": sit.value,
            "senales": ", ".join(sorted({s.signal_id.value for s in out.senales})) or "ninguna",
            "max_pasos": self.max_pasos,
            "guias": render_guias(sel),
        }
        if "mensaje" in template.input_variables:  # planner_v2: el mensaje, como dato
            texto = m.texto if m is not None else ""
            campos["mensaje"] = escape_tags(texto[:MAX_CHARS_MENSAJE]) or "(sin texto)"
            campos["canal"] = escape_attr(m.canal.value if m is not None and m.canal else "desconocido")  # fmt: skip
        messages = template.format_messages(**campos)
        r: LLMResult = self.llm.invoke(messages, PlanLLMOutput, stage="plan")
        ids = {g.id for g, _ in sel}
        plan: list[PasoPlan] = []
        if r.parsed is not None:
            for p in r.parsed.pasos:
                if p.guide_id in ids and p.paso.strip():
                    plan.append(PasoPlan(paso=p.paso.strip(), guide_id=p.guide_id))
                else:
                    info["pasos_descartados"] += 1
        else:
            info["error"] = (r.error or "")[:300]
        if not plan:
            info["fallback_extractivo"] = True
            plan = extractive_plan(sel, self.max_pasos)
        return PlanResult(plan[: self.max_pasos], guias, info)
