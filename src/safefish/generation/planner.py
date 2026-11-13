"""Planner desde Guías (M-GEN.2, V2+).

1. **Selección determinística** de guías por metadata `(modalidad, situación)` sobre las guías
   del snapshot (`chunks.jsonl`), con niveles de *fallback* y, si no hay ninguna por metadata,
   búsqueda densa en la colección `guias`.
2. **Redacción**: el LLM *solo reordena y redacta* pasos de las guías seleccionadas; cada paso
   referencia su `guide_id`. Los `guide_id` que no son de las guías seleccionadas se descartan.
   Si el LLM falla o no deja pasos válidos, se usa el plan **extractivo** (pasos literales).

El planner ve solo campos estructurados (veredicto, modalidad, situación, ids de señales),
nunca el texto del mensaje: así una inyección en el mensaje no llega a esta etapa.
La regla R5 del verificador controla después contactos y organismos inventados.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from safefish.llm import LLMResult
from safefish.prompting import load_prompt_file
from safefish.schemas import (
    EvidenceItem,
    Modalidad,
    PasoPlan,
    SafeFishOutput,
    Situacion,
)
from safefish.taxonomia import load_taxonomia
from safefish.textutils import escape_attr, escape_tags

PROMPTS_DIR = Path(__file__).with_name("prompts")
GENERALES = {"general", "no_determinada", ""}
SITUACION_POR_DEFECTO = Situacion.NO_RESPONDIO

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
        if especifica:
            tiers += [
                ("exacta", lambda g: mod in g.modalidades and sit in g.situaciones),
                ("modalidad", lambda g: mod in g.modalidades and not g.situaciones),
            ]
        tiers += [
            ("general_situacion", lambda g: g.es_general and sit in g.situaciones),
            ("general", lambda g: g.es_general and not g.situaciones),
        ]
        if especifica:
            tiers.append(("modalidad_otra_situacion", lambda g: mod in g.modalidades))

        out: list[tuple[GuideRecord, Via]] = []
        seen: set[str] = set()
        for via, pred in tiers:
            for g in self.guides:
                if len(out) >= k:
                    return out
                if g.id not in seen and pred(g):
                    out.append((g, via))
                    seen.add(g.id)
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
        # Los pasos vienen de la KB (fuentes externas): no pueden abrir ni cerrar delimitadores.
        pasos = "\n".join(f"{i}. {escape_tags(p)}" for i, p in enumerate(g.pasos, 1))
        org = escape_attr(g.metadata.get("organismo") or "-")
        blocks.append(f'<guia id="{escape_attr(g.id)}" organismo="{org}">\n{pasos}\n</guia>')
    return "\n\n".join(blocks)


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

    def plan(self, out: SafeFishOutput, situacion: Situacion | None) -> PlanResult:
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
        messages = template.format_messages(
            veredicto=out.veredicto.value,
            modalidad=out.modalidad.value,
            situacion=sit.value,
            senales=", ".join(sorted({s.signal_id.value for s in out.senales})) or "ninguna",
            max_pasos=self.max_pasos,
            guias=render_guias(sel),
        )
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
