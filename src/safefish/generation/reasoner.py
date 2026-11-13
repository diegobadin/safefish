"""Razonador (M-GEN.1): veredicto, modalidad, señales con cita, casos similares y plan.

Salida con `with_structured_output` (JSON Schema estricto, enums cerrados), temperatura 0 y
un reintento si la respuesta no valida (§2.2). En V0–V1 el plan lo genera el razonador; desde
V2 lo arma el planner a partir de las Guías.
"""

from __future__ import annotations

import hashlib
import re
from functools import cache
from pathlib import Path
from typing import Protocol

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ConfigDict, Field

from safefish.llm import LLMResult
from safefish.schemas import (
    CasoRef,
    ExtractedMessage,
    Modalidad,
    PasoPlan,
    RetrievalBundle,
    RunMeta,
    SafeFishOutput,
    SenalDetectada,
    SignalId,
    Situacion,
    Veredicto,
)
from safefish.taxonomia import render_modalidades, render_senales

PROMPTS_DIR = Path(__file__).with_name("prompts")

# ---------------------------------------------------------------------------
# Esquema que ve el LLM (subconjunto de SafeFishOutput: sin meta, bbox ni fuentes, que se
# completan de forma determinística). Todos los campos requeridos: JSON Schema estricto.
# ---------------------------------------------------------------------------


class _LLM(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SenalLLM(_LLM):
    signal_id: SignalId
    cita: str = Field(description="Fragmento copiado literalmente del mensaje")
    explicacion: str = Field(description="Una oración: por qué es una señal")


class CasoLLM(_LLM):
    case_id: str = Field(description="id de un <caso> o <mensaje_similar> de la evidencia")
    por_que: str


class PasoLLM(_LLM):
    paso: str
    guide_id: str | None = Field(description="id de la <guia> de la que sale el paso, o null")


class ReasonerLLMOutput(_LLM):
    veredicto: Veredicto
    modalidad: Modalidad
    senales: list[SenalLLM]
    casos_similares: list[CasoLLM]
    plan: list[PasoLLM]


class LLMClient(Protocol):
    model: str

    def invoke(self, messages, schema, stage: str = "reason") -> LLMResult: ...


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


@cache
def load_prompt(name: str) -> tuple[ChatPromptTemplate, str]:
    """Devuelve (template, hash corto del archivo)."""
    raw = (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")
    body = "\n".join(ln for ln in raw.splitlines() if not ln.startswith("# "))
    parts = re.split(r"^## (system|human)\s*$", body, flags=re.M)
    sections = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}
    template = ChatPromptTemplate.from_messages(
        [("system", sections["system"]), ("human", sections["human"])]
    )
    return template, hashlib.sha256(raw.encode()).hexdigest()[:12]


def _attr(v) -> str:
    return str(v).replace('"', "'")


def render_evidencia(b: RetrievalBundle) -> str:
    if b.is_empty():
        return "(sin evidencia recuperada)"
    blocks: list[str] = []
    for it in b.casos:
        md = it.metadata
        blocks.append(
            f'<caso id="{_attr(md.get("doc_id", it.id))}" modalidad="{_attr(md.get("modalidad", "?"))}" '
            f'canal="{_attr(md.get("canal", "?"))}">\n{it.texto}\n</caso>'
        )
    for it in b.mensajes:
        md = it.metadata
        blocks.append(
            f'<mensaje_similar id="{_attr(md.get("doc_id", it.id))}" etiqueta="{_attr(md.get("label", "?"))}" '
            f'modalidad="{_attr(md.get("modalidad", "-"))}">\n{it.texto}\n</mensaje_similar>'
        )
    for it in b.senales:
        md = it.metadata
        blocks.append(
            f'<senal_definicion id="{_attr(md.get("signal_id", it.id))}">\n{it.texto}\n</senal_definicion>'
        )
    for it in b.guias:
        md = it.metadata
        blocks.append(
            f'<guia id="{_attr(md.get("doc_id", it.id))}" modalidad="{_attr(md.get("modalidad", "-"))}" '
            f'situacion="{_attr(md.get("situacion", "-"))}">\n{it.texto}\n</guia>'
        )
    return "\n\n".join(blocks)


_TAG = re.compile(r"(?i)</?\s*mensaje_no_confiable[^>]*>")


def sanitize_mensaje(texto: str) -> str:
    """Neutraliza intentos de cerrar el delimitador desde el propio mensaje (§5.2)."""
    return _TAG.sub(lambda m: m.group(0).replace("<", "‹").replace(">", "›"), texto)


def build_messages(
    prompt_name: str, m: ExtractedMessage, b: RetrievalBundle, s: Situacion | None
) -> list[BaseMessage]:
    template, _ = load_prompt(prompt_name)
    return template.format_messages(
        modalidades=render_modalidades(),
        senales=render_senales(),
        evidencia=render_evidencia(b),
        situacion=s.value if s else "no_informada",
        canal=m.canal.value if m.canal else "desconocido",
        mensaje=sanitize_mensaje(m.texto),
    )


# ---------------------------------------------------------------------------


class ReasonerError(RuntimeError):
    def __init__(self, message: str, reintentos: int, raw_text: str = ""):
        super().__init__(message)
        self.reintentos, self.raw_text = reintentos, raw_text


class LLMReasoner:
    def __init__(self, llm: LLMClient, prompt: str = "reasoner_v1", max_retries: int = 1):
        self.llm, self.prompt, self.max_retries = llm, prompt, max_retries

    @property
    def prompt_hash(self) -> str:
        return load_prompt(self.prompt)[1]

    def reason(
        self, m: ExtractedMessage, b: RetrievalBundle, s: Situacion | None, meta: RunMeta
    ) -> SafeFishOutput:
        messages = build_messages(self.prompt, m, b, s)
        result = self.llm.invoke(messages, ReasonerLLMOutput, stage="reason")
        retries = 0
        while result.parsed is None and retries < self.max_retries:
            retries += 1
            fix = HumanMessage(
                content=(
                    "Tu respuesta anterior no cumplió el esquema. Error: "
                    f"{(result.error or 'desconocido')[:500]}\n"
                    "Respondé de nuevo usando solo los valores permitidos."
                )
            )
            result = self.llm.invoke([*messages, fix], ReasonerLLMOutput, stage="reason_retry")
        if result.parsed is None:
            raise ReasonerError(result.error or "salida inválida", retries, result.raw_text)

        meta = meta.model_copy(
            update={
                "reintentos_parseo": retries,
                "modelos": {**meta.modelos, "reasoner": self.llm.model},
            }
        )
        return to_output(result.parsed, b, meta)


def to_output(p: ReasonerLLMOutput, b: RetrievalBundle, meta: RunMeta) -> SafeFishOutput:
    """Completa de forma determinística lo que el LLM no debe generar (URLs de fuente)."""
    urls: dict[str, str | None] = {}
    for it in b.all_items():
        doc_id = it.metadata.get("doc_id", it.id)
        urls.setdefault(doc_id, it.metadata.get("fuente_url"))
    return SafeFishOutput(
        veredicto=p.veredicto,
        modalidad=p.modalidad,
        senales=[
            SenalDetectada(signal_id=x.signal_id, cita=x.cita, explicacion=x.explicacion)
            for x in p.senales
        ],
        casos_similares=[
            CasoRef(case_id=c.case_id, fuente_url=urls.get(c.case_id), por_que=c.por_que)
            for c in p.casos_similares
        ],
        plan=[PasoPlan(paso=x.paso, guide_id=x.guide_id) for x in p.plan],
        meta=meta,
    )
