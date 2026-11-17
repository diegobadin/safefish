"""Transcripción estructurada de capturas con un LLM con visión (M-EXT, §2.3).

El LLM transcribe la conversación **literalmente**, separando segmentos por emisor. El texto
canónico (`ExtractedMessage.texto`) son los segmentos del remitente unidos por salto de línea:
contra ese texto se verifican las citas (R1). Las entidades se extraen después con regex
sobre el texto canónico, no con el LLM.
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from safefish.prompting import load_prompt_file

PROMPTS_DIR = Path(__file__).with_name("prompts")


class _LLM(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SegmentoLLM(_LLM):
    emisor: Literal["remitente", "usuario"] = Field(
        description="remitente = quien envía el mensaje sospechoso; usuario = la persona que pide ayuda"
    )
    texto: str = Field(description="transcripción literal, sin corregir")


class TranscripcionLLM(_LLM):
    canal: Literal["whatsapp", "sms", "email", "desconocido"]
    remitente: str | None = Field(description="nombre, número o dirección del remitente visible, o null")  # fmt: skip
    segmentos: list[SegmentoLLM]
    legible: bool = Field(description="false si la imagen no tiene texto legible")


class TranscripcionConMetadatosLLM(TranscripcionLLM):
    """vision_v3 (V10): además del texto, lo que se ve en la captura."""

    insignia_verificada: bool | None = Field(
        description="true si se ve la insignia de cuenta verificada junto al nombre, false si no, null si no se sabe"
    )  # fmt: skip
    descripcion: str = Field(
        description="2–3 oraciones factuales sobre lo que se ve además del texto: logo o marca, tipo de cuenta, bot con menú, adjuntos, avisos de la app"
    )  # fmt: skip


PROMPTS_SIN_METADATOS = ("vision_v1", "vision_v2")


def image_data_url(path: str | Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return f"data:{mime};base64,{data}"


class VisionTranscriber:
    def __init__(self, llm, prompt: str = "vision_v1"):
        self.llm, self.prompt = llm, prompt

    @property
    def prompt_hash(self) -> str:
        return load_prompt_file(PROMPTS_DIR / f"{self.prompt}.md")[1]

    @property
    def schema(self) -> type[TranscripcionLLM]:
        if self.prompt in PROMPTS_SIN_METADATOS:
            return TranscripcionLLM
        return TranscripcionConMetadatosLLM

    def messages(self, image_path: str):
        template, _ = load_prompt_file(PROMPTS_DIR / f"{self.prompt}.md")
        system, human = template.format_messages()
        return [
            SystemMessage(content=system.content),
            HumanMessage(
                content=[
                    {"type": "text", "text": human.content},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url(image_path), "detail": "high"},
                    },  # fmt: skip
                ]
            ),
        ]

    def transcribe(self, image_path: str) -> TranscripcionLLM:
        r = self.llm.invoke(self.messages(image_path), self.schema, stage="extract_vision")
        if r.parsed is None:
            raise ValueError(f"Transcripción inválida: {(r.error or '')[:300]}")
        return r.parsed
