"""Cliente LLM con salida estructurada, caché y tracking (M-GEN).

`StructuredLLM.invoke(messages, schema)` devuelve el objeto Pydantic parseado o el error de
parseo, sin lanzar: la política de reintentos la decide quien llama (el razonador).
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ValidationError

from safefish.cache import KVCache, cache_enabled, stable_hash
from safefish.config import CACHE_DIR
from safefish.observability import LLMCall, current_tracker


@dataclass
class LLMResult:
    parsed: BaseModel | None
    raw_text: str
    error: str | None = None
    cache_hit: bool = False
    usage: dict = field(default_factory=dict)


def esfuerzo_por_defecto(model: str) -> str | None:
    """GPT-5 (snapshot 2025-08-07 y sus mini/nano) razona por defecto con esfuerzo `medium`:
    tokens de salida extra, latencia y costo. Para clasificar y extraer alcanza `minimal`
    (0 tokens de razonamiento en las pruebas). Los GPT-5.x posteriores ya no razonan por defecto.
    GPT-5 tampoco acepta temperatura 0: LangChain la descarta, y la reproducibilidad queda a
    cargo de la seed y la caché."""
    if es_claude(model):
        # Claude Sonnet 5.5 / Haiku 5.5: el razonamiento adaptativo no se apaga; `low` alcanza
        # para clasificar y extraer con JSON y deja el costo cerca de un modelo sin razonamiento.
        return "low"
    return "minimal" if re.match(r"gpt-5(-mini|-nano)?(-\d{4}-\d{2}-\d{2})?$", model) else None


def es_claude(model: str) -> bool:
    return model.startswith("claude-")


# Modelos de Claude con respaldo del lado del servidor ante un rechazo por política: la API
# re-ejecuta el mismo pedido en otro modelo dentro de la misma llamada. Haiku 5.5 no lo tiene.
CLAUDE_CON_RESPALDO = {"claude-sonnet-5-5", "claude-opus-5-5"}
CLAUDE_MAX_TOKENS = 8000  # JSON de clasificación + razonamiento en `low`; sobra margen


class StructuredLLM:
    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        seed: int | None = None,
        timeout: float = 60.0,
        kv: KVCache | None = None,
        reasoning_effort: str | None = None,
    ):
        self.model, self.temperature, self.seed, self.timeout = model, temperature, seed, timeout
        self.kv = kv
        self.reasoning_effort = reasoning_effort
        self._chat = None

    @classmethod
    def from_cfg(cls, model: str, temperature: float, seed: int | None, timeout: float = 60.0):
        kv = KVCache(CACHE_DIR / "llm.sqlite") if cache_enabled() else None
        return cls(model, temperature, seed, timeout, kv, esfuerzo_por_defecto(model))

    def _client(self):
        if self._chat is None:
            from langchain_openai import ChatOpenAI

            extra = {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}
            self._chat = ChatOpenAI(
                model=self.model,
                temperature=self.temperature,
                seed=self.seed,
                timeout=self.timeout,
                max_retries=3,
                **extra,
            )
        return self._chat

    def _key(self, messages: Sequence[BaseMessage], schema: type[BaseModel]) -> str:
        clave = {
            "model": self.model,
            "temperature": self.temperature,
            "seed": self.seed,
            "schema": schema.model_json_schema(),
            "messages": [(m.type, m.content) for m in messages],
        }
        if self.reasoning_effort:  # solo si aplica: las claves de los demás modelos no cambian
            clave["reasoning_effort"] = self.reasoning_effort
        return stable_hash(clave)

    def invoke(
        self, messages: Sequence[BaseMessage], schema: type[BaseModel], stage: str = "reason"
    ) -> LLMResult:
        key = self._key(messages, schema)
        t0 = time.perf_counter()
        cached = self.kv.get(key) if self.kv else None
        if cached is not None:
            data = json.loads(cached)
            result = _parse(data["text"], schema)
            result.cache_hit, result.usage = True, data["usage"]
            if data.get("error") and result.error is None:
                result.error = data["error"]
        elif es_claude(self.model):
            text, usage, error, parsed = self._invoke_claude(messages, schema)
            result = LLMResult(parsed=parsed, raw_text=text, error=error, usage=usage)
            if result.parsed is None and error is None:
                result = _parse(text, schema)
                result.usage = usage
            if self.kv:
                self.kv.set(
                    key, json.dumps({"text": text, "usage": usage, "error": error}).encode()
                )
        else:
            runnable = self._client().with_structured_output(
                schema, method="json_schema", strict=True, include_raw=True
            )
            out = runnable.invoke(list(messages))
            raw = out["raw"]
            text = raw.content if isinstance(raw.content, str) else json.dumps(raw.content)
            um = getattr(raw, "usage_metadata", None) or {}
            usage = {
                "input_tokens": um.get("input_tokens", 0),
                "output_tokens": um.get("output_tokens", 0),
                "cached_input_tokens": (um.get("input_token_details") or {}).get("cache_read", 0),
            }
            refusal = (raw.additional_kwargs or {}).get("refusal")
            error = refusal or (str(out["parsing_error"]) if out.get("parsing_error") else None)
            result = LLMResult(parsed=out.get("parsed"), raw_text=text, error=error, usage=usage)
            if result.parsed is None and error is None:
                result = _parse(text, schema)
                result.usage = usage
            if self.kv:
                self.kv.set(
                    key, json.dumps({"text": text, "usage": usage, "error": error}).encode()
                )
        _report(self.model, stage, result, (time.perf_counter() - t0) * 1000)
        return result


    # -- Claude (SDK oficial de Anthropic) --------------------------------------------------

    def _anthropic(self):
        if self._chat is None:
            import os

            import anthropic

            # La clave se carga en .env como CLAUDE_API_KEY (o ANTHROPIC_API_KEY, la estándar).
            api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY")
            self._chat = anthropic.Anthropic(api_key=api_key, timeout=self.timeout, max_retries=3)
        return self._chat

    def _invoke_claude(self, messages: Sequence[BaseMessage], schema: type[BaseModel]):
        """Salida estructurada con `messages.parse` (el SDK valida contra el esquema Pydantic).
        Las instrucciones fijas van en `system` con caché de prompt; el resto, como turnos."""
        system, turnos = [], []
        for m in messages:
            contenido = m.content if isinstance(m.content, str) else _bloques_claude(m.content)
            if m.type == "system":
                system.append(contenido if isinstance(contenido, str) else json.dumps(contenido))
            else:
                turnos.append({"role": "assistant" if m.type == "ai" else "user", "content": contenido})  # fmt: skip
        kwargs: dict = {
            "model": self.model,
            "max_tokens": CLAUDE_MAX_TOKENS,
            "system": [{"type": "text", "text": "\n\n".join(system), "cache_control": {"type": "ephemeral"}}],  # fmt: skip
            "messages": turnos,
            "output_format": schema,
        }
        if self.reasoning_effort:
            kwargs["output_config"] = {"effort": self.reasoning_effort}
        client = self._anthropic()
        if self.model in CLAUDE_CON_RESPALDO:
            r = client.beta.messages.parse(
                **kwargs, betas=["server-side-fallback-2026-07-01"], fallbacks="default"
            )
        else:
            r = client.messages.parse(**kwargs)
        u = r.usage
        usage = {
            "input_tokens": (u.input_tokens or 0) + (u.cache_read_input_tokens or 0) + (u.cache_creation_input_tokens or 0),  # fmt: skip
            "output_tokens": u.output_tokens or 0,
            "cached_input_tokens": u.cache_read_input_tokens or 0,
        }
        if getattr(r, "model", self.model) != self.model:
            usage["modelo_respaldo"] = r.model  # respondió el modelo de respaldo
        text = "".join(b.text for b in r.content if getattr(b, "type", None) == "text")
        if r.stop_reason == "refusal":
            cat = getattr(getattr(r, "stop_details", None), "category", None)
            return text, usage, f"refusal: {cat}", None
        if r.stop_reason == "max_tokens":
            return text, usage, "max_tokens", None
        return text, usage, None, getattr(r, "parsed_output", None)


def _bloques_claude(partes: list) -> list[dict]:
    """Partes de un mensaje en formato OpenAI/LangChain (texto e `image_url` con data URL en
    base64) → bloques de contenido de Anthropic. La imagen va antes del texto."""
    imagenes, textos = [], []
    for p in partes:
        if isinstance(p, str):
            textos.append({"type": "text", "text": p})
        elif p.get("type") == "text":
            textos.append({"type": "text", "text": p["text"]})
        elif p.get("type") == "image_url":
            url = p["image_url"]["url"] if isinstance(p["image_url"], dict) else p["image_url"]
            m = re.match(r"data:(image/[\w.+-]+);base64,(.*)", url, re.S)
            if not m:
                raise ValueError("Claude: solo imágenes en base64 (data URL)")
            imagenes.append({"type": "image", "source": {"type": "base64", "media_type": m[1], "data": m[2]}})  # fmt: skip
        else:
            raise ValueError(f"Claude: parte de mensaje no soportada: {p.get('type')}")
    return imagenes + textos


def _parse(text: str, schema: type[BaseModel]) -> LLMResult:
    try:
        return LLMResult(parsed=schema.model_validate_json(text), raw_text=text)
    except ValidationError as e:
        return LLMResult(parsed=None, raw_text=text, error=str(e))


def _report(model: str, stage: str, r: LLMResult, ms: float) -> None:
    tracker = current_tracker()
    if tracker is None:
        return
    tracker.add_llm(
        LLMCall(
            stage=stage,
            model=model,
            input_tokens=r.usage.get("input_tokens", 0),
            output_tokens=r.usage.get("output_tokens", 0),
            cached_input_tokens=r.usage.get("cached_input_tokens", 0),
            cache_hit=r.cache_hit,
            ms=ms,
        )
    )


class FakeStructuredLLM:
    """Doble de prueba: `responder(messages, schema) -> BaseModel | str` (str = JSON crudo,
    útil para simular salidas inválidas). Mismo contrato que `StructuredLLM`."""

    def __init__(self, responder: Callable[[Sequence[BaseMessage], type[BaseModel]], object]):
        self.responder = responder
        self.model = "fake-llm"
        self.calls: list[Sequence[BaseMessage]] = []

    def invoke(self, messages, schema, stage: str = "reason") -> LLMResult:
        self.calls.append(messages)
        out = self.responder(messages, schema)
        usage = {
            "input_tokens": sum(len(str(m.content)) // 4 for m in messages),
            "output_tokens": 50,
        }
        if isinstance(out, str):
            r = _parse(out, schema)
        else:
            r = LLMResult(parsed=out, raw_text=out.model_dump_json())
        r.usage = usage
        _report(self.model, stage, r, 0.0)
        return r
