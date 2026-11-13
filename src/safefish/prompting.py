"""Carga de prompts versionados (archivos `.md` con secciones `## system` y `## human`).

Las líneas que empiezan con `# ` son comentarios. El hash corto del archivo se registra en
`metrics.json` / manifest: cualquier cambio de texto cambia el hash. Para iterar se crea un
archivo nuevo (`*_v2.md`) y se apunta el YAML de la versión.
"""

from __future__ import annotations

import hashlib
import re
from functools import cache
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate


@cache
def load_prompt_file(path: Path) -> tuple[ChatPromptTemplate, str]:
    """Devuelve (template, hash corto del archivo)."""
    raw = path.read_text(encoding="utf-8")
    body = "\n".join(ln for ln in raw.splitlines() if not ln.startswith("# "))
    parts = re.split(r"^## (system|human)\s*$", body, flags=re.M)
    sections = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}
    template = ChatPromptTemplate.from_messages(
        [("system", sections["system"]), ("human", sections["human"])]
    )
    return template, hashlib.sha256(raw.encode()).hexdigest()[:12]
