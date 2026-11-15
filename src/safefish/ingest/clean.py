"""Limpieza de documentos crudos (M-ING.2): boilerplate, idioma y deduplicación.

Todas las funciones son puras sobre `list[Document]` y devuelven también qué descartaron,
para que el manifest del snapshot documente cada filtro (data card).
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence

import numpy as np
from langchain_core.documents import Document

from safefish.textutils import normalize_for_match

# ---------------------------------------------------------------------------
# Boilerplate
# ---------------------------------------------------------------------------

BOILERPLATE = re.compile(
    r"(?i)^(?:"
    r".*\bcookies?\b.*|.*todos los derechos reservados.*|compartir(?: en)?\b.*|seguinos\b.*"
    r"|suscrib[ií]te\b.*|iniciar sesi[oó]n|ingres[aá]|men[uú]|ir al contenido.*|buscar"
    r"|pol[ií]tica de privacidad.*|t[eé]rminos y condiciones.*|volver(?: arriba)?|inicio"
    r"|whatsapp|facebook|instagram|twitter|x|linkedin|youtube|tiktok"
    r")$"
)


def strip_boilerplate(
    docs: Sequence[Document], min_group: int = 3, max_line_len: int = 200
) -> list[Document]:
    """Quita líneas de navegación conocidas y las que se repiten en ≥ 50 % de los documentos
    de una misma fuente (encabezados y pies de un sitio)."""
    by_source: dict[str, list[int]] = defaultdict(list)
    for i, d in enumerate(docs):
        by_source[d.metadata.get("source_id", "")].append(i)

    repeated: dict[str, set[str]] = {}
    for src, idxs in by_source.items():
        if len(idxs) < min_group:
            continue
        freq = Counter(
            line
            for i in idxs
            for line in {ln.strip() for ln in docs[i].page_content.split("\n")}
            if line and len(line) <= max_line_len
        )
        repeated[src] = {ln for ln, c in freq.items() if c >= max(2, len(idxs) / 2)}

    out = []
    for d in docs:
        rep = repeated.get(d.metadata.get("source_id", ""), set())
        lines = [
            ln
            for ln in d.page_content.split("\n")
            if not BOILERPLATE.match(ln.strip()) and ln.strip() not in rep
        ]
        out.append(Document(page_content="\n".join(lines).strip(), metadata=dict(d.metadata)))
    return out


# ---------------------------------------------------------------------------
# Idioma
# ---------------------------------------------------------------------------


def detect_language(text: str) -> str | None:
    from langdetect import DetectorFactory, LangDetectException, detect

    DetectorFactory.seed = 0  # langdetect es no determinístico sin seed
    try:
        return detect(text)
    except LangDetectException:
        return None


def filter_language(
    docs: Sequence[Document], languages: Sequence[str], min_chars: int = 60
) -> tuple[list[Document], list[Document]]:
    """Conserva documentos en `languages`. Los textos cortos no se filtran (detección poco
    confiable). Devuelve (conservados, descartados)."""
    keep, drop = [], []
    for d in docs:
        text = d.page_content
        if len(text) < min_chars:
            keep.append(d)
            continue
        lang = detect_language(text)
        d.metadata["idioma"] = lang
        (keep if lang in languages else drop).append(d)
    return keep, drop


# ---------------------------------------------------------------------------
# Deduplicación
# ---------------------------------------------------------------------------


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize_for_match(text).encode()).hexdigest()


def dedupe_exact(
    docs: Sequence[Document], group_key: str = "doc_tipo"
) -> tuple[list[Document], list[Document]]:
    """Descarta duplicados exactos dentro de cada `group_key` (como `dedupe_near`): una misma
    página puede ser fuente de un caso y de una guía."""
    seen: set[tuple[str, str]] = set()
    keep, drop = [], []
    for d in docs:
        key = (str(d.metadata.get(group_key)), content_hash(d.page_content))
        (drop if key in seen else keep).append(d)
        seen.add(key)
    return keep, drop


def tfidf_vectors(texts: Sequence[str]):
    """Matriz TF-IDF **sparse** (filas con norma L2): con miles de mensajes y n-gramas de
    caracteres, la versión densa no entra en memoria."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), preprocessor=normalize_for_match)
    return vec.fit_transform(texts)


def near_duplicate_pairs(vectors, threshold: float) -> list[tuple[int, int, float]]:
    """Pares (i, j, coseno) con i < j y coseno ≥ threshold. Acepta matrices densas o sparse."""
    if vectors.shape[0] < 2:
        return []
    if hasattr(vectors, "toarray"):  # sparse: normalizar por fila sin densificar
        from sklearn.preprocessing import normalize

        v = normalize(vectors)
        sims = (v @ v.T).toarray()
    else:
        v = vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)
        sims = v @ v.T
    i, j = np.where(np.triu(sims, k=1) >= threshold)
    return [(int(a), int(b), float(sims[a, b])) for a, b in zip(i, j, strict=True)]


def dedupe_near(
    docs: Sequence[Document],
    threshold: float,
    vectorize: Callable[[Sequence[str]], np.ndarray] = tfidf_vectors,
    group_key: str = "doc_tipo",
) -> tuple[list[Document], list[Document]]:
    """Descarta casi-duplicados (coseno ≥ threshold) dentro de cada `group_key`.
    Se conserva el primero según el orden de las fuentes (prioridad de curaduría)."""
    groups: dict[str, list[int]] = defaultdict(list)
    for i, d in enumerate(docs):
        groups[str(d.metadata.get(group_key))].append(i)

    dropped: set[int] = set()
    for idxs in groups.values():
        if len(idxs) < 2:
            continue
        vecs = vectorize([docs[i].page_content for i in idxs])
        for a, b, sim in near_duplicate_pairs(vecs, threshold):
            ia, ib = idxs[a], idxs[b]
            if ia not in dropped and ib not in dropped:
                dropped.add(ib)
                docs[ib].metadata["duplicado_de"] = docs[ia].metadata.get("doc_id", str(ia))
                docs[ib].metadata["duplicado_sim"] = round(sim, 4)
    keep = [d for i, d in enumerate(docs) if i not in dropped]
    drop = [d for i, d in enumerate(docs) if i in dropped]
    return keep, drop
