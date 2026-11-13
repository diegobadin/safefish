"""Alineación cita → bbox sobre las palabras OCR de una captura (M-EXT, §2.3).

La cita sale del texto canónico (transcripción del LLM con visión), no del OCR, así que se
busca la ventana de palabras OCR más parecida (fuzzy) y se devuelve la unión de sus cajas.
Si no alinea por encima del umbral, la señal se muestra sin recuadro.
"""

from __future__ import annotations

from difflib import SequenceMatcher

from safefish.schemas import BBox, OcrWord
from safefish.textutils import normalize_for_match


def union_bbox(boxes: list[BBox]) -> BBox:
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def align_cita(cita: str, words: list[OcrWord], threshold: float = 0.8) -> BBox | None:
    target = normalize_for_match(cita)
    if not target or not words:
        return None
    norm = [normalize_for_match(w.text) for w in words]
    idx = [i for i, t in enumerate(norm) if t]  # palabras OCR con contenido
    n = len(target.split())
    best, best_span = 0.0, None
    for size in range(max(1, n - 2), n + 3):
        for start in range(0, max(1, len(idx) - size + 1)):
            span = idx[start : start + size]
            if not span:
                continue
            window = " ".join(norm[i] for i in span)
            score = SequenceMatcher(None, target, window, autojunk=False).ratio()
            if score > best:
                best, best_span = score, span
    if best < threshold or best_span is None:
        return None
    return union_bbox([words[i].bbox for i in best_span])


def iou(a: BBox, b: BBox) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area if area > 0 else 0.0
