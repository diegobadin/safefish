"""Extractor multimodal (M-EXT): texto plano o captura de pantalla.

- Texto: `TextExtractor` (normalización + entidades por regex).
- Captura: LLM con visión → segmentos por emisor y texto canónico; OCR → palabras con bbox
  (para ubicar las citas en la imagen); entidades por regex sobre el texto canónico.

Si llegan texto e imagen a la vez, manda la **imagen** (el texto de un caso de golden con
captura es la transcripción de referencia, no la entrada).
"""

from __future__ import annotations

import logging

from safefish.config import resolve_path
from safefish.extract.ocr import OCRUnavailable, TesseractOCR
from safefish.extract.text import TextExtractor, extract_entities
from safefish.extract.vision import VisionTranscriber
from safefish.schemas import Canal, ExtractedMessage, RawInput, Segmento
from safefish.textutils import normalize_unicode

log = logging.getLogger(__name__)


class CaptureError(ValueError):
    pass


class MultimodalExtractor:
    def __init__(
        self,
        vision: VisionTranscriber | None = None,
        ocr: TesseractOCR | None = None,
        text: TextExtractor | None = None,
    ):
        self.vision, self.ocr, self.text = vision, ocr, text or TextExtractor()
        self.ocr_disponible: bool | None = None

    def extract(self, raw: RawInput) -> ExtractedMessage:
        if not raw.imagen_path:
            return self.text.extract(raw)
        if self.vision is None:
            raise CaptureError("Entrada por captura sin extractor de visión configurado")
        path = resolve_path(raw.imagen_path)
        if not path.exists():
            raise CaptureError(f"No existe la captura {raw.imagen_path}")

        t = self.vision.transcribe(str(path))
        segmentos = [
            Segmento(emisor=s.emisor, texto=normalize_unicode(s.texto))
            for s in t.segmentos
            if s.texto.strip()
        ]
        remitente = [s.texto for s in segmentos if s.emisor == "remitente"]
        texto = "\n".join(remitente or [s.texto for s in segmentos])
        if not t.legible or not texto.strip():
            raise CaptureError("La captura no tiene texto legible")
        canal = raw.canal or (Canal(t.canal) if t.canal != "desconocido" else None)

        ocr_words = None
        if self.ocr is not None and self.ocr_disponible is not False:
            try:
                ocr_words = self.ocr.words(str(path))
                self.ocr_disponible = True
            except OCRUnavailable as e:
                self.ocr_disponible = False
                log.warning("OCR deshabilitado: %s (las señales irán sin recuadro)", e)

        return ExtractedMessage(
            texto=texto,
            canal=canal,
            segmentos=segmentos,
            entidades=extract_entities(texto),
            ocr_words=ocr_words,
            origen="captura",
        )
