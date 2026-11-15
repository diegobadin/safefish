"""Extractor de texto (M-EXT, entrada `texto`). Las capturas: `extract/capture.py`."""

from __future__ import annotations

from safefish.schemas import Entidades, ExtractedMessage, RawInput, Segmento
from safefish.textutils import (
    RE_EMAIL,
    RE_MONTO,
    find_cbu_alias,
    find_links,
    find_phones,
    normalize_unicode,
)


def extract_entities(texto: str) -> Entidades:
    return Entidades(
        links=find_links(texto),
        telefonos=find_phones(texto),
        montos=[m.group(0) for m in RE_MONTO.finditer(texto)],
        cbu_alias=find_cbu_alias(texto),
        emails=RE_EMAIL.findall(texto),
    )


class TextExtractor:
    """El texto normalizado es el canónico: es contra lo que se verifican las citas."""

    def extract(self, raw: RawInput) -> ExtractedMessage:
        if not raw.texto:
            raise ValueError(
                "TextExtractor solo procesa texto: las capturas van por MultimodalExtractor"
            )
        texto = normalize_unicode(raw.texto)
        return ExtractedMessage(
            texto=texto,
            canal=raw.canal,
            segmentos=[Segmento(emisor="remitente", texto=texto)],
            entidades=extract_entities(texto),
            ocr_words=None,
            origen="texto",
            contexto=normalize_unicode(raw.contexto) if raw.contexto else None,
        )
