"""OCR con cajas (M-EXT): preprocesamiento de imagen + `pytesseract.image_to_data`.

Preprocesamiento (Clase 10): escala de grises, *upscale* de capturas chicas (Tesseract rinde
mejor con letras de ≥ 20 px) y binarización por umbral de Otsu. Las cajas se devuelven en
píxeles de la imagen **original** (se deshace el escalado).

Tesseract es un binario externo. Si no está instalado, `OCRUnavailable`: el extractor sigue
sin OCR y las señales se muestran sin recuadro.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from safefish.schemas import OcrWord

MIN_WIDTH = 1400


class OCRUnavailable(RuntimeError):
    pass


@dataclass
class Preprocessed:
    image: object  # PIL.Image
    scale: float


def otsu_threshold(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(float)
    total = gray.size
    sum_all = float(np.dot(np.arange(256), hist))
    w_b = sum_b = 0.0
    best_t, best_var = 127, -1.0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b, m_f = sum_b / w_b, (sum_all - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > best_var:
            best_var, best_t = var, t
    return best_t


def preprocess(image, enabled: bool = True) -> Preprocessed:
    from PIL import Image

    img = image.convert("L")
    if not enabled:
        return Preprocessed(img, 1.0)
    scale = 1.0
    if img.width < MIN_WIDTH:
        scale = MIN_WIDTH / img.width
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    arr = np.asarray(img)
    t = otsu_threshold(arr)
    binary = np.where(arr > t, 255, 0).astype(np.uint8)
    # Modo oscuro (texto claro sobre fondo oscuro): Tesseract espera texto oscuro.
    if (binary == 0).mean() > 0.5:
        binary = 255 - binary
    return Preprocessed(Image.fromarray(binary), scale)


class TesseractOCR:
    def __init__(self, lang: str = "spa", min_conf: float = 30.0, preprocess: bool = True):
        self.lang, self.min_conf, self.do_preprocess = lang, min_conf, preprocess

    def words(self, image_path: str) -> list[OcrWord]:
        try:
            import pytesseract
            from PIL import Image
        except ImportError as e:  # pragma: no cover - dependencias del proyecto
            raise OCRUnavailable(str(e)) from e
        with Image.open(image_path) as im:
            pre = preprocess(im, self.do_preprocess)
        try:
            data = pytesseract.image_to_data(
                pre.image, lang=self.lang, output_type=pytesseract.Output.DICT
            )
        except (pytesseract.TesseractNotFoundError, pytesseract.TesseractError) as e:
            raise OCRUnavailable(f"Tesseract no disponible: {e}") from e
        return words_from_data(data, pre.scale, self.min_conf)


def words_from_data(data: dict, scale: float, min_conf: float) -> list[OcrWord]:
    out = []
    for i, text in enumerate(data["text"]):
        text = (text or "").strip()
        if not text or float(data["conf"][i]) < min_conf:
            continue
        x, y, w, h = (data[k][i] for k in ("left", "top", "width", "height"))
        bbox = (round(x / scale), round(y / scale), round((x + w) / scale), round((y + h) / scale))
        out.append(OcrWord(text=text, bbox=bbox))
    return out
