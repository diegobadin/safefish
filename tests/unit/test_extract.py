"""M-EXT para capturas: alineación cita → bbox, OCR, extractor multimodal y anotación."""

import numpy as np
import pytest
from PIL import Image

from safefish.extract.align import align_cita, iou
from safefish.extract.capture import CaptureError, MultimodalExtractor
from safefish.extract.ocr import OCRUnavailable, otsu_threshold, preprocess, words_from_data
from safefish.extract.vision import SegmentoLLM, TranscripcionLLM, VisionTranscriber
from safefish.llm import FakeStructuredLLM
from safefish.postprocess.annotate import annotate
from safefish.schemas import OcrWord, RawInput, RunMeta, SafeFishOutput, SenalDetectada


def _words(text, y=10):
    return [OcrWord(text=w, bbox=(20 * i, y, 20 * i + 18, y + 12)) for i, w in enumerate(text.split())]  # fmt: skip


def test_align_cita_con_errores_de_ocr():
    words = _words("Hola pasame el c0digo que te lleg6 porfa") + _words("otra linea", y=40)
    bbox = align_cita("pasame el código que te llegó", words)
    assert bbox == (20, 10, 20 * 6 + 18, 22)
    assert align_cita("transferí a este alias", words) is None


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_words_from_data_deshace_el_escalado_y_filtra_confianza():
    data = {"text": ["Hola", "", "ruido"], "conf": [90, -1, 10], "left": [20, 0, 0], "top": [10, 0, 0], "width": [40, 0, 0], "height": [20, 0, 0]}  # fmt: skip
    assert words_from_data(data, 2.0, 30.0) == [OcrWord(text="Hola", bbox=(10, 5, 30, 15))]


def test_preprocess_binariza_y_reescala():
    arr = np.full((50, 100), 230, dtype=np.uint8)
    arr[20:30, 10:90] = 20  # "texto" oscuro
    assert 20 <= otsu_threshold(arr) < 230
    pre = preprocess(Image.fromarray(arr))
    out = np.asarray(pre.image)
    assert pre.scale > 1 and set(np.unique(out)) <= {0, 255}
    assert (out == 255).mean() > 0.5  # fondo blanco, texto negro


class _FakeOCR:
    def __init__(self, words=None, fail=False):
        self._words, self.fail, self.calls = words or [], fail, 0

    def words(self, path):
        self.calls += 1
        if self.fail:
            raise OCRUnavailable("sin tesseract")
        return self._words


def _vision(segmentos, legible=True, canal="whatsapp"):
    t = TranscripcionLLM(canal=canal, remitente=None, segmentos=segmentos, legible=legible)
    return VisionTranscriber(FakeStructuredLLM(lambda m, s: t))


@pytest.fixture
def captura(tmp_path):
    p = tmp_path / "cap.png"
    Image.new("RGB", (300, 120), "white").save(p)
    return p


def test_extractor_multimodal_usa_segmentos_del_remitente(captura):
    segs = [
        SegmentoLLM(emisor="remitente", texto="Hola! pasame el código de 6 dígitos"),
        SegmentoLLM(emisor="usuario", texto="¿quién sos?"),
        SegmentoLLM(emisor="remitente", texto="Es urgente, transferí a bit.ly/abc"),
    ]
    ocr = _FakeOCR(_words("Hola! pasame el código"))
    m = MultimodalExtractor(_vision(segs), ocr).extract(RawInput(texto="(ignorado)", imagen_path=str(captura)))  # fmt: skip
    assert m.origen == "captura" and m.canal.value == "whatsapp"
    assert m.texto == "Hola! pasame el código de 6 dígitos\nEs urgente, transferí a bit.ly/abc"
    assert [s.emisor for s in m.segmentos] == ["remitente", "usuario", "remitente"]
    assert m.entidades.links == ["bit.ly/abc"] and m.ocr_words


def test_extractor_sigue_sin_ocr_si_no_hay_tesseract(captura):
    ocr = _FakeOCR(fail=True)
    ex = MultimodalExtractor(_vision([SegmentoLLM(emisor="remitente", texto="hola")]), ocr)
    raw = RawInput(imagen_path=str(captura))
    assert ex.extract(raw).ocr_words is None
    ex.extract(raw)
    assert ocr.calls == 1  # no reintenta en cada caso


def test_captura_ilegible_o_inexistente(captura, tmp_path):
    with pytest.raises(CaptureError):
        MultimodalExtractor(_vision([], legible=False)).extract(RawInput(imagen_path=str(captura)))
    with pytest.raises(CaptureError):
        MultimodalExtractor(_vision([])).extract(RawInput(imagen_path=str(tmp_path / "no.png")))


def test_texto_plano_no_llama_a_vision():
    vision = _vision([])
    m = MultimodalExtractor(vision).extract(RawInput(texto="hola  mundo"))
    assert m.texto == "hola mundo" and not vision.llm.calls


def test_mensajes_de_vision_llevan_la_imagen(captura):
    msgs = _vision([]).messages(str(captura))
    content = msgs[1].content
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_annotate_dibuja_solo_senales_verificadas_con_bbox(captura, tmp_path):
    out = SafeFishOutput(
        veredicto="muy_sospechoso",
        modalidad="no_determinada",
        senales=[
            SenalDetectada(signal_id="pedido_codigo_verificacion", cita="x", explicacion="x", bbox=(10, 10, 100, 30), verificada=True),
            SenalDetectada(signal_id="urgencia_temporal", cita="y", explicacion="y", bbox=None, verificada=True),
        ],
        meta=RunMeta(run_id="r", config_hash="h"),
    )  # fmt: skip
    salida = tmp_path / "out.png"
    assert annotate(captura, out, salida) == 1
    assert (np.asarray(Image.open(salida)) != 255).any()
