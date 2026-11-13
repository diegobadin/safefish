from langchain_core.documents import Document

from safefish.config import ChunkingCfg
from safefish.ingest.chunk import chunk_documents, scalarize_metadata
from safefish.ingest.clean import dedupe_exact, dedupe_near, filter_language, strip_boilerplate


def _doc(text, **md):
    return Document(page_content=text, metadata={"source_id": "s", "doc_tipo": "mensaje", **md})


def test_boilerplate_quita_navegacion_y_lineas_repetidas_del_sitio():
    pie = "Banco Ejemplo S.A. - Av. Siempreviva 742"
    docs = [_doc(f"Menú\nAlerta {i}: no compartas tu clave.\n{pie}\nTodos los derechos reservados 2026") for i in range(3)]  # fmt: skip
    out = strip_boilerplate(docs)
    assert all(d.page_content == f"Alerta {i}: no compartas tu clave." for i, d in enumerate(out))


def test_dedupe_exacto_ignora_mayusculas_tildes_y_espacios():
    keep, drop = dedupe_exact(
        [_doc("Pasame el CÓDIGO ya"), _doc("pasame  el codigo ya"), _doc("otro")]
    )
    assert len(keep) == 2 and len(drop) == 1


def test_dedupe_cercano_por_plantilla():
    base = "Hola ma, este es mi número nuevo, agendalo. Necesito que me transfieras hoy {} al alias de un amigo."
    docs = [_doc(base.format("85.000"), doc_id="a"), _doc(base.format("90.000"), doc_id="b"),
            _doc("Tu pedido está en camino, seguilo desde la app oficial.", doc_id="c")]  # fmt: skip
    keep, drop = dedupe_near(docs, threshold=0.9)
    assert [d.metadata["doc_id"] for d in keep] == ["a", "c"]
    assert drop[0].metadata["duplicado_de"] == "a"


def test_dedupe_cercano_no_compara_entre_tipos():
    t = "Ingresá a bancoejemplo-validacion.com para validar tus datos antes de 24 horas."
    keep, _ = dedupe_near([_doc(t, doc_id="a"), _doc(t + " ", doc_id="b", doc_tipo="caso")], 0.9)
    assert len(keep) == 2


def test_filtro_de_idioma():
    es = _doc(
        "Este mensaje es una alerta sobre estafas bancarias que circulan por WhatsApp en Argentina."
    )
    en = _doc(
        "This message is an alert about banking scams circulating on WhatsApp in several countries."
    )
    corto = _doc("ok gracias")
    keep, drop = filter_language([es, en, corto], ["es"], min_chars=60)
    assert keep == [es, corto] and drop == [en]


def test_chunking_recursivo_ids_deterministicos_y_overlap():
    text = " ".join(f"Oración número {i} sobre una estafa." for i in range(80))
    cfg = ChunkingCfg(chunk_size=300, chunk_overlap=45)
    chunks = chunk_documents([_doc(text, doc_id="D1")], cfg)
    assert len(chunks) > 3
    assert [c.id for c in chunks] == [f"D1#c{i:02d}" for i in range(len(chunks))]
    assert all(len(c.page_content) <= 300 for c in chunks)
    assert all(c.page_content.endswith(".") for c in chunks[:-1])  # corta en fin de oración
    assert chunks == chunk_documents([_doc(text, doc_id="D1")], cfg)


def test_scalarize_metadata_para_chroma():
    md = scalarize_metadata({"canal": ["whatsapp", "sms"], "x": None, "tags": ["a", "b"], "n": 3})
    assert md == {"canal_whatsapp": True, "canal_sms": True, "canal_email": False,
                  "canal": "whatsapp,sms", "tags": "a,b", "n": 3}  # fmt: skip
