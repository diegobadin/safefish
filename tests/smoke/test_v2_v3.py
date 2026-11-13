"""Smoke de punta a punta de la Fase 2 sin API: KB fixture normalizada por plantilla
(4 colecciones, 1 caso = 1 chunk), planner, verificador y arnés. No mide calidad."""

import json
from pathlib import Path

import pytest

from eval.golden import load_golden
from eval.run import run_eval
from safefish.config import load_config
from safefish.embeddings import make_embeddings
from safefish.ingest.build_kb import build_snapshot
from safefish.pipeline import SafeFishPipeline
from safefish.schemas import RawInput

from ..conftest import FIXTURES

REPO = Path(__file__).resolve().parents[2]


def _cfg(name):
    return load_config(REPO / "configs" / name)


@pytest.fixture
def kb_v2(tmp_snapshots):
    cfg = _cfg("v2-fixture.yaml")
    emb = make_embeddings(cfg.embeddings, fake=True)
    return build_snapshot(cfg, embeddings=emb), emb


def test_kb_v2_normalizada_por_plantilla_en_4_colecciones(kb_v2, tmp_snapshots):
    manifest, _ = kb_v2
    assert manifest["index_layout"] == "per_tipo"
    assert set(manifest["collections"]) == {"casos", "mensajes", "senales", "guias"}
    assert manifest["counts"]["chunks_por_tipo"] == {
        "caso": 8,
        "mensaje": 8,
        "guia": 3,
        "senal": 15,
    }
    assert manifest["n_chunks"] == manifest["n_docs"]  # 1 documento = 1 chunk
    assert manifest["normalizacion"]["metodo"] == {"template": 34}  # sin LLM
    assert manifest["normalizer_version"] == "template"
    assert manifest["cobertura"]["casos_por_modalidad"]["robo_cuenta_whatsapp"] == 1
    assert len(manifest["cobertura"]["modalidades_bajo_umbral"]) == 10  # mini-KB: 1 caso c/u
    assert manifest["auditoria_inyeccion"] == []
    rows = [json.loads(x) for x in (tmp_snapshots / "kb-fixture-v2" / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]  # fmt: skip
    caso = next(r for r in rows if r["id"] == "FX-C01")
    assert caso["texto"].startswith("Descripción:") and caso["metadata"]["canal_whatsapp"] is True


def test_pipeline_v2_recupera_por_coleccion_y_planifica_desde_guias(kb_v2, fake_llm):
    _, emb = kb_v2
    cfg = _cfg("v2-fixture.yaml")
    pipe = SafeFishPipeline.from_config(cfg, llm=fake_llm, embeddings=emb)
    res = pipe.run(RawInput(texto="Hola! te mandé un código por error, pasame el código porfa", canal="whatsapp", situacion="compartio_codigo"))  # fmt: skip
    assert res.error is None, res.error
    b = res.bundle
    assert (len(b.casos), len(b.mensajes), len(b.senales)) == (4, 6, 6)
    assert {g.id for g in b.guias} == {g["id"] for g in res.plan_info["guias"]}
    assert res.plan_info["guias"][0] == {"id": "FX-G01", "via": "exacta"}
    assert res.output.plan and all(p.guide_id in {g.id for g in b.guias} for p in res.output.plan)
    assert res.output.meta.modelos["planner"] == "fake-llm"
    assert res.output_pre_verify is None and res.verificacion == []  # V2 no verifica


def test_pipeline_v3_verifica_antes_y_despues_del_plan(kb_v2, fake_llm):
    _, emb = kb_v2
    pipe = SafeFishPipeline.from_config(_cfg("v3-fixture.yaml"), llm=fake_llm, embeddings=emb)
    res = pipe.run(RawInput(texto="Che, ¿venís a cenar el sábado?"))
    assert res.error is None
    assert res.output.veredicto.value == "parece_legitimo"
    assert res.plan_info["situacion_asumida"] is True
    assert res.output_pre_verify is not None
    assert "verify" in res.usage["latencias_ms"] and "plan" in res.usage["latencias_ms"]


def test_layout_del_config_debe_coincidir_con_el_snapshot(kb_v2):
    _, emb = kb_v2
    cfg = _cfg("v2-fixture.yaml")
    cfg.index.layout = "single"
    with pytest.raises(ValueError, match="layout"):
        SafeFishPipeline.from_config(cfg, embeddings=emb)


def test_arnes_v3(kb_v2, tmp_path, fake_llm):
    _, emb = kb_v2
    cfg = _cfg("v3-fixture.yaml")
    golden = FIXTURES / "golden" / "dev.jsonl"
    cases = load_golden(golden)
    pipe = SafeFishPipeline.from_config(cfg, llm=fake_llm, embeddings=emb)
    out_dir, metrics = run_eval(cfg, cases, pipe, tmp_path / "results", split="dev", golden_file=golden, workers=2)  # fmt: skip
    t = metrics["task"]
    assert t["n_errores"] == 0
    assert t["grounding_citas"] == 1.0  # invariante post-R1 (gate G2)
    assert t["contactos_no_respaldados"] == 0
    assert "grounding_citas_pre_r1" in t and "eventos_verificador" in t
    assert t["pasos_con_guide_id"] == 1.0
    assert metrics["retrieval"]["k"] == 4
    assert metrics["meta"]["verifier"] == ["R1", "R2", "R3", "R4", "R5"]
    assert metrics["meta"]["planner"]["mode"] == "llm"
    row = json.loads((out_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert {"pred_pre_verify", "verificacion", "plan_info", "contexto_guias_por_id"} <= set(row)


def test_arnes_solo_retrieval_no_llama_al_razonador(kb_v2, tmp_path):
    from safefish.llm import FakeStructuredLLM

    def no_llamar(messages, schema):
        raise AssertionError("el barrido de solo retrieval no debe generar")

    _, emb = kb_v2
    cfg = _cfg("v2-fixture.yaml")
    golden = FIXTURES / "golden" / "dev.jsonl"
    pipe = SafeFishPipeline.from_config(cfg, llm=FakeStructuredLLM(no_llamar), embeddings=emb)
    _, metrics = run_eval(
        cfg, load_golden(golden), pipe, tmp_path / "r", split="dev", golden_file=golden,
        retrieval_only=True,
    )  # fmt: skip
    assert metrics["modo"] == "retrieval_only" and metrics["task"] == {}
    assert metrics["retrieval"]["k"] == 4 and metrics["retrieval"]["n_eval_casos"] > 0
