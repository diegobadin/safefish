"""Smoke de punta a punta sin API: KB fixture con embeddings fake + LLM heurístico.

Verifica que el pipeline completo no crashea, que el JSON es válido y que el arnés escribe
todos los artefactos con las métricas esperadas. No mide calidad.
"""

import json
from pathlib import Path

from eval.compare import compare
from eval.golden import check_golden, load_golden
from eval.run import run_eval
from safefish.config import load_config
from safefish.embeddings import make_embeddings
from safefish.ingest.build_kb import build_snapshot
from safefish.pipeline import SafeFishPipeline
from safefish.schemas import RawInput, RetrievalBundle

from ..conftest import FIXTURES

REPO = Path(__file__).resolve().parents[2]


def _cfg(name):
    return load_config(REPO / "configs" / name)


def test_kb_fixture_y_pipeline_v1(tmp_snapshots, fake_llm):
    cfg = _cfg("v1-fixture.yaml")
    emb = make_embeddings(cfg.embeddings, fake=True)
    manifest = build_snapshot(cfg, embeddings=emb)
    assert manifest["counts"]["docs_por_tipo"] == {"caso": 8, "mensaje": 8, "guia": 3, "senal": 15}
    assert (tmp_snapshots / "kb-fixture" / "manifest.json").exists()
    chunks = (
        (tmp_snapshots / "kb-fixture" / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    )
    assert "FX-C01" in {json.loads(line)["metadata"]["doc_id"] for line in chunks}

    pipe = SafeFishPipeline.from_config(cfg, llm=fake_llm, embeddings=emb)
    res = pipe.run(
        RawInput(
            texto="Hola! te mandé un código por error, pasame el código porfa", canal="whatsapp"
        )
    )
    assert res.error is None
    assert res.output.veredicto.value in {"muy_sospechoso", "estafa_confirmada"}
    assert len(res.bundle.all_items()) == cfg.retrieval.top_k
    assert res.output.meta.kb_snapshot == "kb-fixture"
    assert res.output.meta.tokens["embed"] > 0


def test_v0_no_recupera(fake_llm):
    pipe = SafeFishPipeline.from_config(_cfg("v0.yaml"), llm=fake_llm)
    res = pipe.run(RawInput(texto="Che, ¿venís a cenar?"))
    assert res.bundle == RetrievalBundle(queries=[])
    assert res.output.veredicto.value == "parece_legitimo"
    assert "(sin evidencia recuperada)" in fake_llm.calls[0][1].content


def test_reintento_ante_json_invalido(fake_llm):
    from safefish.llm import FakeStructuredLLM

    from ..conftest import heuristic_responder

    calls = {"n": 0}

    def flaky(messages, schema):
        calls["n"] += 1
        return (
            '{"veredicto": "inventado"}'
            if calls["n"] == 1
            else heuristic_responder(messages, schema)
        )

    pipe = SafeFishPipeline.from_config(_cfg("v0.yaml"), llm=FakeStructuredLLM(flaky))
    res = pipe.run(RawInput(texto="hola"))
    assert res.error is None and res.reintentos_parseo == 1


def test_arnes_y_compare(tmp_snapshots, tmp_path, fake_llm):
    golden_dir = FIXTURES / "golden"
    cases = load_golden(golden_dir / "dev.jsonl")
    v1 = _cfg("v1-fixture.yaml")
    emb = make_embeddings(v1.embeddings, fake=True)
    manifest = build_snapshot(v1, embeddings=emb)
    assert not [p for p in check_golden(golden_dir, manifest) if "LEAKAGE" in p]

    runs = []
    for cfg in (_cfg("v0.yaml"), v1):
        pipe = SafeFishPipeline.from_config(cfg, llm=fake_llm, embeddings=emb)
        out_dir, metrics = run_eval(cfg, cases, pipe, tmp_path / "results", split="dev",
                                    golden_file=golden_dir / "dev.jsonl", workers=2)  # fmt: skip
        for f in (
            "predictions.jsonl",
            "retrieval.jsonl",
            "metrics.json",
            "costs.json",
            "config.yaml",
        ):
            assert (out_dir / f).exists(), f
        assert metrics["task"]["n"] == len(cases)
        assert metrics["task"]["n_errores"] == 0
        assert metrics["task"]["grounding_citas"] == 1.0
        runs.append(out_dir)
    assert metrics["retrieval"]["n_eval_casos"] > 0

    md, stats = compare(runs[0], runs[1])
    assert "## Significancia" in md and "p_value" in stats["mcnemar_binario"]
