"""Genera los 5 `RetrievalBundle` de ejemplo de tests/fixtures/bundles/ (tarea de F0).

    uv run python scripts/make_fixture_bundles.py

Usa el snapshot `kb-fixture` (construir antes con `make kb CONFIG=configs/v1-fixture.yaml`)
y el retriever denso de V1. Sirven para desarrollar generación, planner y verificador sin
depender de la KB real.
"""

from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

from safefish.config import REPO_ROOT, load_config
from safefish.embeddings import make_embeddings
from safefish.extract.text import TextExtractor
from safefish.ingest.index import open_index, read_manifest
from safefish.retrieval.dense import DenseRetriever
from safefish.schemas import Canal, RawInput

EJEMPLOS = {
    "b01_robo_wa": ("Hola, te llegó un código por SMS? me lo pasás que es para un sorteo", Canal.WHATSAPP),
    "b02_falso_familiar": ("Hola ma, cambié de número. Me transferís 60.000 a este alias? te devuelvo mañana", Canal.WHATSAPP),
    "b03_phishing": ("Banco Ejemplo: su cuenta será bloqueada. Valide sus datos en bancoejemplo-clientes.com", Canal.SMS),
    "b04_legitimo_codigo": ("Tu código de verificación es <CODIGO>. No lo compartas con nadie.", Canal.SMS),
    "b05_inversion": ("Invertí con nosotros y ganá 25% mensual garantizado, cupos limitados solo por hoy", Canal.WHATSAPP),
}  # fmt: skip


def main() -> None:
    load_dotenv()
    cfg = load_config(REPO_ROOT / "configs" / "v1-fixture.yaml")
    emb = make_embeddings(cfg.embeddings)
    manifest = read_manifest(cfg.snapshot_dir)
    store = open_index(cfg.snapshot_dir, emb, cfg.embeddings.model)
    retriever = DenseRetriever(store, cfg.retrieval.top_k, manifest["collection"])
    out = REPO_ROOT / "tests" / "fixtures" / "bundles"
    out.mkdir(parents=True, exist_ok=True)
    for name, (texto, canal) in EJEMPLOS.items():
        m = TextExtractor().extract(RawInput(texto=texto, canal=canal))
        b = retriever.retrieve(m)
        b.timings_ms = {}  # no determinístico
        payload = {"mensaje": m.model_dump(mode="json"), "bundle": b.model_dump(mode="json")}
        Path(out / f"{name}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(name, [it.metadata.get("doc_id") for it in b.all_items()])


if __name__ == "__main__":
    main()
