"""Los bundles de ejemplo respetan los contratos (los usa Track B sin la KB real)."""

import json

import pytest

from safefish.schemas import ExtractedMessage, RetrievalBundle

from ..conftest import FIXTURES

BUNDLES = sorted((FIXTURES / "bundles").glob("*.json"))


def test_hay_cinco_bundles():
    assert len(BUNDLES) == 5


@pytest.mark.parametrize("path", BUNDLES, ids=lambda p: p.stem)
def test_bundle_valida(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    ExtractedMessage.model_validate(data["mensaje"])
    b = RetrievalBundle.model_validate(data["bundle"])
    assert b.all_items() and all(it.metadata.get("doc_id") for it in b.all_items())
