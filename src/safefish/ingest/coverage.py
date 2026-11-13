"""Panel de cobertura de la KB (§2.3) y auditoría de inyección indirecta (§5.4).

    python -m safefish.ingest.coverage --config configs/v2.yaml

Cobertura = casos por modalidad × canal. Umbral mínimo: `ingest.min_casos_por_modalidad`
(≥ 5); por debajo, hay que buscar más fuentes antes de medir V2.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable

from safefish.schemas import Canal, DocTipo, Modalidad

CANALES = [c.value for c in Canal]


def _canales(md: dict) -> list[str]:
    raw = str(md.get("canal") or "multi")
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    return CANALES if "multi" in vals or not vals else [v for v in vals if v in CANALES]


def coverage(chunks_md: Iterable[dict], min_casos: int = 5) -> dict:
    """`chunks_md`: metadata de los chunks del snapshot. Cuenta documentos (no chunks) de
    tipo caso por modalidad y por (modalidad, canal); multicanal suma en todos los canales."""
    seen: set[str] = set()
    por_mod: Counter = Counter()
    por_mod_canal: dict[str, Counter] = defaultdict(Counter)
    for md in chunks_md:
        if md.get("doc_tipo") != DocTipo.CASO or md.get("doc_id") in seen:
            continue
        seen.add(md["doc_id"])
        mod = str(md.get("modalidad"))
        por_mod[mod] += 1
        for c in _canales(md):
            por_mod_canal[mod][c] += 1
    modalidades = [m.value for m in Modalidad if m is not Modalidad.NO_DETERMINADA]
    return {
        "min_casos_por_modalidad": min_casos,
        "casos_por_modalidad": {m: por_mod.get(m, 0) for m in modalidades},
        "casos_por_modalidad_canal": {
            m: {c: por_mod_canal[m].get(c, 0) for c in CANALES} for m in modalidades
        },
        "modalidades_bajo_umbral": [m for m in modalidades if por_mod.get(m, 0) < min_casos],
        "modalidades_fuera_de_taxonomia": sorted(set(por_mod) - set(modalidades)),
    }


def render_table(cov: dict) -> str:
    head = f"{'modalidad':<32}" + "".join(f"{c:>10}" for c in CANALES) + f"{'total':>8}"
    lines = [head, "-" * len(head)]
    for m, total in cov["casos_por_modalidad"].items():
        row = cov["casos_por_modalidad_canal"][m]
        flag = "  ✗" if m in cov["modalidades_bajo_umbral"] else ""
        lines.append(f"{m:<32}" + "".join(f"{row[c]:>10}" for c in CANALES) + f"{total:>8}{flag}")
    bajo = cov["modalidades_bajo_umbral"]
    lines.append(
        f"\n✗ = menos de {cov['min_casos_por_modalidad']} casos: {len(bajo)} modalidad(es) bajo umbral"
        if bajo
        else "\nOK: todas las modalidades superan el umbral"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Auditoría de inyección indirecta: instrucciones dirigidas a un LLM dentro de la KB.
# ---------------------------------------------------------------------------

INJECTION_PATTERNS = [
    r"ignor[aá]\w*\s+(?:\w+\s+){0,3}instrucciones",
    r"ignore\s+(?:\w+\s+){0,3}instructions",
    r"\b(?:asistente|assistant|system|sistema)\s*:",
    r"(?:respond[eé]|contest[aá]|clasific[aá])\w*\s+(?:\w+\s+){0,3}(?:que\s+)?(?:es\s+)?leg[ií]timo",
    r"parece_legitimo",
    r"system\s+prompt|prompt\s+del\s+sistema",
    r"</?\s*(?:mensaje_no_confiable|evidencia|caso|guia)\b",
]
_INJ = re.compile("|".join(f"(?:{p})" for p in INJECTION_PATTERNS), re.I)


def audit_injection(chunks: Iterable[tuple[str, str]]) -> list[dict]:
    """`chunks`: pares (doc_id, texto). Devuelve los que matchean algún patrón.

    Los mensajes de la colección `mensajes` pueden contener inyecciones *a propósito* (son
    ejemplos de ataques): la auditoría los lista igual para que el curador los revise."""
    out = []
    for doc_id, texto in chunks:
        if m := _INJ.search(texto):
            out.append({"doc_id": doc_id, "match": m.group(0)[:80]})
    return out


def main(argv: list[str] | None = None) -> int:
    from safefish.config import load_config

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    path = cfg.snapshot_dir / "chunks.jsonl" if cfg.snapshot_dir else None
    if path is None or not path.exists():
        print(f"No existe {path}: corré `make kb CONFIG={args.config}` primero")
        return 1
    with open(path, encoding="utf-8") as f:
        mds = [json.loads(line)["metadata"] for line in f]
    cov = coverage(mds, cfg.ingest.min_casos_por_modalidad)
    print(f"Cobertura de {cfg.kb_snapshot} (casos por modalidad × canal)\n")
    print(render_table(cov))
    return 1 if cov["modalidades_bajo_umbral"] else 0


if __name__ == "__main__":
    sys.exit(main())
