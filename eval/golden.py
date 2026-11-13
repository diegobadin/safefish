"""Carga y validación del golden (§4.5).

    python -m eval.golden --golden-dir golden --kb-config configs/v1.yaml

Chequea esquema, ids únicos, que cada archivo tenga su split, `campaign_id` disjuntos entre
dev y test, y leakage contra la KB (campaign_id presentes en el manifest del snapshot).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from safefish.schemas import GoldenCase


def golden_path(golden_dir: str | Path, split: str) -> Path:
    return Path(golden_dir) / f"{split}.jsonl"


def load_golden(path: str | Path) -> list[GoldenCase]:
    cases, errors = [], []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                cases.append(GoldenCase.model_validate_json(line))
            except ValidationError as e:
                errors.append(f"{path}:{n}: {e.errors()[0]['msg']} ({e.errors()[0]['loc']})")
    if errors:
        raise ValueError("Golden inválido:\n" + "\n".join(errors))
    return cases


def golden_version(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def check_golden(golden_dir: str | Path, kb_manifest: dict | None = None) -> list[str]:
    """Devuelve la lista de problemas encontrados (vacía = OK)."""
    problems: list[str] = []
    by_split: dict[str, list[GoldenCase]] = {}
    for split in ("dev", "test"):
        p = golden_path(golden_dir, split)
        if not p.exists():
            problems.append(f"falta {p}")
            continue
        try:
            by_split[split] = load_golden(p)
        except ValueError as e:
            problems.append(str(e))
            continue
        wrong = [c.id for c in by_split[split] if c.split != split]
        if wrong:
            problems.append(f"{p.name}: casos con split distinto a '{split}': {wrong[:5]}")

    all_cases = [c for cs in by_split.values() for c in cs]
    dup_ids = [i for i, n in Counter(c.id for c in all_cases).items() if n > 1]
    if dup_ids:
        problems.append(f"ids repetidos: {dup_ids[:5]}")
    dup_txt = [t[:60] for t, n in Counter(c.texto.strip() for c in all_cases).items() if n > 1]
    if dup_txt:
        problems.append(f"textos repetidos: {dup_txt[:3]}")

    camp = {s: {c.campaign_id for c in cs if c.campaign_id} for s, cs in by_split.items()}
    if "dev" in camp and "test" in camp and (inter := camp["dev"] & camp["test"]):
        problems.append(
            f"campaign_id en dev y test a la vez (split por grupo): {sorted(inter)[:5]}"
        )
    if kb_manifest:
        kb_camps = set(kb_manifest.get("campaign_ids", []))
        for s, cs in camp.items():
            if leak := cs & kb_camps:
                problems.append(
                    f"LEAKAGE: campaign_id de {s} presentes en la KB: {sorted(leak)[:5]}"
                )
    return problems


def summarize(cases: list[GoldenCase]) -> dict:
    return {
        "n": len(cases),
        "label": dict(Counter(c.label for c in cases)),
        "slices": dict(Counter(s for c in cases for s in c.slices)),
        "canal": dict(Counter(str(c.canal) for c in cases)),
        "modalidad": dict(Counter(str(c.modalidad) for c in cases if c.label == "estafa")),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden-dir", default="golden")
    ap.add_argument("--kb-config", default=None, help="config cuyo snapshot se audita por leakage")
    args = ap.parse_args(argv)

    manifest = None
    if args.kb_config:
        from safefish.config import load_config

        cfg = load_config(args.kb_config)
        if cfg.snapshot_dir and (cfg.snapshot_dir / "manifest.json").exists():
            manifest = json.loads((cfg.snapshot_dir / "manifest.json").read_text())
        else:
            print(f"(sin snapshot {cfg.kb_snapshot}: se omite el chequeo de leakage contra la KB)")

    for split in ("dev", "test"):
        p = golden_path(args.golden_dir, split)
        if p.exists():
            with contextlib.suppress(ValueError):  # los errores se reportan abajo
                print(split, json.dumps(summarize(load_golden(p)), ensure_ascii=False))
    problems = check_golden(args.golden_dir, manifest)
    for pr in problems:
        print("✗", pr)
    print("OK" if not problems else f"{len(problems)} problema(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
