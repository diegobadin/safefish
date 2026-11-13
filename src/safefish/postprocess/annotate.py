"""Captura anotada (M-POST.2): dibuja la bbox de cada señal verificada sobre la imagen.

python -m safefish.postprocess.annotate --imagen captura.png --salida out.png --pred pred.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from safefish.schemas import SafeFishOutput, Severidad
from safefish.taxonomia import severidad

COLORES = {
    Severidad.CRITICA: (220, 38, 38),
    Severidad.ALTA: (234, 88, 12),
    Severidad.MEDIA: (202, 138, 4),
    Severidad.BAJA: (100, 116, 139),
}


def annotate(image_path: str | Path, out: SafeFishOutput, salida: str | Path) -> int:
    """Devuelve la cantidad de recuadros dibujados."""
    from PIL import Image, ImageDraw

    with Image.open(image_path) as im:
        img = im.convert("RGB")
    draw = ImageDraw.Draw(img)
    width = max(2, img.width // 300)
    n = 0
    for i, s in enumerate(out.senales, 1):
        if s.bbox is None or not s.verificada:
            continue
        color = COLORES[severidad(s.signal_id)]
        x0, y0, x1, y1 = s.bbox
        pad = width * 2
        draw.rectangle((x0 - pad, y0 - pad, x1 + pad, y1 + pad), outline=color, width=width)
        draw.text((x0 - pad, max(0, y0 - pad - 14)), f"{i}. {s.signal_id.value}", fill=color)
        n += 1
    img.save(salida)
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    ap.add_argument("--imagen", required=True)
    ap.add_argument("--pred", required=True, help="JSON de SafeFishOutput")
    ap.add_argument("--salida", required=True)
    args = ap.parse_args(argv)
    out = SafeFishOutput.model_validate_json(Path(args.pred).read_text(encoding="utf-8"))
    print(f"{annotate(args.imagen, out, args.salida)} recuadro(s) → {args.salida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
