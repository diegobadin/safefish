"""Proyección UMAP de los embeddings de la base de conocimiento (clase 8: visualizar embeddings).

    uv run --with umap-learn --with matplotlib python scripts/umap_kb.py [--snapshot kb-v2]

UMAP (Uniform Manifold Approximation and Projection) lleva los vectores de 1.536 dimensiones a 2
preservando vecindades: dos puntos cerca en el gráfico tienen embeddings parecidos. Los ejes no
tienen unidad. No llama a la API: lee los vectores guardados en Chroma.

Genera dos figuras en docs/img/:
- umap_kb.png: toda la KB, por colección (mensajes estafa / legítimos, casos, señales y guías).
- umap_casos.png: solo los casos, con el nombre de cada modalidad sobre su grupo.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import chromadb
import numpy as np

# Paleta categórica validada (primeras 3 posiciones, aptas para scatter) y superficies.
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
AZUL, NARANJA, AQUA, GRIS = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29b"
SEMILLA = 7


def leer(snapshot: Path) -> dict[str, tuple[np.ndarray, list[dict]]]:
    client = chromadb.PersistentClient(path=str(snapshot / "chroma"))
    out = {}
    for col in client.list_collections():
        r = col.get(include=["embeddings", "metadatas"])
        tipo = col.name.rsplit("_", 1)[-1]  # kb_v2_casos -> casos
        out[tipo] = (np.asarray(r["embeddings"], dtype=np.float32), r["metadatas"])
    return out


def proyectar(x: np.ndarray, vecinos: int) -> np.ndarray:
    import umap

    return umap.UMAP(
        n_neighbors=vecinos, min_dist=0.1, metric="cosine", random_state=SEMILLA
    ).fit_transform(x)


def ejes(ax, titulo: str, subtitulo: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(titulo, loc="left", fontsize=13, color=TEXT, pad=22, fontweight="semibold")
    ax.text(0, 1.015, subtitulo, transform=ax.transAxes, fontsize=9.5, color=TEXT_2, va="bottom")


def figura_kb(datos, out: Path) -> None:
    import matplotlib.pyplot as plt

    partes, grupos = [], []
    for tipo in ("mensajes", "casos", "senales", "guias"):
        x, mds = datos[tipo]
        partes.append(x)
        for md in mds:
            if tipo == "mensajes":
                grupos.append("estafa" if md.get("label") == "estafa" else "legitimo")
            else:
                grupos.append("casos" if tipo == "casos" else "otros")
    xy = proyectar(np.vstack(partes), vecinos=15)
    g = np.array(grupos)

    estilos = [  # orden de dibujo: lo numeroso abajo
        ("estafa", "Mensajes de estafa", AZUL, 5, 0.45),
        ("legitimo", "Mensajes legítimos", NARANJA, 5, 0.55),
        ("otros", "Señales y guías", GRIS, 22, 0.9),
        ("casos", "Casos documentados", AQUA, 40, 1.0),
    ]
    fig, ax = plt.subplots(figsize=(9, 6.6), dpi=200, facecolor=SURFACE)
    for clave, nombre, color, size, alpha in estilos:
        m = g == clave
        borde = {"edgecolors": SURFACE, "linewidths": 0.8} if size > 10 else {"linewidths": 0}
        ax.scatter(*xy[m].T, s=size, c=color, alpha=alpha, label=f"{nombre} ({m.sum():,})".replace(",", "."), **borde)  # fmt: skip
    ejes(
        ax,
        "Base de conocimiento kb-v2 proyectada con UMAP",
        "Cada punto es un chunk; cerca = embeddings parecidos (similitud coseno). Ejes sin unidad.",
    )
    leg = ax.legend(loc="lower left", frameon=False, fontsize=9, markerscale=1.6, labelcolor=TEXT)
    for h in leg.legend_handles:
        h.set_alpha(1)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def figura_casos(datos, out: Path) -> None:
    import matplotlib.pyplot as plt

    x, mds = datos["casos"]
    xy = proyectar(x, vecinos=8)
    por_mod: dict[str, list[int]] = defaultdict(list)
    for i, md in enumerate(mds):
        por_mod[str(md.get("modalidad"))].append(i)

    fig, ax = plt.subplots(figsize=(9, 6.6), dpi=200, facecolor=SURFACE)
    etiquetas = []
    for mod, idx in sorted(por_mod.items()):
        cx, cy = xy[idx].mean(axis=0)
        for i in idx:  # cada caso unido al centro de su modalidad
            ax.plot([xy[i, 0], cx], [xy[i, 1], cy], color=GRID, lw=1.0, zorder=1)
        etiquetas.append(
            ax.annotate(
                f"{mod.replace('_', ' ')} ({len(idx)})",
                (cx, cy),
                xytext=(0, 0),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=8.5,
                color=TEXT,
                zorder=3,
                bbox={"boxstyle": "round,pad=0.3", "fc": SURFACE, "ec": "#c9c8c2", "lw": 0.6},
            )  # fmt: skip
        )
    ax.scatter(*xy.T, s=46, c=AZUL, edgecolors=SURFACE, linewidths=0.9, zorder=2)
    ax.margins(0.08)
    ejes(
        ax,
        "Casos documentados por modalidad (UMAP)",
        "59 casos; cada punto se une al centro de su modalidad. Entre paréntesis, cuántos casos tiene.",
    )
    fig.tight_layout()
    _separar_etiquetas(fig, etiquetas)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def _separar_etiquetas(fig, etiquetas, margen_pt: float = 3.0, rondas: int = 50) -> None:
    """Corre verticalmente las etiquetas que se superponen (en puntos de pantalla)."""
    escala = 72.0 / fig.dpi
    for _ in range(rondas):
        fig.canvas.draw()
        cajas = [e.get_window_extent() for e in etiquetas]
        movida = False
        for a in range(len(etiquetas)):
            for b in range(a + 1, len(etiquetas)):
                ca, cb = cajas[a], cajas[b]
                if ca.overlaps(cb):
                    dy = (min(ca.y1, cb.y1) - max(ca.y0, cb.y0)) * escala / 2 + margen_pt
                    arriba, abajo = (a, b) if ca.y0 + ca.y1 >= cb.y0 + cb.y1 else (b, a)
                    for k, s in ((arriba, 1), (abajo, -1)):
                        x, y = etiquetas[k].xyann
                        etiquetas[k].xyann = (x, y + s * dy)
                    movida = True
        if not movida:
            return


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--snapshot", default="kb-v2")
    ap.add_argument("--out", default="docs/img")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    datos = leer(Path("data/kb_snapshots") / args.snapshot)
    figura_kb(datos, out / "umap_kb.png")
    figura_casos(datos, out / "umap_casos.png")
    print(f"Figuras en {out}/umap_kb.png y {out}/umap_casos.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
