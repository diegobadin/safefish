"""Proyección UMAP de los embeddings de la base de conocimiento (clase 8: visualizar embeddings).

    uv run --with umap-learn --with matplotlib python scripts/umap_kb.py [--snapshot kb-v2]

UMAP (Uniform Manifold Approximation and Projection) lleva los vectores de 1.536 dimensiones a 2
preservando vecindades: dos puntos cerca en el gráfico tienen embeddings parecidos. Los ejes no
tienen unidad. No llama a la API: lee los vectores guardados en Chroma.

Genera dos figuras en docs/img/:
- umap_kb.png: toda la KB, por colección (mensajes estafa / legítimos, casos, señales y guías).
- umap_casos.png: solo los casos, con el nombre de cada modalidad sobre su grupo.
- umap_colecciones.png: la misma proyección en cuatro paneles, uno por colección.
- umap_descriptivos.png: zoom sobre casos, guías y señales (sin los mensajes).
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


def figura_kb_slide(datos, out: Path) -> None:
    """Versión para proyectar: misma proyección que `figura_kb`, letra grande y sin título."""
    import matplotlib.pyplot as plt

    partes, grupos = [], []
    for tipo in ("mensajes", "casos", "senales", "guias"):
        x, mds = datos[tipo]
        partes.append(x)
        for md in mds:
            if tipo == "mensajes":
                grupos.append("estafa" if md.get("label") == "estafa" else "legitimo")
            else:
                grupos.append({"casos": "casos", "senales": "senales", "guias": "guias"}[tipo])
    xy = proyectar(np.vstack(partes), vecinos=15)
    g = np.array(grupos)
    estilos = [
        ("estafa", "Mensajes de estafa", AZUL, 10, 0.45),
        ("legitimo", "Mensajes legítimos", NARANJA, 10, 0.55),
        ("guias", "Guías", "#6b6a64", 70, 1.0),
        ("senales", "Señales", "#6b6a64", 70, 1.0),
        ("casos", "Casos documentados", AQUA, 90, 1.0),
    ]
    fig, ax = plt.subplots(figsize=(12, 8), dpi=150, facecolor=SURFACE)
    for clave, nombre, color, size, alpha in estilos:
        m = g == clave
        marker = "s" if clave == "senales" else "o"
        borde = {"edgecolors": SURFACE, "linewidths": 1.2} if size > 20 else {"linewidths": 0}
        ax.scatter(*xy[m].T, s=size, c=color, alpha=alpha, marker=marker, label=f"{nombre} ({m.sum():,})".replace(",", "."), **borde)  # fmt: skip
    ax.set_facecolor(SURFACE)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    leg = ax.legend(loc="upper right", frameon=False, fontsize=20, markerscale=1.6, labelcolor=TEXT)
    for h in leg.legend_handles:
        h.set_alpha(1)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


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


def figura_por_coleccion(datos, out: Path) -> None:
    """Una sola proyección de toda la KB, dibujada en cuatro paneles: cada panel resalta una
    colección y deja el resto en gris como referencia. Las posiciones son comparables entre
    paneles porque la proyección es la misma."""
    import matplotlib.pyplot as plt

    tipos = ("mensajes", "casos", "senales", "guias")
    partes, origen, mds = [], [], []
    for tipo in tipos:
        x, md = datos[tipo]
        partes.append(x)
        origen += [tipo] * len(md)
        mds += md
    xy = proyectar(np.vstack(partes), vecinos=15)
    origen = np.array(origen)

    titulos = {
        "mensajes": "Mensajes",
        "casos": "Casos documentados",
        "senales": "Señales",
        "guias": "Guías",
    }
    fig, axs = plt.subplots(2, 2, figsize=(16, 9), dpi=150, facecolor=SURFACE)
    for ax, tipo in zip(axs.flat, tipos, strict=True):
        m = origen == tipo
        ax.scatter(*xy[~m].T, s=4, c=GRID, linewidths=0, zorder=1)
        if tipo == "mensajes":
            lab = np.array([md.get("label") == "estafa" for md in mds])
            for sel, nombre, color in ((m & lab, "estafa", AZUL), (m & ~lab, "legítimos", NARANJA)):  # fmt: skip
                ax.scatter(*xy[sel].T, s=5, c=color, alpha=0.6, linewidths=0, zorder=2, label=f"{nombre} ({sel.sum():,})".replace(",", "."))  # fmt: skip
            leg = ax.legend(loc="lower left", frameon=False, fontsize=10, markerscale=3, labelcolor=TEXT)  # fmt: skip
            for h in leg.legend_handles:
                h.set_alpha(1)
        else:
            ax.scatter(*xy[m].T, s=42, c=AQUA, edgecolors=SURFACE, linewidths=0.8, zorder=3)
        if tipo == "senales":
            idx = np.flatnonzero(m)
            etiquetas = [
                ax.annotate(str(mds[i].get("signal_id", "")).replace("_", " "), xy[i], xytext=(6, 0), textcoords="offset points", fontsize=7.5, color=TEXT_2, va="center", zorder=4)  # fmt: skip
                for i in idx
            ]
            fig.canvas.draw()
            _separar_etiquetas(fig, etiquetas, rondas=30)
        ax.set_facecolor(SURFACE)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{titulos[tipo]} ({m.sum():,})".replace(",", "."), loc="left", fontsize=14, color=TEXT, fontweight="semibold")  # fmt: skip
    fig.suptitle(
        "La misma proyección UMAP, una colección por panel (en gris, el resto de la base)",
        x=0.01, ha="left", fontsize=12, color=TEXT_2,
    )  # fmt: skip
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def figura_descriptivos(datos, out: Path) -> None:
    """Zoom: casos, señales y guías sin los mensajes, para ver su estructura interna."""
    import matplotlib.pyplot as plt

    tipos = (("casos", "Casos", AQUA, 46), ("guias", "Guías", NARANJA, 46), ("senales", "Señales", AZUL, 70))  # fmt: skip
    partes, origen, mds = [], [], []
    for tipo, *_ in tipos:
        x, md = datos[tipo]
        partes.append(x)
        origen += [tipo] * len(md)
        mds += md
    xy = proyectar(np.vstack(partes), vecinos=8)
    origen = np.array(origen)

    fig, ax = plt.subplots(figsize=(12, 7.5), dpi=170, facecolor=SURFACE)
    for tipo, nombre, color, size in tipos:
        m = origen == tipo
        ax.scatter(*xy[m].T, s=size, c=color, edgecolors=SURFACE, linewidths=0.9, zorder=2, label=f"{nombre} ({m.sum()})")  # fmt: skip
    etiquetas = [
        ax.annotate(str(mds[i].get("signal_id", "")).replace("_", " "), xy[i], xytext=(7, 0), textcoords="offset points", fontsize=8, color=TEXT_2, va="center", zorder=3)  # fmt: skip
        for i in np.flatnonzero(origen == "senales")
    ]
    ax.margins(0.1)
    ejes(
        ax,
        "Casos, guías y señales sin los mensajes (UMAP)",
        "Zoom sobre las colecciones descriptivas. Cerca = embeddings parecidos; ejes sin unidad.",
    )
    ax.legend(loc="upper left", frameon=False, fontsize=10, markerscale=1.2, labelcolor=TEXT)
    fig.tight_layout()
    _separar_etiquetas(fig, etiquetas, rondas=40)
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
    figura_por_coleccion(datos, out / "umap_colecciones.png")
    figura_descriptivos(datos, out / "umap_descriptivos.png")
    figura_kb_slide(datos, out / "umap_kb_slide.png")
    print(f"Figuras en {out}/: umap_kb, umap_casos, umap_colecciones y umap_descriptivos (.png)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
