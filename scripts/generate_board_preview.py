"""
Generate a tri-fold science fair board preview as a single PNG.

Images are embedded from graphs/ — no browser or relative-path issues.

Run:
    python scripts/generate_board_preview.py

Output:
    poster/board_preview.png
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
GRAPHS = ROOT / "graphs"
OUT = ROOT / "poster" / "board_preview.png"

# 48 × 36 in board at 80 dpi → good screen preview, ~4 MB PNG
DPI = 80
W_IN, H_IN = 48, 36


def _wrap(text: str, width: int = 42) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def _img(path: Path):
    if path.is_file():
        return mpimg.imread(str(path))
    return None


def _panel_bg(ax, x0, color="#fafafa"):
    ax.add_patch(
        Rectangle((x0, 0), 1 / 3, 1, transform=ax.transAxes, facecolor=color, edgecolor="none", zorder=0)
    )


def _section(ax, x0, y, title: str, body: str, *, title_size=11, body_size=8.5, width=38):
    ax.text(
        x0 + 0.02,
        y,
        title.upper(),
        transform=ax.transAxes,
        fontsize=title_size,
        fontweight="bold",
        va="top",
        ha="left",
        color="#111",
        zorder=5,
    )
    ax.plot([x0 + 0.02, x0 + 0.31], [y - 0.012, y - 0.012], transform=ax.transAxes, color="#222", lw=1.2, zorder=5)
    ax.text(
        x0 + 0.02,
        y - 0.028,
        _wrap(body, width=width),
        transform=ax.transAxes,
        fontsize=body_size,
        va="top",
        ha="left",
        color="#222",
        linespacing=1.35,
        zorder=5,
    )


def _paste_fig(ax, x0, y, w, h, img_path: Path, caption: str, fig_num: int | None = None):
    img = _img(img_path)
    if img is None:
        ax.text(x0 + w / 2, y - h / 2, f"Missing:\n{img_path.name}", ha="center", va="center", fontsize=7)
        return
    inset = ax.inset_axes([x0, y - h, w, h], transform=ax.transAxes)
    inset.imshow(img)
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_edgecolor("#888")
        spine.set_linewidth(0.8)
    cap = f"Figure {fig_num}: {caption}" if fig_num else caption
    ax.text(
        x0,
        y - h - 0.018,
        _wrap(cap, width=int(w * 120)),
        transform=ax.transAxes,
        fontsize=7,
        va="top",
        ha="left",
        color="#333",
        zorder=5,
    )


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(W_IN, H_IN), facecolor="#c5ccd6")
    ax = fig.add_axes([0.02, 0.06, 0.96, 0.9])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Board surface
    board = FancyBboxPatch(
        (0.01, 0.02),
        0.98,
        0.96,
        boxstyle="square,pad=0",
        facecolor="white",
        edgecolor="#999",
        linewidth=2,
        transform=ax.transAxes,
        zorder=1,
    )
    ax.add_patch(board)

    # Fold lines
    for xf in (1 / 3, 2 / 3):
        ax.plot([xf, xf], [0.03, 0.97], transform=ax.transAxes, color="#bbb", ls="--", lw=1.5, zorder=2)

    _panel_bg(ax, 0, "#f5f5f5")
    _panel_bg(ax, 1 / 3, "#ffffff")
    _panel_bg(ax, 2 / 3, "#f5f5f5")

    # ---- LEFT PANEL ----
    lx = 0.0
    _section(
        ax,
        lx,
        0.94,
        "Background",
        "Actinium-225 is a scarce alpha-emitting radiopharmaceutical for targeted alpha therapy (TAT). "
        "Planning irradiation (flux, energy, time) requires a stiff five-isotope chain solved many times. "
        "ODE integrators are accurate but too slow for large parameter sweeps.",
        width=36,
    )
    _section(
        ax,
        lx,
        0.72,
        "Research Question",
        "Can a physics-informed neural network accurately and rapidly predict Ac-225 inventory "
        "across diverse scenarios vs a Bateman ODE reference?",
        width=34,
    )
    rq = FancyBboxPatch(
        (lx + 0.025, 0.52),
        0.28,
        0.14,
        boxstyle="square,pad=0.008",
        facecolor="#eef2ff",
        edgecolor="#333",
        linewidth=1.5,
        transform=ax.transAxes,
        zorder=4,
    )
    ax.add_patch(rq)
    ax.text(
        lx + 0.035,
        0.645,
        _wrap(
            "Can a PINN match ODE reference on held-out Ac-225 while enabling fast screening?",
            width=32,
        ),
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        va="top",
        zorder=5,
    )

    _section(
        ax,
        lx,
        0.48,
        "Hypothesis",
        "Embedding Bateman physics in architecture + loss yields <10% held-out Ac-225 error vs ODE "
        "and orders-of-magnitude faster inference.",
        width=36,
    )
    _section(
        ax,
        lx,
        0.32,
        "Expected Outcomes",
        "Six gates PASS. Held-out median <10%. Strongest: thermal & 14 MeV. Weakest: epithermal & threshold.",
        width=36,
    )
    ax.text(
        lx + 0.03,
        0.12,
        "Ra-226 → Ra-225 → Ac-225  (product)\nRa-226 → Ra-227 → Ac-227  (impurity)",
        transform=ax.transAxes,
        fontsize=9,
        ha="left",
        va="center",
        bbox=dict(boxstyle="square,pad=0.4", facecolor="#f0fdf4", edgecolor="#059669", lw=1.2),
        zorder=5,
    )

    # ---- CENTER PANEL ----
    cx = 1 / 3
    ax.text(
        0.5,
        0.955,
        "Computational Surrogate for Ac-225 Production Planning\nin Targeted Alpha Therapy",
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        ha="center",
        va="top",
        color="#0f172a",
        zorder=5,
    )
    ax.text(
        0.5,
        0.885,
        "Samuel Ogunnubi  ·  Anne Arundel Community College  ·  May 2026",
        transform=ax.transAxes,
        fontsize=10,
        ha="center",
        va="top",
        color="#475569",
        zorder=5,
    )

    _section(
        ax,
        cx,
        0.84,
        "Methodology",
        "1. 0D Bateman ODE reference (NNDC/JENDL, Radau). "
        "2. Train PINN with Bateman backbone + physics loss (600 pretrain + 3400 joint). "
        "3. Six validation gates + 22 held-out scenarios. "
        "4. Streamlit demo for live PINN vs ODE.",
        width=38,
        body_size=8,
    )
    _paste_fig(
        ax,
        cx + 0.02,
        0.52,
        0.28,
        0.22,
        GRAPHS / "isef_mass_conservation.png",
        "Mass conservation — five-species budget drift (ppm); PINN within ±10 ppm band.",
        fig_num=1,
    )

    # ---- RIGHT PANEL ----
    rx = 2 / 3
    _section(
        ax,
        rx,
        0.94,
        "Results",
        "6/6 PASS  ·  4.51% held-out Ac-225 median  ·  weights v63 (sha256 7c21debe)",
        width=34,
        body_size=9,
    )
    table_y = 0.82
    rows = [
        ("Empty-target safety", "PASS"),
        ("Production 14 MeV", "PASS (9.9%)"),
        ("Decay-chain ingrowth", "PASS"),
        ("Quality gate", "PASS"),
        ("Correlation", "PASS"),
        ("Held-out Ac-225 (22)", "4.51% median"),
    ]
    for i, (label, val) in enumerate(rows):
        yy = table_y - i * 0.038
        ax.text(rx + 0.025, yy, label, transform=ax.transAxes, fontsize=7.5, va="top", zorder=5)
        ax.text(rx + 0.22, yy, val, transform=ax.transAxes, fontsize=7.5, fontweight="bold", color="#059669", va="top", zorder=5)

    _paste_fig(
        ax,
        rx + 0.02,
        0.58,
        0.28,
        0.24,
        GRAPHS / "isef_parity_restyled.png",
        "Ac-225 parity — held-out 22 scenarios; median 4.51% vs ODE.",
        fig_num=2,
    )
    _paste_fig(
        ax,
        rx + 0.02,
        0.28,
        0.28,
        0.16,
        GRAPHS / "isef_isotope_evolution.png",
        "Ac-225 evolution vs time; PINN tracks ODE.",
        fig_num=3,
    )

    _section(
        ax,
        rx,
        0.08,
        "Conclusions",
        "PINN enables rapid screening vs repeated ODE. Limitation: ODE-only validation, 0D model.",
        width=34,
        body_size=7.5,
    )

    # Footer
    ax.text(
        0.03,
        0.04,
        "References: Raissi et al. 2019 (PINN); NNDC/JENDL; DOE Isotope Program  ·  "
        "Demo: lhyjrhmwzxqfpuuwsux7zh.streamlit.app  ·  github.com/samogunnubi0-del/PINN",
        transform=ax.transAxes,
        fontsize=7,
        color="#555",
        zorder=5,
    )

    fig.savefig(OUT, dpi=DPI, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT.relative_to(ROOT)} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
