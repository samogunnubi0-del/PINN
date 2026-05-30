"""
Tri-fold science fair board preview (48" × 36") as one PNG.

Layout follows standard 3-panel board rules (Mit-style):
  Left   — Background, Research Question, Hypothesis, Expected Outcomes
  Center — Title, name, Methodology (numbered steps)
  Right  — Results table, small figures with captions, Conclusions
  Footer — Key References, Acknowledgements, demo link

Run:  python scripts/generate_board_preview.py
Out:   poster/board_preview.png
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
GRAPHS = ROOT / "graphs"
OUT = ROOT / "poster" / "board_preview.png"

# Standard tri-fold: 48" wide × 36" tall; 72 dpi → crisp preview
DPI = 72
W_IN, H_IN = 48, 36

# Panel inner margin (fraction of each panel axis)
PAD = 0.06

# Typography — readable from ~6 ft at full print size (pt on 48×36 canvas)
FS_TITLE = 40
FS_SUBTITLE = 22
FS_SECTION = 26
FS_BODY = 20
FS_SMALL = 17
FS_CAPTION = 16
FS_FOOTER = 14


def _wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def _load_img(path: Path):
    return mpimg.imread(str(path)) if path.is_file() else None


def _section_header(ax, y: float, title: str) -> float:
    """Draw ALL-CAPS section label + rule; return y below header."""
    ax.text(PAD, y, title.upper(), fontsize=FS_SECTION, fontweight="bold", va="top", ha="left", color="#111")
    ax.plot([PAD, 1 - PAD], [y - 0.018, y - 0.018], color="#222", lw=1.5, transform=ax.transAxes, clip_on=False)
    return y - 0.045


def _body(ax, y: float, text: str, *, width: int = 52, size: int = FS_BODY) -> float:
    wrapped = _wrap(text, width)
    lines = wrapped.count("\n") + 1
    ax.text(PAD, y, wrapped, fontsize=size, va="top", ha="left", color="#222", linespacing=1.25)
    return y - lines * 0.028 - 0.02


def _figure_block(ax, y: float, img_path: Path, fig_num: int, caption: str, *, w: float, h: float) -> float:
    """Place a small figure + Mit-style caption; return y below block."""
    x0 = PAD
    img = _load_img(img_path)
    if img is not None:
        inset = ax.inset_axes([x0, y - h, w, h])
        inset.imshow(img)
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_edgecolor("#666")
            spine.set_linewidth(0.6)
    else:
        ax.text(x0 + w / 2, y - h / 2, f"[missing {img_path.name}]", ha="center", va="center", fontsize=FS_SMALL)

    cap = f"Figure {fig_num}: {_wrap(caption, width=int(w * 95))}"
    cap_lines = cap.count("\n") + 1
    ax.text(x0, y - h - 0.012, cap, fontsize=FS_CAPTION, va="top", ha="left", color="#333", linespacing=1.2)
    return y - h - cap_lines * 0.022 - 0.025


def _setup_panel(ax, face: str = "#ffffff") -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor=face, edgecolor="none", zorder=0))


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(W_IN, H_IN), facecolor="#d1d5db")
    gs = GridSpec(
        20, 3,
        figure=fig,
        height_ratios=[1] * 18 + [0.55, 0.45],
        width_ratios=[1, 1, 1],
        hspace=0.04,
        wspace=0.06,
        left=0.015,
        right=0.985,
        top=0.98,
        bottom=0.02,
    )

    ax_l = fig.add_subplot(gs[0:18, 0])
    ax_c = fig.add_subplot(gs[0:18, 1])
    ax_r = fig.add_subplot(gs[0:18, 2])
    ax_f = fig.add_subplot(gs[18:20, :])

    _setup_panel(ax_l, "#f8fafc")
    _setup_panel(ax_c, "#ffffff")
    _setup_panel(ax_r, "#f8fafc")
    _setup_panel(ax_f, "#eef2f7")

    # Fold guides (on main figure)
    for fx in (1 / 3, 2 / 3):
        fig.add_artist(
            plt.Line2D([fx, fx], [0.04, 0.92], transform=fig.transFigure, color="#94a3b8", ls="--", lw=1.2, zorder=0)
        )

    # ===================== LEFT PANEL =====================
    y = 0.97
    y = _section_header(ax_l, y, "Background")
    y = _body(
        ax_l,
        y,
        "Actinium-225 is a scarce alpha-emitting radiopharmaceutical used in targeted alpha therapy "
        "(TAT) for cancer. Global supply limits clinical trials and patient access.",
        width=48,
    )
    y = _body(
        ax_l,
        y,
        "Planning irradiation — neutron flux, energy, and time — requires solving a stiff five-isotope "
        "transmutation chain many times. Classical ODE integrators (Radau) are accurate but too slow "
        "for large parameter sweeps.",
        width=48,
    )

    chain = FancyBboxPatch(
        (PAD, y - 0.11),
        1 - 2 * PAD,
        0.10,
        boxstyle="square,pad=0.008",
        facecolor="#f0fdf4",
        edgecolor="#059669",
        linewidth=1.2,
        transform=ax_l.transAxes,
    )
    ax_l.add_patch(chain)
    ax_l.text(
        PAD + 0.02,
        y - 0.02,
        "Five-species chain (0D)\nRa-226 → Ra-225 → Ac-225  (product)\nRa-226 → Ra-227 → Ac-227  (impurity)",
        transform=ax_l.transAxes,
        fontsize=FS_SMALL,
        va="top",
        ha="left",
    )
    y -= 0.14

    y = _section_header(ax_l, y, "Research Question")
    rq = FancyBboxPatch(
        (PAD, y - 0.13),
        1 - 2 * PAD,
        0.12,
        boxstyle="square,pad=0.008",
        facecolor="#eff6ff",
        edgecolor="#1e40af",
        linewidth=1.5,
        transform=ax_l.transAxes,
    )
    ax_l.add_patch(rq)
    ax_l.text(
        PAD + 0.02,
        y - 0.015,
        _wrap(
            "Can a physics-informed neural network accurately and rapidly predict Ac-225 inventory "
            "across diverse irradiation scenarios compared to a trusted Bateman ODE reference?",
            width=44,
        ),
        transform=ax_l.transAxes,
        fontsize=FS_BODY,
        fontweight="bold",
        va="top",
        ha="left",
        linespacing=1.2,
    )
    y -= 0.16

    y = _section_header(ax_l, y, "Hypothesis")
    y = _body(
        ax_l,
        y,
        "If Bateman physics is embedded in the network architecture and training loss, the PINN will "
        "match the ODE within 10% on held-out production scenarios while enabling orders-of-magnitude "
        "faster screening than sequential ODE integration.",
        width=48,
    )

    y = _section_header(ax_l, y, "Expected Outcomes")
    outcomes = (
        "• Six validation gates PASS\n"
        "• Held-out Ac-225 < 10% vs ODE\n"
        "• Best: thermal / 14 MeV (~4–5%)\n"
        "• Hardest: epithermal / threshold"
    )
    ax_l.text(PAD, y, outcomes, fontsize=FS_SMALL, va="top", ha="left", linespacing=1.35, color="#222")

    # ===================== CENTER PANEL =====================
    ax_c.text(
        0.5,
        0.98,
        _wrap("Computational Surrogate for Ac-225 Production Planning in Targeted Alpha Therapy", width=34),
        fontsize=FS_TITLE - 4,
        fontweight="bold",
        ha="center",
        va="top",
        color="#0f172a",
        linespacing=1.08,
    )
    ax_c.text(0.5, 0.84, "Samuel Ogunnubi", fontsize=FS_SUBTITLE, fontweight="bold", ha="center", va="top")
    ax_c.text(
        0.5,
        0.795,
        "Anne Arundel Community College · Dual Enrollment · May 2026",
        fontsize=FS_SUBTITLE - 2,
        ha="center",
        va="top",
        color="#475569",
    )

    y = 0.72
    y = _section_header(ax_c, y, "Methodology")
    steps = (
        "1. Reference model: 0D five-species Bateman ODE with NNDC/ENSDF half-lives and JENDL cross "
        "sections; stiff Radau integrator generates training targets.\n\n"
        "2. Training coverage: Scenarios across thermal, epithermal, threshold (~6.4 MeV), and 14 MeV "
        "energies; virgin and recycled inventories.\n\n"
        "3. Surrogate: Physics-informed NN with semi-analytic Bateman backbone and bounded corrections.\n\n"
        "4. Training: 600-epoch physics pretrain + 3,400-epoch joint training (v63 weights); mass "
        "conservation in loss.\n\n"
        "5. Validation: Six independent gates + 22 held-out scenarios (seed 42).\n\n"
        "6. Demo: Streamlit app for live PINN vs ODE comparison and parameter screening."
    )
    ax_c.text(PAD, y, steps, fontsize=FS_SMALL - 1, va="top", ha="left", linespacing=1.25, color="#222")

    # Small supporting figure — ~10% panel height (board rule: figures support text, not dominate)
    _figure_block(
        ax_c,
        0.19,
        GRAPHS / "isef_mass_conservation.png",
        1,
        "Atom budget drift (ppm) vs time; PINN within ±10 ppm band.",
        w=0.78,
        h=0.10,
    )

    # ===================== RIGHT PANEL =====================
    y = 0.97
    y = _section_header(ax_r, y, "Results")
    ax_r.text(
        PAD,
        y,
        "Overall: 6/6 PASS  ·  Held-out Ac-225: 4.51% median  ·  Weights v63",
        fontsize=FS_SMALL,
        fontweight="bold",
        va="top",
        color="#059669",
    )
    y -= 0.05

    rows = [
        ("Empty-target safety", "PASS"),
        ("Production (14 MeV)", "PASS (9.9%)"),
        ("Decay-chain ingrowth", "PASS"),
        ("Species quality gate", "PASS"),
        ("PINN vs ODE correlation", "PASS"),
        ("Held-out Ac-225 (22)", "4.51% median"),
    ]
    row_h = 0.032
    for i, (label, val) in enumerate(rows):
        yy = y - i * row_h
        ax_r.text(PAD, yy, label, fontsize=FS_SMALL, va="top", ha="left")
        ax_r.text(0.72, yy, val, fontsize=FS_SMALL, fontweight="bold", va="top", ha="right", color="#059669")
    y -= len(rows) * row_h + 0.03

    # Compact figures side-by-side (~8% height each — poster supports text first)
    fig_h = 0.085
    fig_w = 0.40
    gap = 0.05
    y_top = y - 0.02

    fig_items = [
        (GRAPHS / "isef_parity_restyled.png", 2, "Ac-225 parity; 4.51% median vs ODE (22 held-out)."),
        (GRAPHS / "isef_isotope_evolution.png", 3, "Ac-225 vs time; PINN tracks ODE."),
    ]
    for idx, (path, num, cap) in enumerate(fig_items):
        x0 = PAD + idx * (fig_w + gap)
        yt = y_top
        img = _load_img(path)
        if img is not None:
            inset = ax_r.inset_axes([x0, yt - fig_h, fig_w, fig_h])
            inset.imshow(img)
            inset.set_xticks([])
            inset.set_yticks([])
            for spine in inset.spines.values():
                spine.set_edgecolor("#666")
        ax_r.text(
            x0,
            yt - fig_h - 0.006,
            f"Fig {num}: {_wrap(cap, width=38)}",
            fontsize=FS_CAPTION - 1,
            va="top",
            ha="left",
            linespacing=1.1,
        )

    y = y_top - fig_h - 0.10

    y = _section_header(ax_r, y, "Results & Conclusions")
    y = _body(
        ax_r,
        y,
        "The PINN passed 6/6 validation gates with 4.51% held-out Ac-225 median error vs ODE, enabling "
        "rapid scenario screening impractical with repeated stiff solves.",
        width=50,
        size=FS_SMALL,
    )
    _body(
        ax_r,
        y,
        "Limitations: Validated vs ODE only — not reactor or clinical data. 0D lumped model; not patient "
        "dosing or 3D transport (MCNP/OpenMC).",
        width=50,
        size=FS_SMALL,
    )

    # ===================== FOOTER =====================
    ax_f.text(
        0.02,
        0.92,
        "KEY REFERENCES",
        fontsize=FS_SECTION - 4,
        fontweight="bold",
        va="top",
    )
    refs = (
        "Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks. "
        "Journal of Computational Physics, 378, 686–707.\n"
        "NNDC/NuDat decay data; JENDL-4.0 Ra-226 cross sections; DOE Isotope Program (Ac-225 supply)."
    )
    ax_f.text(0.02, 0.72, refs, fontsize=FS_FOOTER, va="top", linespacing=1.25, color="#333")

    ax_f.text(
        0.02,
        0.28,
        "ACKNOWLEDGEMENTS: Adult sponsor and science fair mentor. Faculty reviewers (pending). "
        "Computational training via Colab/Kaggle.",
        fontsize=FS_FOOTER,
        va="top",
        color="#333",
    )
    ax_f.text(
        0.72,
        0.85,
        "LIVE DEMO",
        fontsize=FS_SECTION - 6,
        fontweight="bold",
        ha="left",
        va="top",
    )
    ax_f.text(
        0.72,
        0.62,
        "lhyjrhmwzxqfpuuwsux7zh.streamlit.app\n\ngithub.com/samogunnubi0-del/PINN",
        fontsize=FS_FOOTER,
        ha="left",
        va="top",
        color="#1d4ed8",
    )

    fig.savefig(OUT, dpi=DPI, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    kb = OUT.stat().st_size // 1024
    print(f"Saved {OUT.relative_to(ROOT)} ({kb} KB) — 48×36 in tri-fold @ {DPI} dpi")


if __name__ == "__main__":
    main()
