"""
Tri-fold board PREVIEW — optimized for on-screen reading (not tiny 48×36 print scale).

Uses Pillow for crisp text and properly resized figures.

Run:  python scripts/generate_board_preview.py
Out:   poster/board_preview.png  (3600 × 2700 px, 4:3)
"""
from __future__ import annotations

import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
GRAPHS = ROOT / "graphs"
OUT = ROOT / "poster" / "board_preview.png"

# Canvas: 4:3 like a 48×36 board; big enough to read when opened full-screen
W, H = 3600, 2700
FOOTER_H = 220
COL_W = W // 3
MARGIN = 48

# Colors
BG = (209, 213, 219)
WHITE = (255, 255, 255)
PANEL_L = (248, 250, 252)
PANEL_R = (248, 250, 252)
INK = (15, 23, 42)
MUTED = (71, 85, 105)
ACCENT = (29, 78, 216)
PASS = (5, 150, 105)
RULE = (30, 41, 59)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = ("arialbd.ttf", "Arial Bold.ttf") if bold else ("arial.ttf", "Arial.ttf")
    for name in names:
        path = Path("C:/Windows/Fonts") / name
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _wrap(text: str, width: int) -> list[str]:
    return textwrap.wrap(text, width=width) or [""]


def _line_h(draw: ImageDraw.ImageDraw, font, lines: list[str], spacing: int = 6) -> int:
    if not lines:
        return 0
    bbox = draw.textbbox((0, 0), "Ay", font=font)
    lh = bbox[3] - bbox[1]
    return len(lines) * (lh + spacing) - spacing


def _draw_paragraph(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    font,
    fill=INK,
    width: int = 42,
    spacing: int = 6,
) -> int:
    lines = _wrap(text, width)
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, y), line, font=font)
        y = bbox[3] + spacing
    return y


def _section(draw: ImageDraw.ImageDraw, x: int, y: int, title: str, col_w: int) -> int:
    f = _font(28, bold=True)
    draw.text((x, y), title.upper(), font=f, fill=INK)
    y += 38
    draw.line([(x, y), (x + col_w - 2 * MARGIN, y)], fill=RULE, width=2)
    return y + 20


def _paste_figure(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    path: Path,
    fig_num: int,
    caption: str,
    max_w: int,
    max_h: int,
) -> int:
    if not path.is_file():
        draw.text((x, y), f"[Missing {path.name}]", font=_font(18), fill=(180, 0, 0))
        return y + 40

    img = Image.open(path).convert("RGBA")
    img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
    # Center in box
    ox = x + (max_w - img.width) // 2
    canvas.paste(img, (ox, y), img if img.mode == "RGBA" else None)
    draw.rectangle([x, y, x + max_w, y + img.height], outline=(120, 120, 120), width=1)

    cy = y + img.height + 10
    cap_font = _font(17)
    cap = f"Figure {fig_num}: {caption}"
    for line in _wrap(cap, width=52):
        draw.text((x, cy), line, font=cap_font, fill=MUTED)
        cy += 22
    return cy + 12


class Column:
    def __init__(self, x0: int, width: int, bg: tuple):
        self.x0 = x0
        self.width = width
        self.bg = bg
        self.y = MARGIN + 8

    def inner_x(self) -> int:
        return self.x0 + MARGIN


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)

    body_h = H - FOOTER_H
    for i, bg in enumerate((PANEL_L, WHITE, PANEL_R)):
        draw.rectangle([i * COL_W, 0, (i + 1) * COL_W - 1, body_h], fill=bg)

    # Fold lines
    for fx in (COL_W, 2 * COL_W):
        draw.line([(fx, 20), (fx, body_h - 20)], fill=(148, 163, 184), width=2)

    f_title = _font(44, bold=True)
    f_sub = _font(24)
    f_body = _font(22)
    f_small = _font(19)
    f_bold = _font(22, bold=True)

    inner_w = COL_W - 2 * MARGIN
    fig_w = inner_w
    fig_h_small = 260
    fig_h_med = 300

    # ==================== LEFT ====================
    L = Column(0, COL_W, PANEL_L)
    x = L.inner_x()

    L.y = _section(draw, x, L.y, "Background", COL_W)
    L.y = _draw_paragraph(
        draw, x, L.y,
        "Actinium-225 is a scarce alpha-emitting radiopharmaceutical used in targeted alpha therapy (TAT) for cancer. Supply limits trials and treatment access.",
        f_body, width=46,
    ) + 8
    L.y = _draw_paragraph(
        draw, x, L.y,
        "Planning irradiation (flux, energy, time) requires a stiff five-isotope chain solved many times. ODE integrators are accurate but too slow for large sweeps.",
        f_body, width=46,
    ) + 16

    # Chain box
    box_h = 100
    draw.rounded_rectangle([x, L.y, x + inner_w, L.y + box_h], radius=8, outline=PASS, width=2, fill=(240, 253, 244))
    draw.text((x + 14, L.y + 10), "Five-species chain (0D)", font=f_bold, fill=INK)
    draw.text((x + 14, L.y + 38), "Ra-226 → Ra-225 → Ac-225  (product)", font=f_small, fill=INK)
    draw.text((x + 14, L.y + 64), "Ra-226 → Ra-227 → Ac-227  (impurity)", font=f_small, fill=INK)
    L.y += box_h + 24

    L.y = _section(draw, x, L.y, "Research Question", COL_W)
    rq_h = 130
    draw.rounded_rectangle([x, L.y, x + inner_w, L.y + rq_h], radius=8, outline=ACCENT, width=2, fill=(239, 246, 255))
    rq_lines = _wrap(
        "Can a physics-informed neural network accurately and rapidly predict Ac-225 inventory across diverse irradiation scenarios compared to a trusted Bateman ODE reference?",
        44,
    )
    ty = L.y + 14
    for line in rq_lines:
        draw.text((x + 14, ty), line, font=f_bold, fill=INK)
        ty += 28
    L.y += rq_h + 24

    L.y = _section(draw, x, L.y, "Hypothesis", COL_W)
    L.y = _draw_paragraph(
        draw, x, L.y,
        "If Bateman physics is embedded in the network and loss, the PINN will match the ODE within 10% on held-out scenarios and run much faster than repeated ODE solves.",
        f_body, width=46,
    ) + 12

    L.y = _section(draw, x, L.y, "Expected Outcomes", COL_W)
    for bullet in (
        "Six validation gates PASS",
        "Held-out Ac-225 median < 10%",
        "Best: thermal & 14 MeV (~4–5%)",
        "Hardest: epithermal & threshold",
    ):
        draw.text((x + 8, L.y), f"•  {bullet}", font=f_small, fill=INK)
        L.y += 30

    # ==================== CENTER ====================
    C = Column(COL_W, COL_W, WHITE)
    cx = C.x0 + COL_W // 2
    title_lines = _wrap(
        "Computational Surrogate for Ac-225 Production Planning in Targeted Alpha Therapy",
        32,
    )
    ty = MARGIN + 10
    for line in title_lines:
        bbox = draw.textbbox((0, 0), line, font=f_title)
        tw = bbox[2] - bbox[0]
        draw.text((cx - tw // 2, ty), line, font=f_title, fill=INK)
        ty += 52
    ty += 8
    for line, font in (("Samuel Ogunnubi", _font(28, bold=True)), ("Anne Arundel Community College · May 2026", f_sub)):
        bbox = draw.textbbox((0, 0), line, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((cx - tw // 2, ty), line, font=font, fill=MUTED if font == f_sub else INK)
        ty += 36
    C.y = ty + 20

    x = C.inner_x()
    C.y = _section(draw, x, C.y, "Methodology", COL_W)
    steps = [
        "Build 0D Bateman ODE reference (NNDC/JENDL data, Radau solver).",
        "Generate training scenarios: thermal, epithermal, threshold, 14 MeV.",
        "Train PINN: Bateman backbone + physics loss (600 pretrain + 3400 joint).",
        "Enforce mass budget — no production from empty target.",
        "Validate: six independent gates + 22 held-out scenarios.",
        "Streamlit demo: live PINN vs ODE + parameter screening.",
    ]
    for i, step in enumerate(steps, 1):
        draw.text((x, C.y), f"{i}.", font=f_bold, fill=ACCENT)
        C.y = _draw_paragraph(draw, x + 28, C.y, step, f_small, width=42) + 6

    C.y = _paste_figure(
        canvas, draw, x, C.y + 8,
        GRAPHS / "isef_mass_conservation.png",
        1, "Atom budget drift (ppm); PINN within ±10 ppm band.",
        fig_w, fig_h_small,
    )

    # ==================== RIGHT ====================
    R = Column(2 * COL_W, COL_W, PANEL_R)
    x = R.inner_x()
    R.y = _section(draw, x, R.y, "Results", COL_W)
    draw.text((x, R.y), "6/6 PASS  ·  4.51% held-out Ac-225  ·  v63 weights", font=f_bold, fill=PASS)
    R.y += 36

    rows = [
        ("Empty-target safety", "PASS"),
        ("Production (14 MeV)", "PASS (9.9%)"),
        ("Decay-chain ingrowth", "PASS"),
        ("Quality gate", "PASS"),
        ("Correlation", "PASS"),
        ("Held-out (22 scenarios)", "4.51%"),
    ]
    for label, val in rows:
        draw.text((x, R.y), label, font=f_small, fill=INK)
        draw.text((x + inner_w - 140, R.y), val, font=f_small, fill=PASS)
        R.y += 28
    R.y += 16

    half = (fig_w - 16) // 2
    y_figs = R.y
    for i, (path, num, cap) in enumerate(
        [
            (GRAPHS / "isef_parity_restyled.png", 2, "Ac-225 parity; 4.51% median vs ODE."),
            (GRAPHS / "isef_isotope_evolution.png", 3, "Ac-225 vs time; PINN tracks ODE."),
        ]
    ):
        fx = x + i * (half + 16)
        img = Image.open(path).convert("RGBA") if path.is_file() else None
        if img:
            img.thumbnail((half, fig_h_med), Image.Resampling.LANCZOS)
            ox = fx + (half - img.width) // 2
            canvas.paste(img, (ox, y_figs), img)
            draw.rectangle([fx, y_figs, fx + half, y_figs + img.height], outline=(120, 120, 120), width=1)
            cy = y_figs + img.height + 8
        else:
            cy = y_figs + 40
        for line in _wrap(f"Fig {num}: {cap}", 28):
            draw.text((fx, cy), line, font=_font(16), fill=MUTED)
            cy += 20
        R.y = max(R.y, cy + 12)

    R.y = _section(draw, x, R.y + 8, "Results & Conclusions", COL_W)
    R.y = _draw_paragraph(
        draw, x, R.y,
        "PINN passed 6/6 gates with 4.51% held-out error vs ODE — enabling rapid screening not practical with repeated stiff solves.",
        f_small, width=48,
    ) + 6
    _draw_paragraph(
        draw, x, R.y,
        "Limitation: ODE-only validation; 0D model; not clinical dosing or 3D transport.",
        f_small, width=48,
    )

    # ==================== FOOTER ====================
    fy = body_h
    draw.rectangle([0, fy, W, H], fill=(238, 242, 247))
    draw.line([(0, fy), (W, fy)], fill=RULE, width=2)
    draw.text((MARGIN, fy + 16), "KEY REFERENCES", font=_font(22, bold=True), fill=INK)
    refs = (
        "Raissi et al. (2019) Physics-informed neural networks. J. Comput. Phys. 378, 686–707.  ·  "
        "NNDC/NuDat; JENDL-4.0; DOE Isotope Program."
    )
    _draw_paragraph(draw, MARGIN, fy + 52, refs, _font(17), width=120)
    draw.text(
        (MARGIN, fy + 120),
        "Acknowledgements: Adult sponsor & science fair mentor. Faculty reviewers (pending).",
        font=_font(17), fill=MUTED,
    )
    draw.text((W - 520, fy + 24), "LIVE DEMO", font=_font(22, bold=True), fill=ACCENT)
    draw.text(
        (W - 520, fy + 58),
        "lhyjrhmwzxqfpuuwsux7zh.streamlit.app",
        font=_font(17), fill=ACCENT,
    )
    draw.text((W - 520, fy + 88), "github.com/samogunnubi0-del/PINN", font=_font(17), fill=ACCENT)

    canvas.save(OUT, "PNG", optimize=True)
    print(f"Saved {OUT.relative_to(ROOT)} ({OUT.stat().st_size // 1024} KB, {W}×{H}px)")


if __name__ == "__main__":
    main()
