"""Shared dark matplotlib theme for Streamlit and poster figures."""

from __future__ import annotations

import io

import matplotlib.pyplot as plt

DARK_RC = {
    "figure.facecolor": "#0f1322",
    "axes.facecolor": "#0f1322",
    "axes.edgecolor": "#334155",
    "axes.labelcolor": "#e2e8f0",
    "axes.titlecolor": "#f1f5f9",
    "text.color": "#e2e8f0",
    "xtick.color": "#cbd5e1",
    "ytick.color": "#cbd5e1",
    "grid.color": "#334155",
    "legend.facecolor": "#0f1322",
    "legend.edgecolor": "#475569",
    "legend.labelcolor": "#e2e8f0",
    "font.size": 10,
}


def style_axes_dark(ax, fig: plt.Figure | None = None) -> None:
    """Apply dark-theme colors to one axes (and optional figure)."""
    if fig is not None:
        fig.patch.set_facecolor(DARK_RC["figure.facecolor"])
    ax.set_facecolor(DARK_RC["axes.facecolor"])
    ax.title.set_color(DARK_RC["axes.titlecolor"])
    ax.xaxis.label.set_color(DARK_RC["axes.labelcolor"])
    ax.yaxis.label.set_color(DARK_RC["axes.labelcolor"])
    ax.tick_params(colors=DARK_RC["xtick.color"], which="both")
    for spine in ax.spines.values():
        spine.set_color(DARK_RC["axes.edgecolor"])


def style_colorbar_dark(cbar) -> None:
    cbar.ax.yaxis.label.set_color(DARK_RC["axes.labelcolor"])
    cbar.ax.tick_params(colors=DARK_RC["xtick.color"])


def figure_to_png_bytes(fig: plt.Figure, *, dpi: int = 100) -> bytes:
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=dpi,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )
    plt.close(fig)
    return buf.getvalue()
