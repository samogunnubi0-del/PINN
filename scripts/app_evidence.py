"""
Matplotlib evidence charts for the Streamlit app — emphasize physics-informed training.

Run standalone: python scripts/app_evidence.py
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from train import prepare_loss_history_df_for_plot  # noqa: E402
from scripts.figure_theme import DARK_RC, figure_to_png_bytes, style_axes_dark, style_colorbar_dark  # noqa: E402

ISEF_RC = DARK_RC

# Re-export for app.py
__all__ = [
    "figure_to_png_bytes",
    "compute_pinn_ode_trajectory",
    "figure_pinn_vs_ode_from_arrays",
    "benchmark_pinn_vs_ode_speed",
    "figure_flux_sensitivity_summary",
    "figure_failure_regimes",
]


def _ema(values: np.ndarray, alpha: float = 0.06) -> np.ndarray:
    if len(values) == 0:
        return values
    out: list[float] = []
    current = np.nan
    for val in values:
        if np.isnan(val):
            out.append(np.nan)
            continue
        if np.isnan(current):
            current = float(val)
        else:
            current = alpha * float(val) + (1.0 - alpha) * current
        out.append(current)
    return np.asarray(out, dtype=float)


def load_loss_history_df() -> pd.DataFrame:
    path = ROOT / "results" / "loss_history.csv"
    if not path.is_file():
        return pd.DataFrame()
    return prepare_loss_history_df_for_plot(pd.read_csv(path))


def _pretrain_boundary(df: pd.DataFrame) -> float:
    if df.empty:
        return 600.0
    if "phase" in df.columns and (df["phase"].astype(str).str.lower() == "pretrain").any():
        return float(df.loc[df["phase"].astype(str).str.lower() == "pretrain", "epoch"].max())
    return 600.0


def figure_loss_physics_story(df: pd.DataFrame | None = None) -> plt.Figure:
    """Three-panel: curriculum, raw MSE components, weighted supervised vs physics totals."""
    if df is None:
        df = load_loss_history_df()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    fig.patch.set_facecolor(ISEF_RC["figure.facecolor"])
    if df.empty:
        for ax in axes:
            ax.set_facecolor(ISEF_RC["axes.facecolor"])
            ax.text(0.5, 0.5, "No loss_history.csv", ha="center", va="center", color="#94a3b8")
        fig.tight_layout()
        return fig

    with plt.rc_context(ISEF_RC):
        ep = df["epoch"].astype(float).values
        tp = _pretrain_boundary(df)
        is_pre = ep <= tp

        phys = df["physics_mse"].astype(float).values
        data = df["data_mse"].astype(float).values.copy()
        data[is_pre & (data == 0.0)] = np.nan

        ax0, ax1, ax2 = axes
        for ax in axes:
            ax.set_facecolor(ISEF_RC["axes.facecolor"])
            ax.grid(True, which="both", alpha=0.25)

        # Panel 0 — curriculum timeline
        ax0.axvspan(ep.min(), tp, alpha=0.35, color="#e74c3c", label="Pretrain: physics only")
        ax0.axvspan(tp, ep.max(), alpha=0.25, color="#3498db", label="Joint: physics + ODE data")
        ax0.plot(ep, _ema(phys), color="#f87171", lw=2, label="Physics MSE (EMA)")
        ax0.set_yscale("log")
        ax0.set_xlabel("Epoch")
        ax0.set_ylabel("Physics MSE")
        ax0.set_title("① Physics-first curriculum", fontweight="bold", fontsize=11)
        ax0.legend(loc="upper right", fontsize=8)

        # Panel 1 — both MSE components (joint phase data visible)
        ax1.semilogy(ep, phys, color="#f87171", alpha=0.15)
        ax1.semilogy(ep, data, color="#38bdf8", alpha=0.15)
        ax1.semilogy(ep, _ema(phys), color="#f87171", lw=2.2, label="Bateman residual MSE")
        ax1.semilogy(ep, _ema(data), color="#38bdf8", lw=2.2, label="Supervised ODE data MSE")
        ax1.axvline(tp, color="#94a3b8", ls=":", lw=1.2)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Component MSE (log)")
        ax1.set_title("② Both losses drive training", fontweight="bold", fontsize=11)
        ax1.legend(loc="upper right", fontsize=8)

        # Panel 2 — weighted totals (what actually enters the optimizer)
        if {"supervised_total", "unsupervised_total"}.issubset(df.columns):
            sup = df["supervised_total"].astype(float).values.copy()
            uns = df["unsupervised_total"].astype(float).values
            sup[is_pre] = np.nan
            ax2.semilogy(ep, _ema(sup), color="#38bdf8", lw=2, label="Weighted data loss")
            ax2.semilogy(ep, _ema(uns), color="#f87171", lw=2, label="Weighted physics loss")
            ax2.axvline(tp, color="#94a3b8", ls=":", lw=1.2)
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Weighted loss (log)")
            ax2.set_title("③ Optimizer sees physics + data", fontweight="bold", fontsize=11)
            ax2.legend(loc="upper right", fontsize=8)
        else:
            ax2.text(0.5, 0.5, "Missing loss totals in CSV", ha="center", va="center", color="#94a3b8")

    fig.suptitle(
        "PINN training is not curve-fitting alone — 600 physics-only epochs, then joint Bateman + data",
        fontsize=12,
        fontweight="bold",
        color="#f8fafc",
        y=1.02,
    )
    fig.tight_layout()
    return fig


def figure_heldout_regimes() -> plt.Figure:
    path = ROOT / "analysis" / "validation" / "heldout_validation_summary.csv"
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor(ISEF_RC["figure.facecolor"])
    if not path.is_file():
        ax.text(0.5, 0.5, "No held-out summary CSV", ha="center", va="center", color="#94a3b8")
        return fig

    df = pd.read_csv(path)
    ac = df[(df["species"] == "Ac-225") & (df["case_type"] != "empty")].copy()
    if ac.empty:
        ac = df[df["species"] == "Ac-225"].copy()
    ac["label"] = ac["regime"].astype(str) + " · " + ac["case_type"].astype(str).str.replace("_", " ")
    ac = ac.sort_values("median_rel_error")
    colors = ["#34d399" if v <= 0.06 else "#fbbf24" if v <= 0.10 else "#f87171" for v in ac["median_rel_error"]]

    with plt.rc_context(ISEF_RC):
        ax.set_facecolor(ISEF_RC["axes.facecolor"])
        bars = ax.barh(ac["label"], 100.0 * ac["median_rel_error"], color=colors, edgecolor="#1b223c")
        ax.axvline(10.0, color="#94a3b8", ls="--", lw=1, label="10% gate")
        ax.set_xlabel("Ac-225 median error vs ODE (%)")
        ax.set_title("Held-out generalization by energy regime (v63)", fontweight="bold")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, axis="x", alpha=0.25)
        for bar, val in zip(bars, ac["median_rel_error"]):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{100*val:.1f}%", va="center", fontsize=9, color="#e2e8f0")
    fig.tight_layout()
    return fig


def compute_pinn_ode_trajectory(
    model,
    *,
    phi: float = 1e14,
    energy_ev: float = 14e6,
    hours: float = 300.0,
    n_points: int = 48,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batched PINN + ODE Ac-225 trajectories (fast path for Streamlit live demo)."""
    import torch
    from pinn_model import (
        DEFAULT_N226_SCALE,
        DEFAULT_N225_SCALE,
        DEFAULT_NAC_SCALE,
        DEFAULT_N227_SCALE,
        DEFAULT_NAC227_SCALE,
        DEFAULT_PHI_SCALE,
        DEFAULT_T_REF_H,
        neutron_energy_ev_to_feature_numpy,
    )
    from ra226_ac225_transmutation import IsotopeEnvironment, run_simulation

    n226_0 = 6.022e23
    t_hours = np.linspace(1.0, hours, int(n_points))
    e_nn = float(neutron_energy_ev_to_feature_numpy(energy_ev))
    n = len(t_hours)

    x = torch.tensor(
        np.column_stack([
            t_hours / DEFAULT_T_REF_H,
            np.full(n, phi / DEFAULT_PHI_SCALE),
            np.full(n, e_nn),
            np.full(n, n226_0 / DEFAULT_N226_SCALE),
            np.zeros(n),
            np.zeros(n),
            np.zeros(n),
            np.zeros(n),
        ]),
        dtype=torch.float32,
    )
    scales = np.array([
        DEFAULT_N226_SCALE, DEFAULT_N225_SCALE, DEFAULT_NAC_SCALE,
        DEFAULT_N227_SCALE, DEFAULT_NAC227_SCALE,
    ], dtype=np.float64)
    model.eval()
    with torch.no_grad():
        pac_pinn = (model(x).numpy()[:, 2] * scales[2])

    env = IsotopeEnvironment(phi=phi, neutron_energy_ev=energy_ev)
    t_h, Y = run_simulation(env, t_end_h=hours, n_points=max(80, n_points * 2), N_ra0=n226_0)
    pac_ode = np.interp(t_hours, t_h, Y[:, 2])
    return t_hours, pac_pinn, pac_ode


def figure_pinn_vs_ode_from_arrays(
    t_hours: np.ndarray,
    pac_pinn: np.ndarray,
    pac_ode: np.ndarray,
    *,
    phi: float = 1e14,
    energy_ev: float = 14e6,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4))
    with plt.rc_context(ISEF_RC):
        style_axes_dark(ax, fig)
        ax.semilogy(t_hours, np.maximum(pac_ode, 1.0), color="#e2e8f0", ls="--", lw=2,
                    label="Stiff ODE (Radau reference)")
        ax.semilogy(t_hours, np.maximum(pac_pinn, 1.0), color="#34d399", lw=2.5,
                    label="PINN (Bateman backbone + correction)")
        ax.set_xlabel("Irradiation time (h)")
        ax.set_ylabel(r"$N_{^{225}\mathrm{Ac}}$ (atoms)")
        e_mev = energy_ev / 1e6
        ax.set_title(f"PINN vs ODE — φ={phi:.0e} n/cm²/s, E={e_mev:g} MeV", fontweight="bold")
        leg = ax.legend(loc="best", fontsize=9)
        for t in leg.get_texts():
            t.set_color(ISEF_RC["legend.labelcolor"])
        ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    return fig


def figure_pinn_vs_ode_live(model, *, phi: float = 1e14, energy_ev: float = 14e6, hours: float = 300.0) -> plt.Figure:
    """PINN vs stiff ODE on one production track — not a scatter 'guess'."""
    t_hours, pac_pinn, pac_ode = compute_pinn_ode_trajectory(
        model, phi=phi, energy_ev=energy_ev, hours=hours,
    )
    return figure_pinn_vs_ode_from_arrays(t_hours, pac_pinn, pac_ode, phi=phi, energy_ev=energy_ev)


def figure_nn_vs_pinn_schematic() -> plt.Figure:
    """Side-by-side diagram: standard NN vs physics-informed PINN."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    fig.patch.set_facecolor(ISEF_RC["figure.facecolor"])

    def _box(ax, xy, w, h, label, color="#1e293b", edge="#475569"):
        rect = plt.Rectangle(xy, w, h, facecolor=color, edgecolor=edge, linewidth=1.5, zorder=2)
        ax.add_patch(rect)
        ax.text(xy[0] + w / 2, xy[1] + h / 2, label, ha="center", va="center",
                fontsize=9, color="#e2e8f0", zorder=3, wrap=True)

    def _arrow(ax, x0, y0, x1, y1, color="#64748b"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5), zorder=1)

    with plt.rc_context(ISEF_RC):
        for ax in axes:
            ax.set_facecolor(ISEF_RC["axes.facecolor"])
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.axis("off")

        ax0, ax1 = axes
        ax0.set_title("Standard neural network", fontweight="bold", fontsize=11, pad=10)
        _box(ax0, (0.8, 4.2), 1.6, 1.6, "Inputs\n(t, φ, E, N₀)")
        _box(ax0, (3.6, 4.0), 2.2, 2.0, "Hidden\nlayers")
        _box(ax0, (6.8, 4.2), 1.8, 1.6, "Outputs\nN(t)")
        _arrow(ax0, 2.4, 5.0, 3.6, 5.0)
        _arrow(ax0, 5.8, 5.0, 6.8, 5.0)
        _box(ax0, (3.2, 1.0), 3.6, 1.2, "Trained on data points only", color="#334155", edge="#64748b")
        ax0.text(5.0, 0.35, "Can fit training scatter yet violate decay laws",
                 ha="center", fontsize=8, color="#94a3b8")

        ax1.set_title("Physics-informed PINN (this project)", fontweight="bold", fontsize=11, pad=10)
        _box(ax1, (0.8, 4.2), 1.6, 1.6, "Inputs\n(t, φ, E, N₀)")
        _box(ax1, (3.6, 4.0), 2.2, 2.0, "MLP +\nBateman\nbackbone")
        _box(ax1, (6.8, 4.2), 1.8, 1.6, "Outputs\nN(t)")
        _arrow(ax1, 2.4, 5.0, 3.6, 5.0)
        _arrow(ax1, 5.8, 5.0, 6.8, 5.0)
        _box(ax1, (6.4, 7.2), 2.8, 1.3, "Bateman ODE\nresidual", color="#7f1d1d", edge="#f87171")
        _box(ax1, (3.2, 1.0), 3.6, 1.2, "Data loss + physics loss", color="#134e4a", edge="#34d399")
        _arrow(ax1, 7.7, 7.2, 7.7, 5.8, color="#f87171")
        ax1.text(5.0, 0.35, "Penalized when predictions break transmutation physics",
                 ha="center", fontsize=8, color="#94a3b8")

    fig.suptitle("Why a PINN is different from curve-fitting", fontsize=12, fontweight="bold",
                 color="#f8fafc", y=1.02)
    fig.tight_layout()
    return fig


def benchmark_pinn_vs_ode_speed(model, *, n_scenarios: int = 200, seed: int = 42) -> dict:
    """
    Time N random planning scenarios: one batched PINN forward vs N stiff ODE solves.
    Returns timings in milliseconds and speedup ratio for ISEF / fair display.
    """
    import time
    import torch
    from pinn_model import (
        DEFAULT_N226_SCALE,
        DEFAULT_N225_SCALE,
        DEFAULT_NAC_SCALE,
        DEFAULT_N227_SCALE,
        DEFAULT_NAC227_SCALE,
        DEFAULT_PHI_SCALE,
        DEFAULT_T_REF_H,
        neutron_energy_ev_to_feature_numpy,
    )
    from ra226_ac225_transmutation import IsotopeEnvironment, run_simulation

    n = max(10, int(n_scenarios))
    rng = np.random.default_rng(seed)
    log_flux = rng.uniform(12.0, 15.5, n)
    log_energy = rng.uniform(-2.0, 7.146, n)
    times_h = rng.uniform(10.0, 400.0, n)
    fluxes = 10.0 ** log_flux
    energies = 10.0 ** log_energy
    e_nn = neutron_energy_ev_to_feature_numpy(energies)
    n226_0 = 6.022e23

    x = torch.tensor(
        np.column_stack([
            times_h / DEFAULT_T_REF_H,
            fluxes / DEFAULT_PHI_SCALE,
            e_nn,
            np.full(n, n226_0 / DEFAULT_N226_SCALE),
            np.zeros(n),
            np.zeros(n),
            np.zeros(n),
            np.zeros(n),
        ]),
        dtype=torch.float32,
    )

    model.eval()
    with torch.no_grad():
        _ = model(x[: min(8, n)])
    t0 = time.perf_counter()
    with torch.no_grad():
        pinn_out = model(x).numpy()
    pinn_ms = (time.perf_counter() - t0) * 1000.0

    env0 = IsotopeEnvironment(phi=float(fluxes[0]), neutron_energy_ev=float(energies[0]))
    run_simulation(env0, t_end_h=float(times_h[0]), n_points=121, N_ra0=n226_0)

    t1 = time.perf_counter()
    ode_ac225 = np.empty(n, dtype=np.float64)
    for i in range(n):
        env = IsotopeEnvironment(phi=float(fluxes[i]), neutron_energy_ev=float(energies[i]))
        t_h, y = run_simulation(
            env,
            t_end_h=float(times_h[i]),
            n_points=121,
            N_ra0=n226_0,
        )
        ode_ac225[i] = float(y[-1, 2])
    ode_ms = (time.perf_counter() - t1) * 1000.0

    pinn_ac225 = np.maximum(pinn_out[:, 2] * DEFAULT_NAC_SCALE, 0.0)
    rel = np.abs(pinn_ac225 - ode_ac225) / np.maximum(ode_ac225, 1.0)
    speedup = ode_ms / max(pinn_ms, 1e-6)

    return {
        "n_scenarios": n,
        "pinn_ms": float(pinn_ms),
        "ode_ms": float(ode_ms),
        "speedup_x": float(speedup),
        "pinn_per_scenario_us": float(pinn_ms * 1000.0 / n),
        "ode_per_scenario_ms": float(ode_ms / n),
        "median_rel_err_ac225": float(np.median(rel)),
        "seed": seed,
    }


def figure_flux_sensitivity_summary() -> plt.Figure | None:
    """Bar chart from deferred sensitivity CSV (thermal flux sweep)."""
    path = ROOT / "results" / "isef_sensitivity_ode_deferred.csv"
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    with plt.rc_context(ISEF_RC):
        style_axes_dark(ax, fig)
        x = df["flux_multiplier"].astype(float).values
        med = 100.0 * df["median_rel_err"].astype(float).values
        ax.bar(x.astype(str), med, color="#38bdf8", edgecolor="#1b223c")
        ax.axhline(10.0, color="#94a3b8", ls="--", lw=1, label="10% gate")
        ax.set_xlabel("Flux multiplier (thermal baseline φ₀)")
        ax.set_ylabel("Ac-225 median error vs ODE (%)")
        ax.set_title("Flux sensitivity — PINN tracks ODE across φ (thermal E)", fontweight="bold")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def figure_failure_regimes(buckets: dict[str, float] | None = None) -> plt.Figure:
    """Where the surrogate struggles — held-out Ac-225 error by neutron regime."""
    if not buckets:
        v63 = ROOT / "results" / "v63_validation_20260530.json"
        if v63.is_file():
            import json
            buckets = json.loads(v63.read_text(encoding="utf-8")).get(
                "heldout_buckets_ac225_median_rel", {}
            )
        else:
            buckets = {}

    labels = [
        ("Fast 14 MeV virgin", buckets.get("fast14_virgin")),
        ("Thermal virgin", buckets.get("thermal_virgin")),
        ("Recycled (avg)", buckets.get("thermal_recycled")),
        ("Threshold ~6.4 MeV", buckets.get("threshold_virgin")),
        ("Epithermal virgin", buckets.get("epithermal_virgin")),
        ("Overall held-out", buckets.get("all")),
    ]
    rows = [(a, b) for a, b in labels if b is not None]
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.55 * len(rows))))
    with plt.rc_context(ISEF_RC):
        style_axes_dark(ax, fig)
        if not rows:
            ax.text(0.5, 0.5, "No regime bucket data", ha="center", va="center", color="#94a3b8")
            return fig
        names = [r[0] for r in rows]
        vals = [100.0 * float(r[1]) for r in rows]
        colors = ["#34d399" if v <= 6 else "#fbbf24" if v <= 10 else "#f87171" for v in vals]
        bars = ax.barh(names, vals, color=colors, edgecolor="#1b223c")
        ax.axvline(10.0, color="#94a3b8", ls="--", lw=1, label="10% validation gate")
        ax.set_xlabel("Ac-225 median error vs ODE (%)")
        ax.set_title("Failure geography — where to trust the surrogate", fontweight="bold")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, axis="x", alpha=0.25)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=9, color="#e2e8f0")
    fig.tight_layout()
    return fig


def save_poster_physics_story(out_name: str = "isef_physics_training_story.png") -> pathlib.Path | None:
    """Write combined physics evidence PNG for poster / static fallback."""
    df = load_loss_history_df()
    fig = figure_loss_physics_story(df)
    out = ROOT / "graphs" / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


if __name__ == "__main__":
    p = save_poster_physics_story()
    print(f"Saved {p}")
