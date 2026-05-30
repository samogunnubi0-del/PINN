"""
ISEF publication-grade figures for the IsotopePINN project.

Generates:
  - graphs/isef_isotope_evolution.png   (Ac-225 activity vs time + harvest window)
  - graphs/isef_loss_trajectory_12k.png (dual-axis loss + 12k projection)
  - graphs/isef_mass_conservation.png   (mass budget residuals)
  - graphs/isef_parity_restyled.png     (restyled pred vs true)

Run: python scripts/isef_figures.py
"""

from __future__ import annotations

import os
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import graph_provenance
from train import (
    TRAIN_INIT_AC225,
    TRAIN_INIT_AC227,
    TRAIN_INIT_RA225,
    TRAIN_INIT_RA226,
    TRAIN_INIT_RA227,
    prepare_loss_history_df_for_plot,
    prepare_training_tensors,
)
from pinn_model import (
    DEFAULT_N226_SCALE,
    DEFAULT_N225_SCALE,
    DEFAULT_NAC_SCALE,
    DEFAULT_N227_SCALE,
    DEFAULT_NAC227_SCALE,
    DEFAULT_PHI_SCALE,
    DEFAULT_T_REF_H,
    load_isotope_pinn_checkpoint,
    neutron_energy_ev_to_feature_numpy,
)
from ra226_ac225_transmutation import HALF_LIFE_AC225_H, IsotopeEnvironment, run_simulation
from scripts.figure_theme import DARK_RC, style_axes_dark, style_colorbar_dark

GRAPHS = ROOT / "graphs"
WEIGHTS = ROOT / "weights"
DATA = ROOT / "data"
RESULTS = ROOT / "results"
_RUN_ID = graph_provenance.new_run_id()

# ISEF formatting
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "grid.alpha": 0.35,
    }
)

LN2 = np.log(2.0)
LAMBDA_AC225_H = LN2 / HALF_LIFE_AC225_H
STRICT_AC227_IMPURITY_LIMIT = 0.0015  # 0.15% activity ratio


def _save(fig: plt.Figure, name: str, extra: dict | None = None) -> None:
    path = GRAPHS / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(DARK_RC["figure.facecolor"])
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    graph_provenance.record_graph_write(
        ROOT, path.resolve(), producer="isef_figures.py", run_id=_RUN_ID, extra=extra or {}
    )
    print(f"[isef_figures] saved {path.relative_to(ROOT)}")


def _style_figure(*axes) -> None:
    for ax in axes:
        style_axes_dark(ax, ax.figure)


def _pick_weights() -> pathlib.Path | None:
    for name in ("pinn_best_weights.pth", "pinn_trained_weights.pth"):
        p = WEIGHTS / name
        if p.is_file():
            return p
    return None


def _pinn_trajectory(
    model,
    t_hours: np.ndarray,
    phi: float,
    energy_ev: float,
    n226_0: float,
    n225_0: float = 0.0,
    nac_0: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (Ra226, Ra225, Ac225, Ra227, Ac227) atom trajectories from the PINN."""
    e_nn = float(neutron_energy_ev_to_feature_numpy(energy_ev))
    scales = np.array([
        DEFAULT_N226_SCALE, DEFAULT_N225_SCALE, DEFAULT_NAC_SCALE,
        DEFAULT_N227_SCALE, DEFAULT_NAC227_SCALE,
    ], dtype=np.float64)
    rows = []
    for th in t_hours:
        t_norm = th / DEFAULT_T_REF_H
        phi_nn = phi / DEFAULT_PHI_SCALE
        x = torch.tensor(
            [[
                t_norm,
                phi_nn,
                e_nn,
                n226_0 / DEFAULT_N226_SCALE,
                n225_0 / DEFAULT_N225_SCALE,
                nac_0 / DEFAULT_NAC_SCALE,
                0.0,
                0.0,
            ]],
            dtype=torch.float32,
        )
        with torch.no_grad():
            pred = model(x)[0].numpy()
        rows.append(pred[:5] * scales)
    arr = np.asarray(rows)
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]


def _ode_trajectory(
    t_hours: np.ndarray,
    phi: float,
    energy_ev: float,
    n226_0: float,
    n225_0: float = 0.0,
    nac_0: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    env = IsotopeEnvironment(phi=phi, neutron_energy_ev=energy_ev)
    t_h, Y = run_simulation(
        env,
        t_end_h=float(t_hours[-1]),
        n_points=max(200, len(t_hours) * 4),
        N_ra0=n226_0,
        N_ra225_0=n225_0,
        N_ac0=nac_0,
    )
    n226 = np.interp(t_hours, t_h, Y[:, 0])
    n225 = np.interp(t_hours, t_h, Y[:, 1])
    nac = np.interp(t_hours, t_h, Y[:, 2])
    return n226, n225, nac


def _activity_ac225(atoms: np.ndarray) -> np.ndarray:
    return LAMBDA_AC225_H * np.maximum(atoms, 0.0)


def plot_isotope_evolution(model) -> None:
    """Ac-225 activity vs time with ODE reference and harvest window."""
    phi = 1.0e14
    energy_ev = 14.0e6
    n226_0 = TRAIN_INIT_RA226
    t_hours = np.linspace(0.5, 400.0, 120)

    p226, p225, pac, _p227, _pac227 = _pinn_trajectory(model, t_hours, phi, energy_ev, n226_0)
    o226, o225, oac = _ode_trajectory(t_hours, phi, energy_ev, n226_0)
    act_p = _activity_ac225(pac)
    act_o = _activity_ac225(oac)

    peak_idx = int(np.argmax(act_o))
    t_peak = t_hours[peak_idx]

    fig, ax = plt.subplots(figsize=(9, 5))
    with plt.rc_context(DARK_RC):
        ax.plot(t_hours, act_o, color="#e2e8f0", ls="--", lw=2, label=r"ODE reference ($A_{^{225}\mathrm{Ac}}$)")
        ax.plot(t_hours, act_p, color="#2ecc71", lw=2, label="PINN prediction")
        ax.axvspan(max(0, t_peak - 24), min(t_hours[-1], t_peak + 48), alpha=0.15, color="#3498db",
                   label="Suggested harvest window")
        ax.set_xlabel(r"Time $t$ (h)")
        ax.set_ylabel(r"Activity $A$ (decays h$^{-1}$)")
        ax.set_title(r"$^{225}$Ac production — fast flux ($\phi=10^{14}$ n cm$^{-2}$ s$^{-1}$)")
        leg = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
        for t in leg.get_texts():
            t.set_color(DARK_RC["legend.labelcolor"])
        ax.grid(True, alpha=0.25)
        _style_figure(ax)
    _save(fig, "isef_isotope_evolution.png", {"phi": phi, "energy_ev": energy_ev})


def _load_loss_history() -> pd.DataFrame:
    csv_path = RESULTS / "loss_history.csv"
    if not csv_path.is_file():
        return pd.DataFrame()
    raw = pd.read_csv(csv_path)
    return prepare_loss_history_df_for_plot(raw)


def plot_loss_trajectory_12k() -> None:
    """Dual-axis loss history with linear extrapolation toward 12k epochs, cleaned and smoothed."""
    df = _load_loss_history()
    if df.empty:
        print("[isef_figures] skip loss trajectory — no loss_history.csv")
        return

    ep = df["epoch"].astype(float).values
    data_mse = df["data_mse"].astype(float).values
    phys_mse = df["physics_mse"].astype(float).values
    total = df.get("supervised_total", pd.Series(0.0, index=df.index)).astype(float).values + \
            df.get("unsupervised_total", pd.Series(0.0, index=df.index)).astype(float).values

    # Determine pretrain epoch boundary
    tp = 600.0
    if "phase" in df.columns and "pretrain" in df["phase"].values:
        tp = float(df[df["phase"] == "pretrain"]["epoch"].max())

    def ema(values: np.ndarray, alpha: float = 0.04) -> np.ndarray:
        if len(values) == 0:
            return values
        smoothed = []
        current = values[0]
        for val in values:
            if np.isnan(val):
                smoothed.append(np.nan)
            else:
                if np.isnan(current):
                    current = val
                current = alpha * val + (1.0 - alpha) * current
                smoothed.append(current)
        return np.asarray(smoothed)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    with plt.rc_context(DARK_RC):
        data_mse_joint = np.where(ep > tp, data_mse, np.nan)
        total_joint = np.where(ep > tp, total, np.nan)

        phys_smooth = ema(phys_mse, alpha=0.04)
        data_smooth = ema(data_mse_joint, alpha=0.04)
        total_smooth = ema(total_joint, alpha=0.04)

        ax1.semilogy(ep, phys_mse, color="#e74c3c", alpha=0.12, label="Raw Physics MSE")
        ax1.semilogy(ep, data_mse_joint, color="#3498db", alpha=0.12, label="Raw Data MSE")
        ax1.semilogy(ep, total_joint, color="#64748b", alpha=0.08)
        ax1.semilogy(ep, phys_smooth, color="#e74c3c", lw=2, label="Smoothed Physics MSE")
        ax1.semilogy(ep, data_smooth, color="#3498db", lw=2, label="Smoothed Data MSE")
        ax1.semilogy(ep, total_smooth, color="#94a3b8", lw=1.5, ls="--", label="Total Loss (Smoothed)")
        ax1.axvline(tp, color="#94a3b8", ls=":", lw=1.5, alpha=0.8, label="Joint Phase Start")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss (log scale)")
        ax1.set_title("Training Loss Trajectory", fontsize=12, fontweight="bold")
        ax1.legend(loc="best", fontsize=9)
        ax1.grid(True, which="both", alpha=0.25)

        joint = df[df["phase"] == "joint"] if "phase" in df.columns else df
        if len(joint) >= 5:
            y = joint["data_mse"].astype(float).values
            x = joint["epoch"].astype(float).values
            y_smooth = ema(y, alpha=0.04)
            n_fit = max(5, len(x) // 5)
            y_fit = np.maximum(y_smooth[-n_fit:], 1e-12)
            coef = np.polyfit(x[-n_fit:], np.log10(y_fit), 1)
            x_proj = np.array([x[-1], 12000.0])
            y_proj = 10 ** np.polyval(coef, x_proj)
            ax2.semilogy(x, y, color="#3498db", alpha=0.15, label="Raw Observed Data MSE")
            ax2.semilogy(x, y_smooth, "-", color="#3498db", lw=2, label="Smoothed Observed Data MSE")
            if coef[0] < 0:
                ax2.semilogy(x_proj, y_proj, "--", color="#9b59b6", lw=2,
                             label=r"Linear log-projection $\to$ 12k epochs")
            else:
                ax2.text(0.05, 0.05, "Not converged — projection omitted", transform=ax2.transAxes,
                         fontsize=9, color="#94a3b8")
            ax2.axvline(x[-1], color="#94a3b8", ls=":", alpha=0.6, label=f"Current end ({int(x[-1])})")

        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Data MSE (log scale)")
        ax2.set_title("12k Scalability Projection", fontsize=12, fontweight="bold")
        ax2.legend(loc="best", fontsize=9)
        ax2.grid(True, which="both", alpha=0.25)
        _style_figure(ax1, ax2)

    fig.tight_layout()
    _save(fig, "isef_loss_trajectory_12k.png", {"rows": len(df)})


def plot_mass_conservation(model) -> None:
    """Mass budget relative residual vs time for virgin Ra-226 + fast flux."""
    phi = 1.0e14
    energy_ev = 14.0e6
    n226_0 = TRAIN_INIT_RA226
    t_hours = np.linspace(1.0, 300.0, 80)
    n0 = n226_0

    p226, p225, pac, p227, pac227 = _pinn_trajectory(model, t_hours, phi, energy_ev, n226_0)
    # Full tracked inventory: Ra-226 -> Ra-225 -> Ac-225 and Ra-226 -> Ra-227 -> Ac-227
    # both keep atoms inside the 5-species set, so the sum should equal N0 up to the tiny
    # fraction that decays out of the chain (Ac-225/Ac-227 alpha). A 3-species sum would
    # falsely show ~1e-4 drift because it drops the Ra-227/Ac-227 branch.
    total = p226 + p225 + pac + p227 + pac227
    rel_res = (total - n0) / max(n0, 1.0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    with plt.rc_context(DARK_RC):
        ax.plot(t_hours, rel_res, color="#a78bfa", lw=2, label=r"$(N_\mathrm{tot}-N_0)/N_0$ (PINN, 5 species)")
        ax.axhline(0.0, color="#94a3b8", lw=0.8)
        ax.axhspan(-1e-5, 1e-5, alpha=0.2, color="#2ecc71", label=r"Target band $\pm10^{-5}$")
        ax.set_xlabel(r"Time $t$ (h)")
        ax.set_ylabel("Relative mass residual")
        ax.set_title("Mass conservation — Ra-226 feed + flux (5-species inventory)")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.25)
        _style_figure(ax)
    _save(fig, "isef_mass_conservation.png")


def _load_training_csv_for_parity() -> pd.DataFrame | None:
    """Load supervised rows; fall back to kaggle_kernel copy if data/ is empty."""
    candidates = [
        DATA / "pinn_training_data.csv",
        ROOT / "kaggle_kernel" / "data" / "pinn_training_data.csv",
    ]
    for path in candidates:
        if path.is_file():
            return pd.read_csv(path)
    return None


def _ensure_parity_init_columns(tdf: pd.DataFrame) -> pd.DataFrame:
    """Targets N_* are states at time t; inputs need init_N* (virgin IC if absent)."""
    out = tdf.copy()
    defaults = {
        "init_N226": TRAIN_INIT_RA226,
        "init_N225": TRAIN_INIT_RA225,
        "init_NAc": TRAIN_INIT_AC225,
        "init_N227": TRAIN_INIT_RA227,
        "init_NAc227": TRAIN_INIT_AC227,
    }
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default
    for col in ("N_Ra227", "N_Ac227"):
        if col not in out.columns:
            out[col] = 0.0
    return out


def plot_parity_restyled(model) -> None:
    """Restyled Ac-225 parity: PINN vs ODE targets using correct init_N* inputs."""
    tdf = _load_training_csv_for_parity()
    if tdf is None:
        print("[isef_figures] skip parity — no pinn_training_data.csv")
        return
    if not {"N_Ac225", "time", "phi", "energy"}.issubset(tdf.columns):
        print("[isef_figures] skip parity — missing columns")
        return

    tdf = _ensure_parity_init_columns(tdf)
    pos = tdf["N_Ac225"].astype(float) > 0
    sub = tdf.loc[pos]
    if len(sub) > 800:
        sub = sub.sample(n=800, random_state=42)

    device = torch.device("cpu")
    dtype = getattr(getattr(model, "daughter_rate_log_scales", None), "dtype", torch.float32)
    inputs, targets = prepare_training_tensors(sub, device, dtype=dtype)
    model.eval()
    with torch.no_grad():
        pred_norm = model(inputs)
    true_atoms = targets[:, 2].cpu().numpy()
    pred_atoms = (pred_norm[:, 2] * DEFAULT_NAC_SCALE).cpu().numpy()
    mask = (true_atoms > 0) & (pred_atoms > 0)
    if not mask.any():
        print("[isef_figures] skip parity — no positive Ac-225 pairs")
        return

    t_pos = true_atoms[mask]
    p_pos = pred_atoms[mask]
    med = float(np.median(np.abs(p_pos - t_pos) / t_pos))

    fig, ax = plt.subplots(figsize=(7, 7))
    with plt.rc_context(DARK_RC):
        sc = ax.scatter(t_pos, p_pos, c=np.log10(np.abs(p_pos - t_pos) / t_pos + 1e-12), s=12, alpha=0.55, cmap="viridis")
        lo = min(t_pos.min(), p_pos.min()) * 0.5
        hi = max(t_pos.max(), p_pos.max()) * 2.0
        ax.plot([lo, hi], [lo, hi], color="#e2e8f0", ls="--", lw=1, label=r"$y=x$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"True $N_{^{225}\mathrm{Ac}}$ (atoms)")
        ax.set_ylabel(r"Predicted $N_{^{225}\mathrm{Ac}}$ (atoms)")
        ax.set_title(r"$^{225}$Ac parity (ISEF restyled)")
        ax.text(0.05, 0.95, f"Median rel. error: {med:.2%}", transform=ax.transAxes, va="top", color="#e2e8f0")
        cbar = plt.colorbar(sc, ax=ax, label=r"$\log_{10}$ rel. error")
        style_colorbar_dark(cbar)
        ax.legend(loc="lower right")
        ax.grid(True, which="both", alpha=0.35)
        _style_figure(ax)
    _save(fig, "isef_parity_restyled.png", {"n_points": int(mask.sum()), "median_rel": med})


def main() -> None:
    GRAPHS.mkdir(parents=True, exist_ok=True)
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "app_evidence", ROOT / "scripts" / "app_evidence.py"
        )
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.save_poster_physics_story()
    except Exception as exc:
        print(f"[isef_figures] physics story plot skipped: {exc}")
    w = _pick_weights()
    if w is None:
        print("[isef_figures] no weights — loss/mass plots only")
        plot_loss_trajectory_12k()
        return
    model, _info = load_isotope_pinn_checkpoint(str(w), map_location="cpu")
    model.eval()
    plot_isotope_evolution(model)
    plot_loss_trajectory_12k()
    plot_mass_conservation(model)
    plot_parity_restyled(model)
    print("[isef_figures] done")


if __name__ == "__main__":
    main()
