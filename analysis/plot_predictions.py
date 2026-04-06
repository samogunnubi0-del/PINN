"""
Quick utilities to plot PINN predictions vs ODE ground-truth.
Run `run_quick_demo()` for a fast smoke-test that saves PNGs to ./analysis/figs.
"""
from __future__ import annotations

import os
import sys
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from pinn_model import (
    IsotopePINN,
    DEFAULT_N226_SCALE,
    DEFAULT_N225_SCALE,
    DEFAULT_NAC_SCALE,
    DEFAULT_PHI_SCALE,
    DEFAULT_T_REF_H,
    neutron_energy_ev_to_feature_numpy,
)
from ra226_ac225_transmutation import IsotopeEnvironment, run_simulation

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG_DIR = pathlib.Path(__file__).resolve().parent / "figs"
WEIGHTS_PATH = ROOT / "pinn_trained_weights.pth"

# Align plots with train.SINGLE_SUPPLY_MODE (virgin Ra-226 → Ac-225 narrative).
SINGLE_SUPPLY_PLOTS = True


def load_model(device: torch.device = torch.device("cpu")) -> IsotopePINN:
    model = IsotopePINN()
    if WEIGHTS_PATH.exists():
        try:
            st = torch.load(WEIGHTS_PATH, map_location=device)
            model.load_state_dict(st)
        except Exception:
            try:
                # attempt loading keys-only state dict
                model.load_state_dict(st)
            except Exception:
                print("Warning: failed to load weights cleanly; using uninitialized model.")
    else:
        print(f"No weights found at {WEIGHTS_PATH}; using untrained model.")
    model.to(device)
    model.eval()
    return model


def evaluate_pinn(
    model: IsotopePINN,
    phi: float,
    energy_ev: float,
    times: np.ndarray,
    n226_0: float,
    n225_0: float,
    nac_0: float,
    use_raw: bool = False,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    t_nn = times / float(DEFAULT_T_REF_H)
    phi_nn = float(phi) / float(DEFAULT_PHI_SCALE)
    energy = neutron_energy_ev_to_feature_numpy(
        np.full_like(times, float(energy_ev), dtype=float)
    )
    n0_226_nn = float(n226_0) / float(DEFAULT_N226_SCALE)
    n0_225_nn = float(n225_0) / float(DEFAULT_N225_SCALE)
    n0_ac_nn = float(nac_0) / float(DEFAULT_NAC_SCALE)

    inputs = np.vstack(
        [
            t_nn,
            np.full_like(times, phi_nn),
            energy,
            np.full_like(times, n0_226_nn),
            np.full_like(times, n0_225_nn),
            np.full_like(times, n0_ac_nn),
        ]
    ).T

    x = torch.tensor(inputs, dtype=torch.float32, device=device)
    with torch.no_grad():
        if use_raw and hasattr(model, "forward_raw"):
            pred_nn = model.forward_raw(x)
        else:
            pred_nn = model(x)
    pred = pred_nn.cpu().numpy()
    pred_atoms = np.stack(
        [pred[:, 0] * DEFAULT_N226_SCALE, pred[:, 1] * DEFAULT_N225_SCALE, pred[:, 2] * DEFAULT_NAC_SCALE],
        axis=1,
    )
    return pred_atoms


def simulate_ode(phi: float, energy_ev: float, times: np.ndarray, n226_0: float, n225_0: float, nac_0: float) -> tuple[np.ndarray, np.ndarray]:
    t_max = float(np.max(times))
    env = IsotopeEnvironment(phi=phi, sigma_ra226=1e-24, neutron_energy_ev=energy_ev)
    n_points = max(200, int(t_max * 5) + 10)
    t_h, Y = run_simulation(env, t_end_h=t_max, n_points=n_points, N_ra0=n226_0, N_ra225_0=n225_0, N_ac0=nac_0)
    times = np.asarray(times, dtype=float)
    Y_interp = np.vstack([np.interp(times, t_h, Y[:, i]) for i in range(3)]).T
    return times, Y_interp


def plot_case(
    model: IsotopePINN,
    name: str,
    phi: float,
    energy_ev: float,
    times: np.ndarray,
    n226_0: float,
    n225_0: float,
    nac_0: float,
    outdir: pathlib.Path = FIG_DIR,
    device: torch.device = torch.device("cpu"),
) -> pathlib.Path:
    outdir.mkdir(parents=True, exist_ok=True)
    pred_atoms = evaluate_pinn(model, phi, energy_ev, times, n226_0, n225_0, nac_0, use_raw=False, device=device)
    _, y_true = simulate_ode(phi, energy_ev, times, n226_0, n225_0, nac_0)

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True, layout="constrained")
    species = [("Ra-226", 0), ("Ra-225", 1), ("Ac-225", 2)]
    colors = ["#2563eb", "#ea580c", "#16a34a"]
    for (label, idx), ax, c in zip(species, axes, colors):
        ax.plot(times, y_true[:, idx], "k-", lw=2.2, label="ODE (reference)")
        ax.plot(times, pred_atoms[:, idx], "--", color=c, lw=2.2, label="PINN")
        ax.set_ylabel(f"{label}\n(atoms)", fontsize=11)
        ax.set_yscale("log")
        ax.legend(loc="best", framealpha=0.92, fontsize=10)
        ax.grid(True, which="both", ls="--", alpha=0.35)
        ax.tick_params(axis="both", labelsize=10)
        pt = pred_atoms[:, idx]
        yt = y_true[:, idx]
        pos = np.concatenate([pt[np.isfinite(pt) & (pt > 0)], yt[np.isfinite(yt) & (yt > 0)]])
        if pos.size >= 1:
            lo = float(np.min(pos)) * 0.25
            hi = float(np.max(pos)) * 4.0
            if lo < hi and lo > 0:
                ax.set_ylim(lo, hi)

    axes[-1].set_xlabel("Time (hours)", fontsize=11)
    fig.suptitle(f"PINN vs ODE — {name.replace('_', ' ')}", fontsize=13, fontweight="600")
    outpath = outdir / f"pred_vs_true_{name}.png"
    fig.savefig(outpath, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return outpath


def run_quick_demo():
    device = torch.device("cpu")
    model = load_model(device)
    times = np.linspace(0.0, 300.0, 31)

    if SINGLE_SUPPLY_PLOTS:
        scenarios = [
            ("ra226_dom_reference_supply", 1e14, 0.025, 6.022e23, 0.0, 0.0),
        ]
    else:
        scenarios = [
            ("ra225_dom", 0.0, 0.025, 0.0, 1e18, 0.0),
            ("ra226_dom", 1e14, 0.025, 6.022e23, 0.0, 0.0),
            ("low_flux_mixed", 1e12, 0.04, 1e20, 1e18, 1e18),
        ]

    created: list[pathlib.Path] = []
    for name, phi, e_ev, n226_0, n225_0, nac_0 in scenarios:
        p = plot_case(model, name, phi, e_ev, times, n226_0, n225_0, nac_0)
        print(f"Saved: {p}")
        created.append(p)
    return created


if __name__ == "__main__":
    run_quick_demo()
