"""Audit poster graphs: compare PINN vs ODE under each plot's scenario."""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pinn_model import DEFAULT_NAC_SCALE, DEFAULT_N226_SCALE, load_isotope_pinn_checkpoint
from ra226_ac225_transmutation import IsotopeEnvironment, run_simulation
from train import TRAIN_INIT_RA226, prepare_training_tensors, TRAIN_INIT_RA225, TRAIN_INIT_AC225, TRAIN_INIT_RA227, TRAIN_INIT_AC227

GRAPHS = ROOT / "graphs"
WEIGHTS = ROOT / "weights" / "pinn_best_weights.pth"


def _load_model():
    model, _ = load_isotope_pinn_checkpoint(str(WEIGHTS), map_location="cpu")
    model.eval()
    return model


def _ode_ac225(phi, e_ev, t_h, n226_0, n225_0=0.0, nac_0=0.0) -> float:
    env = IsotopeEnvironment(phi=phi, neutron_energy_ev=e_ev)
    th, y = run_simulation(env, t_end_h=t_h, n_points=400, N_ra0=n226_0, N_ra225_0=n225_0, N_ac0=nac_0)
    return float(np.interp(t_h, th, y[:, 2]))


def _pinn_ac225(model, phi, e_ev, t_h, n226_0, n225_0=0.0, nac_0=0.0) -> float:
    from scripts.isef_figures import _pinn_trajectory

    _, _, pac = _pinn_trajectory(model, np.array([t_h]), phi, e_ev, n226_0, n225_0, nac_0)
    return float(pac[0])


def _parity_median(model) -> float:
    csv = ROOT / "kaggle_kernel" / "data" / "pinn_training_data.csv"
    tdf = pd.read_csv(csv)
    for col, d in [
        ("init_N226", TRAIN_INIT_RA226),
        ("init_N225", TRAIN_INIT_RA225),
        ("init_NAc", TRAIN_INIT_AC225),
        ("init_N227", TRAIN_INIT_RA227),
        ("init_NAc227", TRAIN_INIT_AC227),
    ]:
        if col not in tdf.columns:
            tdf[col] = d
    sub = tdf[tdf["N_Ac225"] > 0].sample(800, random_state=42)
    inp, tgt = prepare_training_tensors(sub, torch.device("cpu"))
    with torch.no_grad():
        pred = model(inp)
    true = tgt[:, 2].numpy()
    pred_ac = (pred[:, 2] * DEFAULT_NAC_SCALE).numpy()
    return float(np.median(np.abs(pred_ac - true) / true))


def main() -> None:
    model = _load_model()
    print("=== Graph audit (v63 weights) ===\n")

    # isef_isotope_evolution scenario
    phi, e_ev = 1e14, 14e6
    for label, n226 in [("isef_evolution (1e22 IC)", 1e22), ("train default IC", TRAIN_INIT_RA226)]:
        errs = []
        for t in [50.0, 150.0, 300.0]:
            o = _ode_ac225(phi, e_ev, t, n226)
            p = _pinn_ac225(model, phi, e_ev, t, n226)
            errs.append(abs(p - o) / max(o, 1.0))
        print(f"{label}: median Ac-225 rel err @ 50/150/300h = {[f'{e:.2%}' for e in errs]}")

    med_parity = _parity_median(model)
    print(f"\nparity (correct init_N*): median rel = {med_parity:.2%}")

    # Buggy parity IC demo
    row = pd.read_csv(ROOT / "kaggle_kernel" / "data" / "pinn_training_data.csv").iloc[0]
    o = float(row["N_Ac225"])
    p_bug = _pinn_ac225(model, row["phi"], row["energy"], row["time"], row["N_Ra226"], row["N_Ra225"], row["N_Ac225"])
    p_ok = _pinn_ac225(model, row["phi"], row["energy"], row["time"], TRAIN_INIT_RA226, 0, 0)
    print(f"\nrow0 true={o:.3e} | buggy_IC_pred={p_bug:.3e} ({abs(p_bug-o)/o:.1%}) | correct_IC={p_ok:.3e} ({abs(p_ok-o)/o:.1%})")

    print("\nGraph files present:")
    for p in sorted(GRAPHS.glob("*.png")):
        print(f"  {p.name} ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
