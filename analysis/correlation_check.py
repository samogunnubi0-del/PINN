"""
Post-training sanity check: PINN vs ODE correlation on the scenarios we claim.

Run after train.py + weights saved:
  python analysis/correlation_check.py

Exits with code 1 if in-distribution reference scenario is weak (guards overclaims).
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import train as train_cfg  # noqa: E402
from analysis.plot_predictions import (  # noqa: E402
    evaluate_pinn,
    load_model,
    simulate_ode,
)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0:
        return 1.0 if ss_res <= 0 else 0.0
    return 1.0 - ss_res / ss_tot


def main() -> None:
    device = torch.device("cpu")
    model = load_model(device)
    times = np.linspace(0.0, 300.0, 61)

    # Primary claim (matches plot_predictions SINGLE_SUPPLY + training focus)
    primary = ("reference_Ra226_supply", 1e14, 0.025, 6.022e23, 0.0, 0.0)
    name, phi, e_ev, n226, n225, nac = primary
    pred = evaluate_pinn(model, phi, e_ev, times, n226, n225, nac, device=device)
    _, truth = simulate_ode(phi, e_ev, times, n226, n225, nac)

    print("=== Correlation / fit vs ODE (same inputs as dashboard reference plot) ===")
    print(f"Scenario: {name}  phi={phi:g}  E={e_ev} eV  IC=({n226:.3e}, {n225:g}, {nac:g})\n")

    worst_r2 = 1.0
    for i, lab in enumerate(["Ra-226", "Ra-225", "Ac-225"]):
        r2 = r2_score(truth[:, i], pred[:, i])
        rel = np.abs(pred[:, i] - truth[:, i]) / np.maximum(np.abs(truth[:, i]), 1.0)
        worst_r2 = min(worst_r2, r2)
        print(f"  {lab}: R² = {r2:.6f}   max|rel err| = {np.max(rel):.4e}")

    # Loose threshold: trained model should explain most variance on in-distribution path
    ok = worst_r2 >= 0.85
    print()
    if ok:
        print("PASS: reference supply scenario tracks ODE well enough for the stated claim.")
    else:
        print(
            "WARN: R² below 0.85 on at least one species — do not claim tight clinical/industrial "
            "agreement; retrain longer or check weights."
        )

    if getattr(train_cfg, "SINGLE_SUPPLY_MODE", False):
        print("\n(SINGLE_SUPPLY_MODE: scenarios outside Ra-226 fuel + CSV flux are out-of-distribution.)")

    # Optional OOD probe (informational only)
    ood = ("OOD_Ra225_decay", 0.0, 0.025, 0.0, 1e18, 0.0)
    name2, phi2, e2, a, b, c = ood
    p2 = evaluate_pinn(model, phi2, e2, times, a, b, c, device=device)
    _, t2 = simulate_ode(phi2, e2, times, a, b, c)
    r2_ac = r2_score(t2[:, 2], p2[:, 2])
    print(f"\nOOD probe {name2} (not single-supply trained): Ac-225 R² = {r2_ac:.6f} (expect worse)")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
