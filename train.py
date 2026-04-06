"""
Train IsotopePINN on pinn_training_data.csv with Adam and physics-informed loss.

Max-Fix v2: 12,000 total epochs (2,000 physics pretrain + 10,000 joint),
1/v energy scaling, Ra-225 physics residual x5, non-negativity penalty,
secular equilibrium ceiling, 30% empty-tank collocation, diverse ICs enabled,
and Time/Flux normalized to [0, 1].

Speed env vars (hardware only, no data/epoch changes):
  PINN_FAST_CPU=1         all CPU threads for PyTorch, larger joint chunks
  PINN_JOINT_CHUNK=0      full-batch joint step (needs ~16GB RAM)
  PINN_ODE_PARALLEL=0     disable multi-process ODE cache on Windows
"""

from __future__ import annotations

import os
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pinn_model import (
    IsotopePINN,
    THERMAL_REFERENCE_EV,
    compute_physics_loss,
    neutron_energy_ev_to_feature_torch,
)
from ra226_ac225_transmutation import IsotopeEnvironment, run_simulation

DATA_PATH = pathlib.Path(__file__).resolve().parent / "pinn_training_data.csv"
WEIGHTS_PATH = pathlib.Path(__file__).resolve().parent / "pinn_trained_weights.pth"
LOSS_PLOT_PATH = pathlib.Path(__file__).resolve().parent / "pinn_loss_history.png"
AC225_SCATTER_PATH = pathlib.Path(__file__).resolve().parent / "pinn_ac225_pred_vs_true.png"

# --- Scaling (MUST match pinn_model.py; 0-1 normalised for Time & Flux) ------
N226_SCALE = 6.022e23
N225_SCALE = 1e20
NAC_SCALE = 1e20
PHI_SCALE = 1e15       # max flux -> phi_nn in [0, 1]
TIME_SCALE_H = 500.0   # max irradiation time -> t_nn in [0, 1]
TRAIN_INIT_RA226 = 6.022e23
TRAIN_INIT_RA225 = 0.0
TRAIN_INIT_AC225 = 0.0

# Time-shift augmentation
USE_TIME_SHIFT_AUGMENT = True
AUGMENT_PER_ROW = 2
MIN_T_REM_H = 0.05
TRAJ_CACHE_MAX = 4096
AUGMENT_BASE_ROW_LIMIT = 0
ODE_PREP_MAX_WORKERS = 0

# Enable diverse ICs so the network learns decay-only and mixed scenarios
SINGLE_SUPPLY_MODE = False
INVERTED_IC_N_EXTRA = 400
DIVERSE_IC_N_EXTRA = 500

# --- 12,000 total epochs: 2,000 pretrain + 10,000 joint ----------------------
EPOCHS = 10_000
PHYS_PRETRAIN_EPOCHS = 2_000
COLLOCATION_POINTS = 600
LR = 1e-3
LOG_EVERY = 500
LR_PLATEAU_PATIENCE = 500
LR_PLATEAU_FACTOR = 0.5

# Loss weights
PHYSICS_WEIGHT = 2_000.0
DATA_WEIGHT = 50.0
DATA_SPECIES_WEIGHTS = (1.0, 80.0, 80.0)
RA225_PHYSICS_WEIGHT = 5.0          # Ra-225 Bateman residual multiplier
NON_NEG_WEIGHT = 50.0               # penalty for negative predictions
SECULAR_EQ_WEIGHT = 25.0            # Ac-225 transient equilibrium ceiling
MAX_GRAD_NORM = 5.0
D_T_INPUT_D_T_HOURS = 1.0 / TIME_SCALE_H

# Joint training stability
WARMUP_EPOCHS = 1_000
GRAD_CLIP_NORM = 5.0
NAN_PATIENCE = 3
BEST_CKPT_PATH = pathlib.Path(__file__).resolve().parent / "pinn_best_weights.pth"
MASS_WEIGHT = 350.0
PHYS_PRETRAIN_LR = 1.0e-4
PHYS_PRETRAIN_PHYS_WEIGHT = 1_000.0
PHYS_PRETRAIN_MASS_WEIGHT = 800.0
PHYS_PRETRAIN_FUEL_ANCHOR_WEIGHT = 40.0
FUEL_ANCHOR_WEIGHT = 100.0
EMPTY_FEED_FRACTION = 0.30          # 30% empty-tank collocation points
EMPTY_FEED_HIGH_FLUX = 1.0e15

# Joint chunk size
_jc_env = os.environ.get("PINN_JOINT_CHUNK", "").strip()
if _jc_env == "":
    JOINT_CHUNK_SIZE = 512
elif _jc_env.lower() in ("0", "full", "none"):
    JOINT_CHUNK_SIZE = 0
else:
    JOINT_CHUNK_SIZE = max(32, int(_jc_env))


def _traj_cache_key(phi: float, e_ev: float, t_end: float) -> tuple[float, float, float]:
    return (round(phi, 8), round(float(e_ev), 12), round(t_end, 8))


def _reference_traj_worker(
    args: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[np.ndarray, np.ndarray]]:
    phi, e_ev, t_end = args
    key = _traj_cache_key(phi, e_ev, t_end)
    env = IsotopeEnvironment(phi=phi, sigma_ra226=1e-24, neutron_energy_ev=e_ev)
    n_pts = max(400, int(np.clip(t_end * 8, 400, 15000)))
    t_h, Y = run_simulation(
        env, t_end_h=t_end, n_points=n_pts,
        N_ra0=TRAIN_INIT_RA226, N_ra225_0=TRAIN_INIT_RA225, N_ac0=TRAIN_INIT_AC225,
    )
    return key, (t_h, Y)


def _collect_unique_reference_triples(df: pd.DataFrame) -> list[tuple[float, float, float]]:
    pe_t = df[["phi", "energy", "time"]]
    seen_keys: set[tuple[float, float, float]] = set()
    out: list[tuple[float, float, float]] = []
    for row in pe_t.itertuples(index=False, name=None):
        phi, e_ev, t_full = float(row[0]), float(row[1]), float(row[2])
        k = _traj_cache_key(phi, e_ev, t_full)
        if k in seen_keys:
            continue
        seen_keys.add(k)
        out.append((phi, e_ev, t_full))
    return out


def build_reference_traj_cache_parallel(
    df: pd.DataFrame, *, max_workers: int,
) -> dict[tuple[float, float, float], tuple[np.ndarray, np.ndarray]]:
    triples = _collect_unique_reference_triples(df)
    if not triples:
        return {}
    n = len(triples)
    workers = max(1, min(max_workers, n))
    cache: dict[tuple[float, float, float], tuple[np.ndarray, np.ndarray]] = {}
    if workers == 1:
        for args in triples:
            key, val = _reference_traj_worker(args)
            cache[key] = val
        return cache
    chunksize = max(1, n // (workers * 4))
    try:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for key, val in ex.map(_reference_traj_worker, triples, chunksize=chunksize):
                cache[key] = val
    except BrokenProcessPool:
        cache.clear()
        for args in triples:
            key, val = _reference_traj_worker(args)
            cache[key] = val
    return cache


def get_trajectory(
    phi: float, e_ev: float, t_end: float,
    cache: dict[tuple[float, float, float], tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    key = _traj_cache_key(phi, e_ev, t_end)
    if key in cache:
        return cache[key]
    env = IsotopeEnvironment(phi=phi, sigma_ra226=1e-24, neutron_energy_ev=e_ev)
    n_pts = max(400, int(np.clip(t_end * 8, 400, 15000)))
    t_h, Y = run_simulation(
        env, t_end_h=t_end, n_points=n_pts,
        N_ra0=TRAIN_INIT_RA226, N_ra225_0=TRAIN_INIT_RA225, N_ac0=TRAIN_INIT_AC225,
    )
    if len(cache) < TRAJ_CACHE_MAX:
        cache[key] = (t_h, Y)
    return t_h, Y


def state_at_time(t_h: np.ndarray, Y: np.ndarray, t_query: float) -> np.ndarray:
    t_q = float(np.clip(t_query, float(t_h[0]), float(t_h[-1])))
    return np.array([float(np.interp(t_q, t_h, Y[:, i])) for i in range(3)], dtype=np.float64)


def augment_rows_time_shift(
    df: pd.DataFrame,
    rng: np.random.Generator,
    augment_per_row: int,
    include_unshifted_base: bool = True,
    *,
    initial_cache: dict[tuple[float, float, float], tuple[np.ndarray, np.ndarray]] | None = None,
) -> pd.DataFrame:
    traj_cache: dict[tuple[float, float, float], tuple[np.ndarray, np.ndarray]] = {}
    if initial_cache:
        traj_cache.update(initial_cache)
    rows_out: list[dict[str, float]] = []

    pe_t = df[["phi", "energy", "time"]]
    for row in pe_t.itertuples(index=False, name=None):
        phi = float(row[0])
        e_ev = float(row[1])
        t_full = float(row[2])

        t_h, Y = get_trajectory(phi, e_ev, t_full, traj_cache)
        n_at_T = Y[-1].astype(np.float64)

        if include_unshifted_base:
            rows_out.append({
                "phi": phi, "energy": e_ev, "time": t_full,
                "init_N226": TRAIN_INIT_RA226, "init_N225": TRAIN_INIT_RA225,
                "init_NAc": TRAIN_INIT_AC225,
                "N_Ra226": n_at_T[0], "N_Ra225": n_at_T[1], "N_Ac225": n_at_T[2],
            })

        for _ in range(augment_per_row):
            delta = float(rng.uniform(0.0, max(t_full - MIN_T_REM_H, 1e-9)))
            t_rem = t_full - delta
            if t_rem < MIN_T_REM_H:
                continue
            n_delta = state_at_time(t_h, Y, delta)
            rows_out.append({
                "phi": phi, "energy": e_ev, "time": t_rem,
                "init_N226": n_delta[0], "init_N225": n_delta[1], "init_NAc": n_delta[2],
                "N_Ra226": n_at_T[0], "N_Ra225": n_at_T[1], "N_Ac225": n_at_T[2],
            })
    return pd.DataFrame(rows_out)


def augment_inverted_ic_scenarios(rng: np.random.Generator, n_extra: int = 400) -> pd.DataFrame:
    """Ra-225 dominant, Ra-226 low: pure decay / low-flux scenarios."""
    rows_out: list[dict[str, float]] = []
    for _ in range(n_extra):
        ra225_0 = float(rng.uniform(5e17, 2e19))
        ra226_0 = float(rng.uniform(0.0, 1e17))
        ac_0 = 0.0
        phi = 0.0 if rng.uniform() < 0.7 else float(rng.uniform(1e12, 1e13))
        energy_ev = float(rng.uniform(0.015, 0.08))
        time_h = float(rng.uniform(20.0, 300.0))

        env = IsotopeEnvironment(phi=phi, sigma_ra226=1e-24, neutron_energy_ev=energy_ev)
        t_h, Y = run_simulation(
            env, t_end_h=time_h, n_points=max(200, int(time_h * 5)),
            N_ra0=ra226_0, N_ra225_0=ra225_0, N_ac0=ac_0,
        )
        n_final = Y[-1].astype(np.float64)
        rows_out.append({
            "phi": phi, "energy": energy_ev, "time": time_h,
            "init_N226": ra226_0, "init_N225": ra225_0, "init_NAc": ac_0,
            "N_Ra226": n_final[0], "N_Ra225": n_final[1], "N_Ac225": n_final[2],
        })
        for shift_frac in [0.25, 0.5]:
            t_shift = time_h * shift_frac
            n_shifted = state_at_time(t_h, Y, t_shift)
            t_remaining = time_h - t_shift
            if t_remaining > MIN_T_REM_H:
                rows_out.append({
                    "phi": phi, "energy": energy_ev, "time": t_remaining,
                    "init_N226": n_shifted[0], "init_N225": n_shifted[1],
                    "init_NAc": n_shifted[2],
                    "N_Ra226": n_final[0], "N_Ra225": n_final[1], "N_Ac225": n_final[2],
                })
    return pd.DataFrame(rows_out)


def augment_diverse_ic_scenarios(rng: np.random.Generator, n_extra: int = 500) -> pd.DataFrame:
    """Diverse ICs: Ra225-dominant, Ra226-dominant, mixed, Ac-dominant."""
    rows_out: list[dict[str, float]] = []
    for _ in range(n_extra):
        regime_choice = float(rng.uniform())
        if regime_choice < 0.5:
            regime = "ra225_dom"
        elif regime_choice < 0.75:
            regime = "ra226_dom"
        elif regime_choice < 0.9:
            regime = "mixed"
        else:
            regime = "ac_dom"

        if regime == "ra226_dom":
            ra226_0 = float(rng.uniform(1e23, 1e24))
            ra225_0 = float(rng.uniform(0.0, 1e18))
            ac_0 = float(rng.uniform(0.0, 1e17))
        elif regime == "ra225_dom":
            ra225_0 = float(rng.uniform(1e18, 5e19))
            ra226_0 = float(rng.uniform(0.0, 1e17))
            ac_0 = float(rng.uniform(0.0, 1e17))
        elif regime == "ac_dom":
            ac_0 = float(rng.uniform(1e18, 1e20))
            ra225_0 = float(rng.uniform(0.0, 1e18))
            ra226_0 = float(rng.uniform(0.0, 1e17))
        else:
            ra226_0 = float(rng.uniform(1e17, 1e24))
            ra225_0 = float(rng.uniform(1e17, 1e20))
            ac_0 = float(rng.uniform(1e16, 1e20))

        phi = float(rng.uniform(0.0, 1e15))
        energy_ev = float(rng.uniform(0.015, 0.08))
        time_h = float(rng.uniform(10.0, 300.0))

        env = IsotopeEnvironment(phi=phi, sigma_ra226=1e-24, neutron_energy_ev=energy_ev)
        t_h, Y = run_simulation(
            env, t_end_h=time_h, n_points=max(200, int(time_h * 5)),
            N_ra0=ra226_0, N_ra225_0=ra225_0, N_ac0=ac_0,
        )
        n_final = Y[-1].astype(np.float64)
        rows_out.append({
            "phi": phi, "energy": energy_ev, "time": time_h,
            "init_N226": ra226_0, "init_N225": ra225_0, "init_NAc": ac_0,
            "N_Ra226": n_final[0], "N_Ra225": n_final[1], "N_Ac225": n_final[2],
        })
        for shift_frac in [0.3, 0.6]:
            t_shift = time_h * shift_frac
            n_shifted = state_at_time(t_h, Y, t_shift)
            t_remaining = time_h - t_shift
            if t_remaining > MIN_T_REM_H:
                rows_out.append({
                    "phi": phi, "energy": energy_ev, "time": t_remaining,
                    "init_N226": n_shifted[0], "init_N225": n_shifted[1],
                    "init_NAc": n_shifted[2],
                    "N_Ra226": n_final[0], "N_Ra225": n_final[1], "N_Ac225": n_final[2],
                })
    return pd.DataFrame(rows_out)


def augment_empty_tank_rows(rng: np.random.Generator, n_extra: int = 300) -> pd.DataFrame:
    """Empty initial inventories at various flux/time/energy -> all outputs must be 0."""
    rows_out: list[dict[str, float]] = []
    for _ in range(n_extra):
        phi = float(10.0 ** rng.uniform(13.0, 15.0))
        energy_ev = float(rng.uniform(0.015, 0.08))
        time_h = float(rng.uniform(10.0, 500.0))
        rows_out.append({
            "phi": phi, "energy": energy_ev, "time": time_h,
            "init_N226": 0.0, "init_N225": 0.0, "init_NAc": 0.0,
            "N_Ra226": 0.0, "N_Ra225": 0.0, "N_Ac225": 0.0,
        })
    return pd.DataFrame(rows_out)


def prepare_training_tensors(
    df: pd.DataFrame, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    time_h = torch.tensor(df["time"].values, dtype=torch.float32, device=device)
    phi = torch.tensor(df["phi"].values, dtype=torch.float32, device=device)
    energy_raw = torch.tensor(df["energy"].values, dtype=torch.float32, device=device)
    energy_nn = neutron_energy_ev_to_feature_torch(energy_raw)
    init226 = torch.tensor(df["init_N226"].values, dtype=torch.float32, device=device)
    init225 = torch.tensor(df["init_N225"].values, dtype=torch.float32, device=device)
    init_ac = torch.tensor(df["init_NAc"].values, dtype=torch.float32, device=device)

    t_input = (time_h / TIME_SCALE_H).clone()
    phi_nn = (phi / PHI_SCALE).clone()
    inputs = torch.cat([
        t_input.unsqueeze(1),
        phi_nn.unsqueeze(1),
        energy_nn.unsqueeze(1).detach(),
        (init226 / N226_SCALE).unsqueeze(1).detach(),
        (init225 / N225_SCALE).unsqueeze(1).detach(),
        (init_ac / NAC_SCALE).unsqueeze(1).detach(),
    ], dim=1).detach()
    inputs.requires_grad_(True)
    targets = torch.tensor(
        df[["N_Ra226", "N_Ra225", "N_Ac225"]].values,
        dtype=torch.float32, device=device,
    )
    return inputs, targets


def build_zero_flux_collocation(
    n: int, rng: np.random.Generator, device: torch.device, *, t_max_h: float,
) -> torch.Tensor:
    time_h = torch.tensor(rng.uniform(1e-4, float(t_max_h), n), dtype=torch.float32, device=device)
    phi_nn = torch.zeros((n, 1), dtype=torch.float32, device=device)
    energy_raw = torch.tensor(rng.uniform(0.015, 0.08, n), dtype=torch.float32, device=device)
    energy_nn = neutron_energy_ev_to_feature_torch(energy_raw)

    lo = 1.0
    hi_226 = N226_SCALE * 1.1
    hi_225 = N225_SCALE * 10.0
    hi_ac = NAC_SCALE * 10.0
    u = rng.uniform(0.0, 1.0, size=(n, 3)).astype(np.float64)
    init226_raw = np.exp(np.log(lo) + u[:, 0] * (np.log(hi_226) - np.log(lo))).astype(np.float32)
    init225_raw = np.exp(np.log(lo) + u[:, 1] * (np.log(hi_225) - np.log(lo))).astype(np.float32)
    initac_raw = np.exp(np.log(lo) + u[:, 2] * (np.log(hi_ac) - np.log(lo))).astype(np.float32)
    init226 = torch.tensor(init226_raw, dtype=torch.float32, device=device)
    init225 = torch.tensor(init225_raw, dtype=torch.float32, device=device)
    init_ac = torch.tensor(initac_raw, dtype=torch.float32, device=device)

    t_input = (time_h / TIME_SCALE_H).clone()
    colloc = torch.cat([
        t_input.unsqueeze(1),
        phi_nn,
        energy_nn.unsqueeze(1),
        (init226 / N226_SCALE).unsqueeze(1).detach(),
        (init225 / N225_SCALE).unsqueeze(1).detach(),
        (init_ac / NAC_SCALE).unsqueeze(1).detach(),
    ], dim=1)

    # 30% empty-tank collocation (some with high flux): no feedstock -> no production
    n_empty = int(max(1, round(float(n) * EMPTY_FEED_FRACTION)))
    idx = torch.randperm(n, device=device)[:n_empty]
    colloc[idx, 1] = float(EMPTY_FEED_HIGH_FLUX / PHI_SCALE)
    colloc[idx, 3:6] = 0.0

    return colloc.detach().requires_grad_(True)


def plot_loss_history(epochs, data_loss, physics_loss, path):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(epochs, np.clip(data_loss, 1e-30, None), label="Data MSE", color="C0")
    ax.semilogy(epochs, np.clip(physics_loss, 1e-30, None), label="Physics MSE", color="C1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("PINN training: data loss vs physics loss (12k epochs)")
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ac225_pred_vs_true(n_ac_true_atoms, n_ac_pred_atoms, path):
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    mask = (n_ac_true_atoms > 0) & (n_ac_pred_atoms > 0)
    true_m = n_ac_true_atoms[mask]
    pred_m = n_ac_pred_atoms[mask]
    if len(true_m) == 0:
        plt.close(fig)
        return
    lo = max(float(np.nanmin(np.minimum(true_m, pred_m))), 1e-300) * 0.5
    hi = float(np.nanmax(np.maximum(true_m, pred_m))) * 2.0
    log_err = np.abs(np.log10(pred_m + 1e-300) - np.log10(true_m + 1e-300))
    sc = ax.scatter(
        true_m, pred_m, c=log_err, cmap="cool", s=28, alpha=0.75,
        edgecolors="white", linewidths=0.3, vmin=0, vmax=max(1.0, float(np.percentile(log_err, 95))),
        zorder=3,
    )
    ax.plot([lo, hi], [lo, hi], color="#10b981", ls="--", lw=2, label="Perfect agreement", zorder=2)
    ax.fill_between(
        [lo, hi], [lo * 0.5, hi * 0.5], [lo * 2, hi * 2],
        color="#10b981", alpha=0.07, label="2x envelope", zorder=1,
    )
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(r"ODE Simulated $^{225}$Ac (atoms)", fontsize=12, color="white")
    ax.set_ylabel(r"PINN Predicted $^{225}$Ac (atoms)", fontsize=12, color="white")
    ax.set_title(r"Ac-225 Parity: PINN vs ODE", fontsize=14, fontweight="bold", color="white", pad=12)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(colors="white", which="both")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(True, which="major", ls="-", alpha=0.15, color="white")
    ax.grid(True, which="minor", ls=":", alpha=0.08, color="white")
    cb = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label("Log10 Absolute Error", fontsize=10, color="white")
    cb.ax.tick_params(colors="white")
    ax.legend(fontsize=10, loc="upper left", facecolor="#1e293b", edgecolor="#334155", labelcolor="white")
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="#0f172a", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    else:
        n_thr = max(1, int(os.environ.get("PINN_CPU_THREADS", os.cpu_count() or 1)))
        interop = os.environ.get("PINN_CPU_INTEROP", "").strip()
        n_interop = max(1, int(interop)) if interop.isdigit() else max(1, min(4, n_thr // 2))
        torch.set_num_threads(n_thr)
        try:
            torch.set_num_interop_threads(n_interop)
        except RuntimeError:
            pass
        print(f"CPU: torch threads={n_thr}, interop={n_interop}")

    df_raw = pd.read_csv(
        DATA_PATH,
        usecols=["phi", "energy", "time", "N_Ra226", "N_Ra225", "N_Ac225"],
        dtype={c: "float64" for c in ["phi", "energy", "time", "N_Ra226", "N_Ra225", "N_Ac225"]},
        engine="python",
    )
    df_raw["energy"] = df_raw["energy"].replace([np.inf, -np.inf], np.nan)
    df_raw["energy"] = df_raw["energy"].fillna(THERMAL_REFERENCE_EV)
    df_raw["energy"] = df_raw["energy"].clip(lower=1e-6, upper=1e8)
    if AUGMENT_BASE_ROW_LIMIT > 0:
        df_raw = df_raw.iloc[:AUGMENT_BASE_ROW_LIMIT].copy()
        print(f"AUGMENT_BASE_ROW_LIMIT={AUGMENT_BASE_ROW_LIMIT}: using first {len(df_raw)} CSV rows.")

    rng = np.random.default_rng(42)

    if USE_TIME_SHIFT_AUGMENT:
        n_cpu = os.cpu_count() or 1
        prep_workers = ODE_PREP_MAX_WORKERS if ODE_PREP_MAX_WORKERS > 0 else n_cpu
        prep_workers = max(1, min(prep_workers, n_cpu))
        if sys.platform == "win32":
            ode_par = os.environ.get("PINN_ODE_PARALLEL", "").strip().lower()
            if ode_par in ("0", "false", "no"):
                prep_workers = 1
            elif ode_par not in ("1", "true", "yes"):
                prep_workers = max(1, min(n_cpu - 1, 8)) if n_cpu >= 4 else 1
        print(f"ODE trajectory cache: {prep_workers} worker(s), {n_cpu} logical CPU(s)...")
        try:
            ref_cache = build_reference_traj_cache_parallel(df_raw, max_workers=prep_workers)
        except (BrokenProcessPool, OSError) as exc:
            print(f"Parallel ODE prep failed ({exc!r}); retrying single-process.")
            ref_cache = build_reference_traj_cache_parallel(df_raw, max_workers=1)
        print(f"  Distinct (phi, E, T) paths: {len(ref_cache)}")
        aug_n = AUGMENT_PER_ROW
        _ap = os.environ.get("PINN_AUGMENT_PER_ROW", "").strip()
        if _ap.isdigit():
            aug_n = max(0, int(_ap))
        train_dat = augment_rows_time_shift(
            df_raw, rng, augment_per_row=aug_n,
            include_unshifted_base=True, initial_cache=ref_cache,
        )
        print(f"Time-shift: {len(df_raw)} base -> {len(train_dat)} samples ({aug_n} shifts/row + base)")
    else:
        train_dat = pd.DataFrame({
            "phi": df_raw["phi"], "energy": df_raw["energy"], "time": df_raw["time"],
            "init_N226": TRAIN_INIT_RA226, "init_N225": TRAIN_INIT_RA225,
            "init_NAc": TRAIN_INIT_AC225,
            "N_Ra226": df_raw["N_Ra226"], "N_Ra225": df_raw["N_Ra225"],
            "N_Ac225": df_raw["N_Ac225"],
        })

    # Diverse ICs for pure-decay / mixed / Ac-dominant scenarios
    if not SINGLE_SUPPLY_MODE:
        print("Generating inverted-IC training data...")
        inverted_rows = augment_inverted_ic_scenarios(rng, n_extra=INVERTED_IC_N_EXTRA)
        train_dat = pd.concat([train_dat, inverted_rows], ignore_index=True)
        print(f"  Added {len(inverted_rows)} inverted-IC rows. Total: {len(train_dat)}")
        print("Generating diverse-IC training data...")
        diverse_rows = augment_diverse_ic_scenarios(rng, n_extra=DIVERSE_IC_N_EXTRA)
        train_dat = pd.concat([train_dat, diverse_rows], ignore_index=True)
        print(f"  Added {len(diverse_rows)} diverse-IC rows. Total: {len(train_dat)}")
    else:
        print("SINGLE_SUPPLY_MODE: skipping inverted/diverse synthetic rows.")

    # Empty-tank training rows (output must be all zeros)
    print("Generating empty-tank training rows...")
    empty_rows = augment_empty_tank_rows(rng, n_extra=300)
    train_dat = pd.concat([train_dat, empty_rows], ignore_index=True)
    print(f"  Added {len(empty_rows)} empty-tank rows. Total: {len(train_dat)}")

    inputs, targets = prepare_training_tensors(train_dat, device)
    t_max_h = max(float(train_dat["time"].max()), 1.0)

    train_epochs = EPOCHS
    train_pre = PHYS_PRETRAIN_EPOCHS
    if os.environ.get("PINN_QUICK_TRAIN", "").lower() in ("1", "true", "yes"):
        train_epochs = min(train_epochs, 400)
        train_pre = min(train_pre, 120)
        print("PINN_QUICK_TRAIN=1: short run for debugging.")
    if os.environ.get("PINN_SMOKE", "").lower() in ("1", "true", "yes"):
        train_epochs = min(train_epochs, 3)
        train_pre = min(train_pre, 3)
        print("PINN_SMOKE=1: tiny run for CI only.")

    total_epochs = train_pre + train_epochs
    print(f"\nMax-Fix v2: {train_pre} pretrain + {train_epochs} joint = {total_epochs} total epochs")
    print(f"  Ra-225 physics weight: {RA225_PHYSICS_WEIGHT}x")
    print(f"  Non-negativity weight: {NON_NEG_WEIGHT}")
    print(f"  Secular eq weight:     {SECULAR_EQ_WEIGHT}")
    print(f"  Empty-tank colloc:     {EMPTY_FEED_FRACTION*100:.0f}%")
    print(f"  Input norm: t/={TIME_SCALE_H}, phi/={PHI_SCALE:.0e}")
    print()

    model = IsotopePINN().to(device)
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_PLATEAU_FACTOR, patience=LR_PLATEAU_PATIENCE,
    )

    epoch_list: list[int] = []
    hist_data: list[float] = []
    hist_phys: list[float] = []

    rng_colloc = np.random.default_rng(2026)

    # ---- Physics-only pretrain ----
    print(
        f"Physics pretrain: {train_pre} epochs, {COLLOCATION_POINTS} collocation pts/epoch, "
        f"lr={PHYS_PRETRAIN_LR:.0e}"
    )
    saved_pretrain_lr = optimizer.param_groups[0]["lr"]
    optimizer.param_groups[0]["lr"] = PHYS_PRETRAIN_LR
    phys_skip_streak = 0
    for epoch in range(1, train_pre + 1):
        colloc = build_zero_flux_collocation(COLLOCATION_POINTS, rng_colloc, device, t_max_h=t_max_h)
        optimizer.zero_grad(set_to_none=True)
        pred = model.forward_raw(colloc)
        if not torch.isfinite(pred).all():
            phys_skip_streak += 1
            if phys_skip_streak <= 2 or epoch % LOG_EVERY == 0:
                print(f"  !! [phys] epoch {epoch}: non-finite pred (streak {phys_skip_streak})")
            continue
        loss, info = compute_physics_loss(
            model, colloc, pred, targets=None,
            physics_weight=PHYS_PRETRAIN_PHYS_WEIGHT, data_weight=0.0,
            mass_weight=PHYS_PRETRAIN_MASS_WEIGHT,
            fuel_anchor_weight=PHYS_PRETRAIN_FUEL_ANCHOR_WEIGHT,
            non_neg_weight=NON_NEG_WEIGHT, secular_eq_weight=SECULAR_EQ_WEIGHT,
            ra225_physics_weight=RA225_PHYSICS_WEIGHT,
            n226_scale=N226_SCALE, n225_scale=N225_SCALE, nac_scale=NAC_SCALE,
            phi_scale=PHI_SCALE, d_t_input_d_t_hours=D_T_INPUT_D_T_HOURS,
            use_one_over_v_energy=True,
        )
        if not torch.isfinite(loss):
            phys_skip_streak += 1
            if phys_skip_streak <= 2 or epoch % LOG_EVERY == 0:
                print(f"  !! [phys] epoch {epoch}: non-finite loss (streak {phys_skip_streak})")
            continue
        phys_skip_streak = 0
        loss.backward()
        clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            optimizer.zero_grad(set_to_none=True)
            continue
        optimizer.step()
        scheduler.step(float(loss.detach().cpu()))

        epoch_list.append(epoch)
        hist_data.append(float(info["data_mse"].detach().cpu()))
        hist_phys.append(float(info["physics_mse"].detach().cpu()))

        if epoch == 1 or epoch % LOG_EVERY == 0 or epoch == train_pre:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"epoch {epoch:6d} [phys] | lr {lr:.2e} | total {info['total_loss'].item():.4e} "
                f"| phys {info['physics_mse'].item():.4e} | mass {info['mass_cons_loss'].item():.4e} "
                f"| neg {info['non_neg_loss'].item():.4e} | sec_eq {info['secular_eq_loss'].item():.4e}"
            )

    optimizer.param_groups[0]["lr"] = saved_pretrain_lr

    # ---- Joint training ----
    best_loss = float("inf")
    nan_streak = 0
    joint_epoch = 0
    torch.save(model.state_dict(), BEST_CKPT_PATH)

    n_train = int(inputs.size(0))
    joint_chunk_sz = JOINT_CHUNK_SIZE
    if (
        device.type == "cpu"
        and os.environ.get("PINN_FAST_CPU", "").lower() in ("1", "true", "yes")
        and not os.environ.get("PINN_JOINT_CHUNK", "").strip()
        and joint_chunk_sz > 0
    ):
        joint_chunk_sz = max(joint_chunk_sz, 2048)
        print(f"PINN_FAST_CPU: joint chunk >= 2048")
    chunk_sz = n_train if joint_chunk_sz <= 0 else min(joint_chunk_sz, n_train)
    print(f"Joint training: {train_epochs} epochs, chunks {chunk_sz}/{n_train}")

    for j in range(1, train_epochs + 1):
        epoch = train_pre + j
        joint_epoch += 1
        model.train()

        ramp = min(1.0, joint_epoch / float(WARMUP_EPOCHS))
        cur_data_w = ramp * DATA_WEIGHT
        cur_fuel_w = ramp * FUEL_ANCHOR_WEIGHT

        optimizer.zero_grad(set_to_none=True)

        total_loss_val = 0.0
        wsum_phys = wsum_data = wsum_mass = wsum_fuel = wsum_zero = 0.0
        wsum_neg = wsum_sec = 0.0

        for start in range(0, n_train, chunk_sz):
            end = min(start + chunk_sz, n_train)
            n_b = end - start
            w = float(n_b) / float(n_train)
            inp = inputs[start:end].detach().clone()
            inp.requires_grad_(True)
            tgt = targets[start:end]

            pred_raw = model.forward_raw(inp)
            pred_capped = model(inp)

            loss_b, info_b = compute_physics_loss(
                model, inp, pred_raw, targets=tgt,
                physics_weight=PHYSICS_WEIGHT, data_weight=cur_data_w,
                mass_weight=MASS_WEIGHT, fuel_anchor_weight=cur_fuel_w,
                non_neg_weight=NON_NEG_WEIGHT, secular_eq_weight=SECULAR_EQ_WEIGHT,
                ra225_physics_weight=RA225_PHYSICS_WEIGHT,
                pred_for_data=pred_capped, data_species_weights=DATA_SPECIES_WEIGHTS,
                n226_scale=N226_SCALE, n225_scale=N225_SCALE, nac_scale=NAC_SCALE,
                phi_scale=PHI_SCALE, d_t_input_d_t_hours=D_T_INPUT_D_T_HOURS,
                use_one_over_v_energy=True,
            )
            if not torch.isfinite(loss_b):
                total_loss_val = float("nan")
                optimizer.zero_grad(set_to_none=True)
                break
            (loss_b * w).backward()
            total_loss_val += float(loss_b.detach().cpu()) * w
            wsum_phys += w * float(info_b["physics_mse"].detach().cpu())
            wsum_data += w * float(info_b["data_mse"].detach().cpu())
            wsum_mass += w * float(info_b["mass_cons_loss"].detach().cpu())
            wsum_fuel += w * float(info_b["fuel_anchor_loss"].detach().cpu())
            wsum_zero += w * float(info_b["zero_injection_loss"].detach().cpu())
            wsum_neg  += w * float(info_b["non_neg_loss"].detach().cpu())
            wsum_sec  += w * float(info_b["secular_eq_loss"].detach().cpu())

        info = {
            "data_mse": torch.tensor(wsum_data, device=device),
            "physics_mse": torch.tensor(wsum_phys, device=device),
            "mass_cons_loss": torch.tensor(wsum_mass, device=device),
            "fuel_anchor_loss": torch.tensor(wsum_fuel, device=device),
            "zero_injection_loss": torch.tensor(wsum_zero, device=device),
            "non_neg_loss": torch.tensor(wsum_neg, device=device),
            "secular_eq_loss": torch.tensor(wsum_sec, device=device),
        }

        if not np.isfinite(total_loss_val):
            nan_streak += 1
            if nan_streak >= NAN_PATIENCE:
                print(f"  !! {NAN_PATIENCE} NaN streak -- restore best, halve LR")
                model.load_state_dict(torch.load(BEST_CKPT_PATH, map_location=device, weights_only=True))
                optimizer.state.clear()
                for pg in optimizer.param_groups:
                    pg["lr"] = max(pg["lr"] * 0.5, 1.0e-7)
                nan_streak = 0
            continue

        nan_streak = 0
        clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            nan_streak += 1
            optimizer.zero_grad(set_to_none=True)
            if nan_streak >= NAN_PATIENCE:
                print(f"  !! {NAN_PATIENCE} non-finite grads -- restore best, halve LR")
                model.load_state_dict(torch.load(BEST_CKPT_PATH, map_location=device, weights_only=True))
                optimizer.state.clear()
                for pg in optimizer.param_groups:
                    pg["lr"] = max(pg["lr"] * 0.5, 1.0e-7)
                nan_streak = 0
            continue

        nan_streak = 0
        optimizer.step()

        loss_val = total_loss_val
        scheduler.step(loss_val)

        epoch_list.append(epoch)
        hist_data.append(float(info["data_mse"].detach().cpu()))
        hist_phys.append(float(info["physics_mse"].detach().cpu()))

        if loss_val < best_loss:
            best_loss = loss_val
            torch.save(model.state_dict(), BEST_CKPT_PATH)

        if j == 1 or j % LOG_EVERY == 0 or j == train_epochs:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"epoch {epoch:6d} | lr {lr:.2e} | total {loss_val:.4e} "
                f"| data {wsum_data:.4e} (w={cur_data_w:.1f}) "
                f"| phys {wsum_phys:.4e} | mass {wsum_mass:.4e} "
                f"| neg {wsum_neg:.4e} | sec_eq {wsum_sec:.4e}"
            )

    model.load_state_dict(torch.load(BEST_CKPT_PATH, map_location=device, weights_only=True))
    print(f"Loaded best checkpoint (loss={best_loss:.6e})")

    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"Saved state dict to {WEIGHTS_PATH}")

    plot_loss_history(epoch_list, hist_data, hist_phys, LOSS_PLOT_PATH)
    print(f"Saved loss plot to {LOSS_PLOT_PATH}")

    model.eval()
    with torch.no_grad():
        pred_final = model(inputs.detach())
    n_ac_true = targets[:, 2].detach().cpu().numpy()
    n_ac_pred = (pred_final[:, 2] * NAC_SCALE).detach().cpu().numpy()
    plot_ac225_pred_vs_true(n_ac_true, n_ac_pred, AC225_SCATTER_PATH)
    print(f"Saved Ac-225 scatter to {AC225_SCATTER_PATH}")


if __name__ == "__main__":
    main()
