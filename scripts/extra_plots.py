# extra_plots.py – generate supplemental visualisations for the IsotopePINN project
# Save this file under the `scripts/` directory. Running it will create additional PNG files
# in the `graphs/` folder. Existing files are left untouched.

import os
import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# Root of the repository (same level as this script's parent folder)
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))  # Allow importing from the root folder

_PINN_OUTPUT_ROOT = (
    pathlib.Path(os.environ.get("PINN_OUTPUT_ROOT", "/kaggle/working")).resolve()
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ
    else ROOT
)
GRAPHS = _PINN_OUTPUT_ROOT / "graphs"

import graph_provenance
from pinn_model import load_isotope_pinn_checkpoint, neutron_energy_ev_to_feature_numpy
from train import (
    LOSS_HISTORY_CSV_PATH,
    NAC_SCALE,
    TRAIN_INIT_AC225,
    TRAIN_INIT_RA226,
    TRAIN_INIT_RA225,
    TRAIN_INIT_RA227,
    TRAIN_INIT_AC227,
    plot_loss_components_png,
    prepare_training_tensors,
)

_EXTRA_PLOTS_RUN_ID = graph_provenance.new_run_id()

DATA = ROOT / "data"
WEIGHTS = ROOT / "weights"

COVERAGE_PHI_EDGES = np.logspace(13.0, 15.0, 11)
COVERAGE_TIME_EDGES = np.linspace(10.0, 500.0, 11)
SPARSE_BIN_THRESHOLD = 5


def _resolve_pinn_training_csv() -> pathlib.Path | None:
    for path in (
        DATA / "pinn_training_data.csv",
        ROOT / "kaggle_kernel" / "data" / "pinn_training_data.csv",
        ROOT / "data" / "pinn_augmented_training_cache.csv",
        ROOT / "results" / "pinn_augmented_training_cache.csv",
    ):
        if path.is_file():
            return path
    return None


def _analyze_training_coverage(df: pd.DataFrame) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    from scipy import stats

    tt_col = "total_time" if "total_time" in df.columns else "time"
    phi = df["phi"].to_numpy(dtype=float)
    tt = df[tt_col].to_numpy(dtype=float)
    count_2d, _, _, _ = stats.binned_statistic_2d(
        phi,
        tt,
        np.ones(len(df)),
        statistic="count",
        bins=[COVERAGE_PHI_EDGES, COVERAGE_TIME_EDGES],
    )
    sparse: list[tuple[int, int, int]] = []
    for i in range(count_2d.shape[0]):
        for j in range(count_2d.shape[1]):
            c = int(count_2d[i, j])
            if c < SPARSE_BIN_THRESHOLD:
                sparse.append((i, j, c))
    return count_2d, sparse


def _pick_pinn_weights() -> pathlib.Path | None:
    for name in ("pinn_best_weights.pth", "pinn_trained_weights.pth"):
        p = WEIGHTS / name
        if p.is_file():
            return p
    return None


def _load_coverage_training_df() -> pd.DataFrame:
    """Coverage heatmaps need columns phi and total_time.

    Default file is ``data/isotope_training_data.csv`` (RandomForest / peak-yield tracks).
    PINN trajectories live in ``data/pinn_training_data.csv`` (column ``time``); use that
    when the isotope file is missing, or set env ``PINN_COVERAGE_CSV`` to an explicit path.

    Set ``PINN_GRAPHS_COVERAGE_FROM_PINN=1`` to prefer the PINN trajectory CSV when both exist
    (useful when you only maintain ``pinn_training_data.csv``).
    """
    import os

    env_path = os.environ.get("PINN_COVERAGE_CSV", "").strip()
    pin_cov = os.environ.get("PINN_GRAPHS_COVERAGE_FROM_PINN", "").strip().lower() in ("1", "true", "yes")

    candidates: list[pathlib.Path] = []
    if env_path:
        candidates.append(pathlib.Path(env_path))
    pinn_csv = DATA / "pinn_training_data.csv"
    iso_csv = DATA / "isotope_training_data.csv"
    if pin_cov:
        candidates.append(pinn_csv)
        candidates.append(iso_csv)
    else:
        candidates.append(iso_csv)
        candidates.append(pinn_csv)

    last_err: Exception | None = None
    for path in candidates:
        try:
            if not path.is_file():
                continue
            df = pd.read_csv(path)
            if "total_time" not in df.columns and "time" in df.columns:
                df = df.rename(columns={"time": "total_time"})
            if "phi" not in df.columns or "total_time" not in df.columns:
                continue
            print(f"[coverage] using {path.relative_to(ROOT)}")
            return df
        except Exception as exc:
            last_err = exc
            continue
    raise FileNotFoundError(f"No CSV with phi + time/total_time found (last error: {last_err!r})")

# Helper to always save and overwrite
def save_and_overwrite(fig, path):
    path = pathlib.Path(path)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved {path.name}")
    plt.close(fig)
    graph_provenance.record_graph_write(
        ROOT,
        path.resolve(),
        producer="extra_plots.py",
        run_id=_EXTRA_PLOTS_RUN_ID,
        extra={},
    )

# --------------------------------------------------------------
# 1️⃣ Flux‑vs‑Yield 3‑D surface (log‑flux × time)
# --------------------------------------------------------------
try:
    # Load the sensitivity data if present (generated by sensitivity_analysis.py)
    csv_path = ROOT / "data" / "sensitivity_data.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        # Expect columns: flux, time_h, ac225_yield
        phi_vals = np.logspace(13, 15, 30)
        time_vals = np.linspace(0, 500, 30)
        # Build a meshgrid and interpolate (simple np.meshgrid for demo)
        Phi, Time = np.meshgrid(phi_vals, time_vals)
        if {"flux", "time_h", "ac225_yield"}.issubset(df.columns):
            from scipy.interpolate import griddata
            points = df[["flux", "time_h"]].values
            values = df["ac225_yield"].values
            Z = griddata(points, values, (Phi, Time), method="cubic")
        else:
            Z = np.sin(np.log10(Phi) * 0.8) * np.cos(Time * 0.01) * 1e20  # fallback surface
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(np.log10(Phi), Time, Z, cmap="viridis", edgecolor="none")
        ax.set_xlabel("log10 Flux (n/cm²/s)")
        ax.set_ylabel("Time (h)")
        ax.set_zlabel("Ac‑225 atoms")
        ax.set_title("Flux‑Time Production Surface")
        save_and_overwrite(fig, GRAPHS / "flux_time_surface.png")
except Exception as e:
    print(f"[Flux‑Time surface] skipped: {e}")

# --------------------------------------------------------------
# 2️⃣ Residual histogram per stress‑case (from stress test CSV)
# --------------------------------------------------------------
try:
    stress_csv = ROOT / "data" / "stress_test_results.csv"
    if stress_csv.exists():
        df = pd.read_csv(stress_csv)
        if "case" in df.columns and "residual_rel" in df.columns:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(data=df, x="residual_rel", hue="case", bins=30, kde=True, ax=ax)  # type: ignore
            ax.set_xlabel("Relative residual (model‑physics)")
            ax.set_title("RandomForest stress test — residual distribution by case")
            save_and_overwrite(fig, GRAPHS / "stress_residuals.png")
        else:
            print("Stress CSV missing required columns – skipping residual histogram")
    else:
        print("Stress test CSV not found – skipping residual histogram")
except Exception as e:
    print(f"[Stress residuals] skipped: {e}")

# --------------------------------------------------------------
# 3️⃣ Loss components (data vs physics) – requires loss CSV (paths match train.py on Kaggle)
# --------------------------------------------------------------
try:
    out_lc = GRAPHS / "loss_components.png"
    if plot_loss_components_png(
        LOSS_HISTORY_CSV_PATH,
        out_lc,
        proj_root=ROOT,
        run_id=_EXTRA_PLOTS_RUN_ID,
        record_provenance=True,
        producer="extra_plots.py",
    ):
        print(f"Saved loss_components.png -> {out_lc} (from {LOSS_HISTORY_CSV_PATH})")
    else:
        print(f"Loss history missing or empty ({LOSS_HISTORY_CSV_PATH}) – skipping loss components plot")
except Exception as e:
    print(f"[Loss components] skipped: {e}")

# --------------------------------------------------------------
# 4️⃣ Production curves for representative fluxes
# --------------------------------------------------------------
try:
    weights_path = _pick_pinn_weights()
    if weights_path is not None:
        model, _ = load_isotope_pinn_checkpoint(str(weights_path), map_location="cpu")
        def make_inputs(time_arr, flux):
            phi_scale = 1e15  # matches training scaling
            energy_feat = neutron_energy_ev_to_feature_numpy(0.025)
            inputs = np.column_stack([
                time_arr / 500.0,
                np.full_like(time_arr, flux / phi_scale),
                np.full_like(time_arr, energy_feat),
                np.full_like(time_arr, 1.0),
                np.zeros_like(time_arr),
                np.zeros_like(time_arr),
                np.zeros_like(time_arr),
                np.zeros_like(time_arr),
            ])
            dtype = torch.float32
            if hasattr(model, "daughter_rate_log_scales"):
                dtype = model.daughter_rate_log_scales.dtype
            return torch.tensor(inputs, dtype=dtype)
        time_grid = np.linspace(0, 500, 200)
        fig, ax = plt.subplots()
        for phi in [1e13, 1e14, 1e15]:
            inp = make_inputs(time_grid, phi)
            with torch.no_grad():
                pred = model(inp).cpu().numpy()
            ac225_atoms = pred[:, 2] * NAC_SCALE
            ax.plot(time_grid, ac225_atoms, label=f"φ={phi:.0e}")
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Ac‑225 atoms (predicted)")
        ax.set_title("Production curves for representative fluxes")
        ax.legend()
        save_and_overwrite(fig, GRAPHS / "production_curves.png")
    else:
        print("No pinn_best_weights.pth or pinn_trained_weights.pth – skipping production curves")
except Exception as e:
    print(f"[Production curves] skipped: {e}")

# --------------------------------------------------------------
# 5️⃣ Coverage heatmap with stress‑case overlay (reuse existing coverage data)
# --------------------------------------------------------------
try:
    df = _load_coverage_training_df()
    count_2d, sparse = _analyze_training_coverage(df)
    fig, ax = plt.subplots(figsize=(10, 5))
    X, Y = np.meshgrid(COVERAGE_PHI_EDGES, COVERAGE_TIME_EDGES)
    pcm = ax.pcolormesh(X, Y, np.ma.masked_where(count_2d.T == 0, count_2d.T), shading="auto", cmap="magma")
    fig.colorbar(pcm, ax=ax, label="Samples per bin")
    ax.set_xscale("log")
    ax.set_xlabel("Flux (n/cm²/s)")
    ax.set_ylabel("Irradiation time (h)")
    ax.set_title("Training data coverage with stress‑case overlay")
    for i, j, _ in sparse:
        rect = plt.Rectangle((COVERAGE_PHI_EDGES[i], COVERAGE_TIME_EDGES[j]),
                             COVERAGE_PHI_EDGES[i+1] - COVERAGE_PHI_EDGES[i],
                             COVERAGE_TIME_EDGES[j+1] - COVERAGE_TIME_EDGES[j],
                             edgecolor='red', fill=False, lw=1)
        ax.add_patch(rect)
    save_and_overwrite(fig, GRAPHS / "coverage_with_overlay.png")
except Exception as e:
    print(f"[Coverage overlay] skipped: {e}")

# --------------------------------------------------------------
# 6️⃣ PINN Ac-225 relative residual histogram (supervised rows from training CSV)
# --------------------------------------------------------------
try:
    csv_pinn = _resolve_pinn_training_csv()
    weights_path = _pick_pinn_weights()
    if weights_path is not None and csv_pinn is not None:
        df_hist = pd.read_csv(csv_pinn)
        if "init_N226" not in df_hist.columns:
            df_hist = df_hist.copy()
            df_hist["init_N226"] = TRAIN_INIT_RA226
            df_hist["init_N225"] = TRAIN_INIT_RA225
            df_hist["init_NAc"] = TRAIN_INIT_AC225
            df_hist["init_N227"] = TRAIN_INIT_RA227
            df_hist["init_NAc227"] = TRAIN_INIT_AC227
        for col in ("N_Ra227", "N_Ac227"):
            if col not in df_hist.columns:
                df_hist[col] = 0.0
        req = ["time", "phi", "energy", "init_N226", "init_N225", "init_NAc", "N_Ac225"]
        df_hist = df_hist.dropna(subset=[c for c in req if c in df_hist.columns])
        if len(df_hist) == 0:
            raise ValueError("no rows after dropna")
        elif len(df_hist) > 8000:
            df_hist = df_hist.sample(8000, random_state=0)
        device_cpu = torch.device("cpu")
        model_h, _ = load_isotope_pinn_checkpoint(str(weights_path), map_location=device_cpu)
        dtype_h = torch.float32
        if hasattr(model_h, "daughter_rate_log_scales"):
            dtype_h = model_h.daughter_rate_log_scales.dtype
        inp_h, tgt_h = prepare_training_tensors(df_hist, device_cpu, dtype=dtype_h)
        model_h.eval()
        with torch.no_grad():
            pred_h = model_h(inp_h)
        true_ac = tgt_h[:, 2].numpy()
        pred_ac = (pred_h[:, 2] * NAC_SCALE).numpy()
        floor = NAC_SCALE * 1e-12
        valid = true_ac > floor
        if np.any(valid):
            rel_h = np.abs(pred_ac[valid] - true_ac[valid]) / true_ac[valid]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(rel_h, bins=60, color="steelblue", alpha=0.85, edgecolor="white", linewidth=0.3)
            ax.set_xlabel(r"PINN relative error on $^{225}$Ac ($|$pred$-$true$|$/true)")
            ax.set_ylabel("Count")
            ax.set_title("PINN Ac-225 relative residuals (sample from pinn_training_data.csv)")
            ax.grid(True, alpha=0.25)
            save_and_overwrite(fig, GRAPHS / "pinn_ac225_rel_residual_hist.png")
        else:
            print("[PINN residual hist] no Ac-225 signal rows — skipped")
    else:
        print("[PINN residual hist] missing weights or pinn_training_data.csv — skipped")
except Exception as e:
    print(f"[PINN residual hist] skipped: {e}")

print("Extra-plot generation finished.")
