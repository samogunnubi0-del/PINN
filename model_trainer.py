"""
Machine-learning pipeline for isotope_training_data.csv: EDA pairplot, Random Forest,
metrics (R^2, MAE), feature importance, training coverage diagnostics, and stress tests
against the physics simulator.
"""

from __future__ import annotations

import json
import pathlib

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from ra226_ac225_transmutation import IsotopeEnvironment, run_simulation_peak

DATA_CSV = pathlib.Path(__file__).resolve().parent / "isotope_training_data.csv"
MODEL_PATH = pathlib.Path(__file__).resolve().parent / "ac225_yield_model.joblib"
PAIRPLOT_PATH = pathlib.Path(__file__).resolve().parent / "training_pairplot.png"
FEATURE_IMPORTANCE_PATH = pathlib.Path(__file__).resolve().parent / "feature_importance.png"
COVERAGE_COUNTS_PATH = pathlib.Path(__file__).resolve().parent / "training_coverage_counts.png"
STRESS_TEST_CSV = pathlib.Path(__file__).resolve().parent / "stress_test_results.csv"
STRESS_METRICS_JSON = pathlib.Path(__file__).resolve().parent / "stress_metrics.json"

FEATURE_COLUMNS = ["phi", "total_time"]
TARGET_COLUMN = "max_ac225_yield"

# Stress cases that sit in or near low-flux / long-time extrapolation (see targeted augmentation).
STRESS_FAIL_ZONE_CASES = frozenset(
    {
        "below_flux_floor_1y",
        "long_beyond_training_t",
        "just_below_training_phi",
        "min_flux_max_scheduled_t",
        "low_flux_long",
    }
)

# Bins aligned with simulation ranges: log flux 1e13–1e5, time 10–500 h
COVERAGE_PHI_EDGES = np.logspace(13.0, 15.0, 11)
COVERAGE_TIME_EDGES = np.linspace(10.0, 500.0, 11)
SPARSE_BIN_THRESHOLD = 5


def plot_feature_importance(
    model: RandomForestRegressor,
    feature_names: list[str],
    path: pathlib.Path,
) -> None:
    """Bar chart of Gini-based mean decrease in impurity (RandomForest)."""
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    names = [feature_names[i] for i in order]
    vals = importances[order]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, vals, color="steelblue", edgecolor="white")
    ax.set_title(r"What the forest uses most for $^{225}$Ac peak-yield prediction")
    ax.set_ylabel("Importance (MDI)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def analyze_training_coverage(df: pd.DataFrame) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """
    2D bin counts over (phi, total_time). Returns count grid and list of sparse bins
    (i_phi, i_time, count) where count < SPARSE_BIN_THRESHOLD.
    """
    phi = df["phi"].to_numpy(dtype=float)
    tt = df["total_time"].to_numpy(dtype=float)
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


def plot_coverage_counts(count_2d: np.ndarray, path: pathlib.Path) -> None:
    """Heatmap of sample counts per (flux, time) bin — empty or dim cells are 'holes'."""
    Z = np.ma.masked_where(count_2d.T == 0, count_2d.T)
    fig, ax = plt.subplots(figsize=(10, 5))
    X, Y = np.meshgrid(COVERAGE_PHI_EDGES, COVERAGE_TIME_EDGES)
    pcm = ax.pcolormesh(X, Y, Z, shading="auto", cmap="magma")
    fig.colorbar(pcm, ax=ax, label="Number of training runs in bin")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\phi$ (n cm$^{-2}$ s$^{-1}$)")
    ax.set_ylabel("Irradiation time (h)")
    ax.set_title("Training data density (under-sampled = dark / holes)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def default_stress_cases() -> list[dict[str, float | str]]:
    """
    Ten deliberately awkward (phi, time) points: extrapolation, corners, and extremes.
    Ground truth comes from the ODE simulator, not the CSV.
    """
    return [
        {"name": "below_flux_floor_1y", "phi": 5e12, "total_time": 8760.0},
        {"name": "above_flux_ceiling_short", "phi": 5e15, "total_time": 10.0},
        {"name": "max_flux_min_scheduled_t", "phi": 1e15, "total_time": 10.0},
        {"name": "min_flux_max_scheduled_t", "phi": 1e13, "total_time": 500.0},
        {"name": "mid_mid", "phi": 1e14, "total_time": 200.0},
        {"name": "high_flux_very_short", "phi": 9.5e14, "total_time": 12.0},
        {"name": "low_flux_long", "phi": 1.2e13, "total_time": 480.0},
        {"name": "just_below_training_phi", "phi": 9e12, "total_time": 100.0},
        {"name": "long_beyond_training_t", "phi": 1e14, "total_time": 3000.0},
        {"name": "high_flux_mid_time", "phi": 8e14, "total_time": 250.0},
    ]


def run_stress_tests(
    model: RandomForestRegressor,
    sigma: float,
    cases: list[dict[str, float | str]] | None = None,
) -> pd.DataFrame:
    rows = []
    cases = cases or default_stress_cases()
    for c in cases:
        name = str(c["name"])
        phi = float(c["phi"])
        total_time = float(c["total_time"])
        env = IsotopeEnvironment(phi=phi, sigma_ra226=sigma)
        y_phys, _, _, _ = run_simulation_peak(env, total_time)
        y_ml = float(model.predict(np.array([[phi, total_time]], dtype=float))[0])
        resid = y_ml - y_phys
        rel = resid / y_phys if y_phys != 0 else float("nan")
        rows.append(
            {
                "case": name,
                "phi": phi,
                "total_time": total_time,
                "y_physics": y_phys,
                "y_model": y_ml,
                "residual_abs": resid,
                "residual_rel": rel,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(DATA_CSV)
    sigma_default = float(df["sigma"].iloc[0]) if "sigma" in df.columns else 1e-24

    plot_df = df[["phi", "total_time", "max_ac225_yield"]].copy()
    grid = sns.pairplot(
        plot_df,
        corner=True,
        plot_kws={"alpha": 0.35, "s": 12, "edgecolor": "none"},
        diag_kws={"alpha": 0.85},
    )
    grid.fig.suptitle(
        r"Feature / target relationships ($\phi$, irradiation time, $^{225}$Ac peak yield)",
        y=1.02,
    )
    grid.savefig(PAIRPLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close("all")

    count_2d, sparse = analyze_training_coverage(df)
    print(
        f"Coverage: {len(sparse)} / {count_2d.size} bins have fewer than "
        f"{SPARSE_BIN_THRESHOLD} samples (possible under-sampling / holes)."
    )
    if sparse:
        print("Examples (phi_bin, time_bin, count):", sparse[:8], "..." if len(sparse) > 8 else "")
    plot_coverage_counts(count_2d, COVERAGE_COUNTS_PATH)
    print(f"Saved coverage heatmap to {COVERAGE_COUNTS_PATH}")

    feature_cols = list(FEATURE_COLUMNS)
    if "sigma" in df.columns and df["sigma"].nunique() > 1:
        feature_cols.append("sigma")

    X = df[feature_cols].to_numpy(dtype=float)
    y = df[TARGET_COLUMN].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Test R^2 score: {r2:.6f}")
    print(f"Test MAE (atoms): {mae:.6e}")

    plot_feature_importance(model, feature_cols, FEATURE_IMPORTANCE_PATH)
    print(f"Saved feature importance plot to {FEATURE_IMPORTANCE_PATH}")

    stress_df = run_stress_tests(model, sigma=sigma_default)
    stress_df.to_csv(STRESS_TEST_CSV, index=False)
    print(f"Stress test results -> {STRESS_TEST_CSV}")
    print(stress_df.to_string(index=False))

    abs_rel = stress_df["residual_rel"].abs()
    fail_mask = stress_df["case"].isin(STRESS_FAIL_ZONE_CASES)
    mean_abs_rel_all = float(abs_rel.mean())
    mean_abs_rel_fail = float(abs_rel[fail_mask].mean())
    print(f"\nStress summary: mean |relative error| (all cases) = {mean_abs_rel_all:.4f}")
    print(
        f"Stress summary: mean |relative error| (low-flux / long-time subset) = "
        f"{mean_abs_rel_fail:.4f}"
    )
    with open(STRESS_METRICS_JSON, "w", encoding="utf-8") as jf:
        json.dump(
            {
                "mean_abs_rel_error_all": mean_abs_rel_all,
                "mean_abs_rel_error_fail_zone_cases": mean_abs_rel_fail,
                "r2_test": r2,
                "mae_test": mae,
                "n_training_rows": len(df),
            },
            jf,
            indent=2,
        )
    print(f"Wrote metrics summary to {STRESS_METRICS_JSON}")

    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_cols,
            "target_column": TARGET_COLUMN,
            "r2_test": r2,
            "mae_test": mae,
        },
        MODEL_PATH,
    )
    print(f"Saved bundle to {MODEL_PATH}")
    print(f"Saved pairplot to {PAIRPLOT_PATH}")


if __name__ == "__main__":
    main()
