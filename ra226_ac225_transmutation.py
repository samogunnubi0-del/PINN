"""
High-fidelity Bateman-style simulation for ^226Ra -> ^225Ra -> ^225Ac.

Three-species chain with an (n,2n)-style production step into ^225Ra and beta decay to ^225Ac:

    dN_Ra226/dt = -lambda_Ra226 * N_Ra226 - k * N_Ra226
    dN_Ra225/dt = +k * N_Ra226 - lambda_Ra225 * N_Ra225
    dN_Ac225/dt = +lambda_Ra225 * N_Ra225 - lambda_Ac225 * N_Ac225

where k = phi * sigma_eff is the effective first-order rate for ^226Ra -> ^225Ra.
sigma_eff = sigma_ra226 * S(E): either discrete ``energy_group`` weights, or continuous
eV energy with a schematic 1/v factor sqrt(E_ref / E) (lower cross section at higher E).
^225Ra half-life ~14.9 d introduces ingrowth lag before ^225Ac peaks.

Time integration uses scipy.integrate.odeint (legacy API; stable for smooth linear ODEs).
"""

from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass, field
from enum import Enum

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.integrate import odeint


# --- Physical constants (literature half-lives; adjust for your evaluation) ---
LN2 = np.log(2.0)
# ^226Ra half-life ~1600 y (NNDC / Table of Isotopes scale)
HALF_LIFE_RA226_H = 1600.0 * 365.25 * 24.0
# ^225Ra half-life ~14.9 d (beta^- to ^225Ac)
HALF_LIFE_RA225_H = 14.9 * 24.0
# ^225Ac half-life ~9.92 d
HALF_LIFE_AC225_H = 9.92 * 24.0

# Neutron energy (eV) for continuous-E PINN data and 1/v scaling reference
THERMAL_REFERENCE_EV = 0.025
PINN_ENERGY_MIN_EV = 0.025
PINN_ENERGY_MAX_EV = 1.0e6


def sigma_scale_one_over_v(energy_ev: float, reference_ev: float = THERMAL_REFERENCE_EV) -> float:
    """
    Schematic 1/v cross-section scaling: σ(E) ∝ 1/v ∝ E^{-1/2} for non-relativistic speeds.

    Returns dimensionless factor σ(E)/σ(reference), with ``reference`` typically thermal 0.025 eV.
    """
    E = max(float(energy_ev), 1e-30)
    return float(np.sqrt(float(reference_ev) / E))


class NeutronEnergyGroup(str, Enum):
    """Broad neutron energy groups for spectrum-averaged cross-section scaling."""

    THERMAL = "Thermal"
    EPITHERMAL = "Epithermal"
    FAST = "Fast"


# Toy resonance / 1/v-style weights: thermal neutrons see the largest effective
# production-channel cross section in this simplified PINN-oriented model.
# Fast = 1.0 keeps default env behavior aligned with legacy sigma_ra226-only runs.
ENERGY_SIGMA_SCALE: dict[str, float] = {
    NeutronEnergyGroup.THERMAL.value: 4.0,
    NeutronEnergyGroup.EPITHERMAL.value: 2.0,
    NeutronEnergyGroup.FAST.value: 1.0,
}


def sigma_scale_for_energy_group(energy_group: str) -> float:
    for g in NeutronEnergyGroup:
        if g.value.lower() == energy_group.strip().lower():
            return float(ENERGY_SIGMA_SCALE[g.value])
    raise ValueError(
        f"Unknown energy_group {energy_group!r}; use "
        f"{', '.join(repr(g.value) for g in NeutronEnergyGroup)}"
    )


@dataclass
class IsotopeEnvironment:
    """
    Holds adjustable nuclear/engineering parameters for the transmutation + decay model.

    Attributes
    ----------
    phi : float
        Scalar neutron flux [neutrons / (cm^2 * s)]. Must pair consistently with sigma.
    sigma_ra226 : float
        Reference microscopic cross-section [cm^2] for ^226Ra -> ^225Ra channel
        before energy scaling. 1 barn = 1e-24 cm^2.
    energy_group : str
        One of ``Thermal``, ``Epithermal``, ``Fast``. Scales sigma via
        ``ENERGY_SIGMA_SCALE`` when ``neutron_energy_ev`` is not set.
        Default ``Fast`` (scale 1.0) matches legacy runs that ignored energy.
    neutron_energy_ev : float, optional
        If set, overrides ``energy_group`` scaling with continuous **1/v** factor
        ``sqrt(E_ref / E)`` with ``E_ref`` = :data:`THERMAL_REFERENCE_EV` (0.025 eV).
    lambda_ra226_per_h, lambda_ra225_per_h, lambda_ac225_per_h : float
        Decay constants in 1/hour so time t can be tracked in hours directly.
    """

    phi: float = 1.0e14
    sigma_ra226: float = 1e-24  # 1 barn reference at E_ref; tune for your physics case
    energy_group: str = NeutronEnergyGroup.FAST.value
    neutron_energy_ev: float | None = None
    lambda_ra226_per_h: float = field(init=False)
    lambda_ra225_per_h: float = field(init=False)
    lambda_ac225_per_h: float = field(init=False)
    _sigma_energy_scale: float = field(init=False)

    def __post_init__(self) -> None:
        self.lambda_ra226_per_h = LN2 / HALF_LIFE_RA226_H
        self.lambda_ra225_per_h = LN2 / HALF_LIFE_RA225_H
        self.lambda_ac225_per_h = LN2 / HALF_LIFE_AC225_H
        if self.neutron_energy_ev is not None:
            self._sigma_energy_scale = sigma_scale_one_over_v(self.neutron_energy_ev)
        else:
            self._sigma_energy_scale = sigma_scale_for_energy_group(self.energy_group)

    def effective_sigma_ra226(self) -> float:
        """sigma_ra226 multiplied by the dimensionless energy factor (group or 1/v)."""
        return float(self.sigma_ra226) * self._sigma_energy_scale

    def transmutation_rate_constant_per_s(self) -> float:
        """
        Effective first-order rate k = phi * sigma_eff [1/s] for the production/removal term.

        sigma_eff includes the energy-group multiplier (simplified resonance / epithermal
        resonance tail vs fast continuum in this schematic model).
        """
        return float(self.phi) * self.effective_sigma_ra226()

    def transmutation_rate_constant_per_h(self) -> float:
        """Same as transmutation_rate_constant_per_s but converted to 1/hour for t in hours."""
        return self.transmutation_rate_constant_per_s() * 3600.0


def bateman_transmutation_rhs(
    N: np.ndarray,
    t: float,
    env: IsotopeEnvironment,
) -> np.ndarray:
    """
    Right-hand side for odeint: dN/dt as a function of state N and time t.

    Parameters
    ----------
    N : ndarray, shape (3,)
        N[0] = N_Ra226, N[1] = N_Ra225, N[2] = N_Ac225.
    t : float
        Current time [hours] (unused for this autonomous system).

    Returns
    -------
    dN_dt : ndarray, shape (3,)

    Physics
    -------
    Neutron channel feeds ^225Ra; ^225Ra beta-decays to ^225Ac (Bateman coupling);
    ^225Ac decays with its own lambda. This produces ingrowth lag versus a direct
    Ra226 -> Ac225 shortcut.
    """
    N_ra226, N_ra225, N_ac = float(N[0]), float(N[1]), float(N[2])
    k_h = env.transmutation_rate_constant_per_h()
    lam226 = env.lambda_ra226_per_h
    lam225 = env.lambda_ra225_per_h
    lam_ac = env.lambda_ac225_per_h

    dN_ra226 = -lam226 * N_ra226 - k_h * N_ra226
    dN_ra225 = k_h * N_ra226 - lam225 * N_ra225
    dN_ac = lam225 * N_ra225 - lam_ac * N_ac

    return np.array([dN_ra226, dN_ra225, dN_ac], dtype=float)


def run_simulation(
    env: IsotopeEnvironment,
    t_end_h: float = 200.0,
    n_points: int = 501,
    N_ra0: float = 6.022e23,
    N_ra225_0: float = 0.0,
    N_ac0: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Integrate the Bateman/transmutation system from t=0 to t=t_end_h.

    odeint solves the initial value problem dy/dt = f(y, t) from y0 at t[0] along the
    array of sample times t. Here y = [N_Ra226, N_Ra225, N_Ac225].

    The solver evaluates f repeatedly (adaptive internal substeps) and returns y at
    each requested time in t_hours—those are the 'output' times, not necessarily the
    internal step sizes used for error control.
    """
    t_hours = np.linspace(0.0, t_end_h, n_points)
    y0 = np.array([N_ra0, N_ra225_0, N_ac0], dtype=float)

    # args=(env,) passes our environment into bateman_transmutation_rhs as the third arg.
    Y = odeint(
        bateman_transmutation_rhs,
        y0,
        t_hours,
        args=(env,),
        rtol=1e-9,
        atol=1e-12,
    )
    return t_hours, Y


def ac225_peak_metrics(
    t_hours: np.ndarray,
    N_ac225: np.ndarray,
) -> tuple[float, float]:
    """
    From a time series of ^225Ac inventory, return (max yield, time of first maximum).

    If the curve is flat at the top, argmax returns the first index at that level.
    """
    i_peak = int(np.argmax(N_ac225))
    return float(N_ac225[i_peak]), float(t_hours[i_peak])


def run_simulation_peak(
    env: IsotopeEnvironment,
    t_end_h: float,
    n_points: int | None = None,
    N_ra0: float = 6.022e23,
    N_ac0: float = 0.0,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Run one irradiation scenario to t_end_h and report peak ^225Ac yield and time-to-peak.

    n_points scales with duration so short runs stay fast and long runs resolve the curve.
    """
    if n_points is None:
        # Long irradiations need enough samples so argmax(N_Ac) is not dominated by discretization.
        n_points = max(400, int(np.clip(t_end_h * 8, 400, 15000)))
    t_h, Y = run_simulation(
        env, t_end_h=t_end_h, n_points=n_points, N_ra0=N_ra0, N_ra225_0=0.0, N_ac0=N_ac0
    )
    N_ac = Y[:, 2]
    max_yield, t_peak = ac225_peak_metrics(t_h, N_ac)
    return max_yield, t_peak, t_h, N_ac


def generate_training_runs(
    n_runs: int = 1000,
    sigma_ra226: float = 1e-24,
    rng: np.random.Generator | None = None,
    N_ra0: float = 6.022e23,
) -> list[dict[str, float]]:
    """
    Batch Monte Carlo: random flux (log-uniform 1e13..1e15 n/cm^2/s) and duration
    (uniform 10..500 h). Each entry records inputs and peak ^225Ac metrics.
    """
    if rng is None:
        rng = np.random.default_rng()

    rows: list[dict[str, float]] = []
    for _ in range(n_runs):
        log_phi = rng.uniform(13.0, 15.0)
        phi = 10.0**log_phi
        total_time = float(rng.uniform(10.0, 500.0))

        env = IsotopeEnvironment(phi=phi, sigma_ra226=sigma_ra226)
        max_ac225, time_to_peak, _, _ = run_simulation_peak(env, total_time, N_ra0=N_ra0)

        rows.append(
            {
                "phi": phi,
                "sigma": sigma_ra226,
                "total_time": total_time,
                "max_ac225_yield": max_ac225,
                "time_to_peak": time_to_peak,
            }
        )
    return rows


# Targeted augmentation: regions where the random trainer is sparse (extrapolation stress tests).
TARGETED_PHI_LOG_MIN = 11.0  # 1e11 n/cm^2/s floor for low-flux tail
TARGETED_PHI_LT = 1e13  # strictly below nominal training minimum phi
TARGETED_TIME_MIN_H = 500.0  # strictly above nominal training maximum time (10–500 h)
TARGETED_TIME_MAX_H = 8760.0  # one year upper bound for long campaigns


def generate_targeted_data(
    n_runs: int = 200,
    sigma_ra226: float = 1e-24,
    rng: np.random.Generator | None = None,
    N_ra0: float = 6.022e23,
) -> list[dict[str, float]]:
    """
    Sample runs in 'fail zones' for the baseline random design: flux below 1e13 n/cm^2/s
    and irradiation time beyond 500 h. Intended to reduce ML extrapolation error.
    """
    if rng is None:
        rng = np.random.default_rng()

    log_phi_hi = np.log10(TARGETED_PHI_LT) - 1e-6
    rows: list[dict[str, float]] = []
    for _ in range(n_runs):
        log_phi = float(rng.uniform(TARGETED_PHI_LOG_MIN, log_phi_hi))
        phi = 10.0**log_phi
        total_time = float(rng.uniform(TARGETED_TIME_MIN_H, TARGETED_TIME_MAX_H))

        env = IsotopeEnvironment(phi=phi, sigma_ra226=sigma_ra226)
        max_ac225, time_to_peak, _, _ = run_simulation_peak(env, total_time, N_ra0=N_ra0)

        rows.append(
            {
                "phi": phi,
                "sigma": sigma_ra226,
                "total_time": total_time,
                "max_ac225_yield": max_ac225,
                "time_to_peak": time_to_peak,
            }
        )
    return rows


def append_training_csv(path: str, rows: list[dict[str, float]]) -> None:
    """Append rows to an existing training CSV (creates file with header if missing)."""
    fieldnames = ["phi", "sigma", "total_time", "max_ac225_yield", "time_to_peak"]
    exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerows(rows)


def augment_fail_zone_training(
    csv_path: str = "isotope_training_data.csv",
    n_targeted: int = 200,
    sigma_ra226: float = 1e-24,
    rng: np.random.Generator | None = None,
) -> int:
    """
    Append ``n_targeted`` simulations from low-flux / long-time domains to ``csv_path``.
    Returns number of rows written.
    """
    rng = rng or np.random.default_rng(2026)
    new_rows = generate_targeted_data(
        n_runs=n_targeted,
        sigma_ra226=sigma_ra226,
        rng=rng,
    )
    append_training_csv(csv_path, new_rows)
    return len(new_rows)


def save_training_csv(path: str, rows: list[dict[str, float]]) -> None:
    """Write batch summary for ML training: phi, sigma, total_time, max_ac225_yield, time_to_peak."""
    fieldnames = ["phi", "sigma", "total_time", "max_ac225_yield", "time_to_peak"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def generate_pinn_training_runs(
    n_runs: int = 1500,
    sigma_ra226: float = 1e-24,
    rng: np.random.Generator | None = None,
    N_ra0: float = 6.022e23,
) -> list[dict[str, float]]:
    """
    PINN-oriented dataset: random flux (log-uniform), neutron energy (eV, log-uniform
    between ``PINN_ENERGY_MIN_EV`` and ``PINN_ENERGY_MAX_EV``), and irradiation time.
    Cross section uses **1/v** scaling via :class:`IsotopeEnvironment`.
    Each row is the three-species Bateman state at end of irradiation.
    """
    if rng is None:
        rng = np.random.default_rng()

    log_e_min = np.log10(PINN_ENERGY_MIN_EV)
    log_e_max = np.log10(PINN_ENERGY_MAX_EV)
    rows: list[dict[str, float]] = []
    for _ in range(n_runs):
        log_phi = float(rng.uniform(13.0, 15.0))
        phi = 10.0**log_phi
        time_h = float(rng.uniform(10.0, 500.0))
        log_e = float(rng.uniform(log_e_min, log_e_max))
        energy_ev = float(10.0**log_e)

        env = IsotopeEnvironment(
            phi=phi,
            sigma_ra226=sigma_ra226,
            neutron_energy_ev=energy_ev,
        )
        n_points = max(400, int(np.clip(time_h * 8, 400, 15000)))
        _, Y = run_simulation(env, t_end_h=time_h, n_points=n_points, N_ra0=N_ra0)
        n226, n225, nac = (float(Y[-1, 0]), float(Y[-1, 1]), float(Y[-1, 2]))

        rows.append(
            {
                "phi": phi,
                "energy": energy_ev,
                "time": time_h,
                "N_Ra226": n226,
                "N_Ra225": n225,
                "N_Ac225": nac,
            }
        )
    return rows


def save_pinn_training_csv(path: str, rows: list[dict[str, float]]) -> None:
    """Write PINN table: phi, energy (eV), time (h), N_Ra226, N_Ra225, N_Ac225 (atoms)."""
    fieldnames = ["phi", "energy", "time", "N_Ra226", "N_Ra225", "N_Ac225"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def plot_yield_heatmap_flux_time(
    rows: list[dict[str, float]],
    outfile: str = "ac225_yield_heatmap.png",
    n_bins_flux: int = 24,
    n_bins_time: int = 24,
) -> None:
    """
    2D heatmap: neutron flux vs irradiation time, color = max ^225Ac yield in each bin.

    Random (phi, time) pairs are aggregated with scipy.stats.binned_statistic_2d using
    the maximum yield observed in each cell so sparse coverage still forms a grid.
    Flux bin edges are logarithmic (1e13..1e15) because phi spans two decades.
    """
    phi = np.array([r["phi"] for r in rows], dtype=float)
    tt = np.array([r["total_time"] for r in rows], dtype=float)
    yld = np.array([r["max_ac225_yield"] for r in rows], dtype=float)

    phi_edges = np.logspace(13.0, 15.0, n_bins_flux + 1)
    time_edges = np.linspace(10.0, 500.0, n_bins_time + 1)

    stat, _, _, _ = stats.binned_statistic_2d(
        phi,
        tt,
        yld,
        statistic="max",
        bins=[phi_edges, time_edges],
    )

    # Cells with no samples stay NaN; mask for cleaner color scale.
    # stat[i_phi, i_time] from SciPy; pcolormesh(meshgrid(phi,time)) expects C[time, phi] = stat.T.
    Z = np.ma.masked_invalid(stat.T)

    fig, ax = plt.subplots(figsize=(10, 6))
    X, Y = np.meshgrid(phi_edges, time_edges)
    pcm = ax.pcolormesh(X, Y, Z, shading="auto", cmap="viridis")
    cb = fig.colorbar(pcm, ax=ax, label=r"max $^{225}$Ac yield (atoms)")
    ax.set_xscale("log")
    ax.set_xlabel(r"Neutron flux $\phi$ (n cm$^{-2}$ s$^{-1}$)")
    ax.set_ylabel("Irradiation time (hours)")
    ax.set_title(r"$^{225}$Ac yield heatmap (max per bin over Monte Carlo runs)")
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


def save_results_csv(
    path: str,
    t_hours: np.ndarray,
    N_ra226: np.ndarray,
    N_ra225: np.ndarray,
    N_ac225: np.ndarray,
) -> None:
    """Write time series to CSV: time_h, N_Ra226, N_Ra225, N_Ac225."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_h", "N_Ra226", "N_Ra225", "N_Ac225"])
        for i in range(len(t_hours)):
            w.writerow([t_hours[i], N_ra226[i], N_ra225[i], N_ac225[i]])


def plot_ac225_growth(
    t_hours: np.ndarray,
    N_ac225: np.ndarray,
    outfile: str | None = "ac225_growth.png",
) -> None:
    """Matplotlib figure: ^225Ac inventory vs time."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t_hours, N_ac225, color="darkgreen", lw=2, label=r"$^{225}$Ac")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel(r"$N_{^{225}\mathrm{Ac}}$ (atoms)")
    ax.set_title(r"Growth of $^{225}$Ac from $^{226}$Ra transmutation (model)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if outfile:
        fig.savefig(outfile, dpi=150)
    plt.close(fig)


def main() -> None:
    rng = np.random.default_rng(42)
    rows = generate_training_runs(n_runs=1000, sigma_ra226=1e-24, rng=rng)
    save_training_csv("isotope_training_data.csv", rows)
    plot_yield_heatmap_flux_time(rows, outfile="ac225_yield_heatmap.png")


def main_pinn() -> None:
    rng = np.random.default_rng(43)
    rows = generate_pinn_training_runs(n_runs=1500, sigma_ra226=1e-24, rng=rng)
    save_pinn_training_csv("pinn_training_data.csv", rows)
    print("Wrote pinn_training_data.csv with", len(rows), "rows")


def demo_single_run() -> None:
    """Optional: one deterministic run and legacy time-series CSV + line plot."""
    env = IsotopeEnvironment(phi=1.0e14, sigma_ra226=1e-24)
    t_h, Y = run_simulation(env, t_end_h=200.0, n_points=501)
    N_ra = Y[:, 0]
    N_ra225 = Y[:, 1]
    N_ac = Y[:, 2]
    save_results_csv("raw_physics_data.csv", t_h, N_ra, N_ra225, N_ac)
    plot_ac225_growth(t_h, N_ac, outfile="ac225_growth.png")


if __name__ == "__main__":
    cmd = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    if cmd == "augment":
        n = augment_fail_zone_training()
        print(f"Appended {n} targeted rows to isotope_training_data.csv")
    elif cmd == "pinn":
        main_pinn()
    else:
        main()
