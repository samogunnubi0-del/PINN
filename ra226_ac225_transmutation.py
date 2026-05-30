"""
Bateman-style simulation for the five-species Ra-226 transmutation chain.

Two competing neutron channels on Ra-226:

  (n,2n)  Ra-226 -> Ra-225 -> Ac-225   (desired product)  [THRESHOLD reaction, ≥6.42 MeV]
  (n,γ)   Ra-226 -> Ra-227 -> Ac-227   (impurity pathway) [1/v reaction, active at thermal]

    dN_Ra226/dt = -lambda_226 * N_226 - k_n2n * N_226 - k_ngamma * N_226
    dN_Ra225/dt = +k_n2n * N_226 - lambda_225 * N_225
    dN_Ac225/dt = +lambda_225 * N_225 - lambda_Ac225 * N_Ac225
    dN_Ra227/dt = +k_ngamma * N_226 - lambda_227 * N_227
    dN_Ac227/dt = +lambda_227 * N_227 - lambda_Ac227 * N_Ac227

PHYSICS NOTES (NNDC / JENDL-5 / ENDF/B-VIII.0 verified):
  sigma_n2n(Ra-226):  threshold ~6.42 MeV; 27 mb spectrum-averaged (fast reactor).
                      ZERO for thermal neutrons. NOT 1/v.
  sigma_ngamma(Ra-226): ~12.8 barns at thermal (0.025 eV). Follows 1/v law.
  Ra-225 T½: 14.8 days (NNDC NuDat3 best value, prev 14.9 was within uncertainty)
  Ra-226 T½: 1600 years | Ac-225 T½: 9.92 days
  Ra-227 T½: 42.2 min  | Ac-227 T½: 21.772 years

Solver upgraded from odeint → solve_ivp with method='Radau' (A-stable stiff solver),
which handles the stiffness ratio λ_Ra227/λ_Ac225 ≈ 338 correctly.
"""

from __future__ import annotations

import csv
import os
import pathlib
import sys
from dataclasses import dataclass, field
from enum import Enum

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Physical constants  (NNDC NuDat3 / JENDL-5 / ENDF/B-VIII.0)
# ---------------------------------------------------------------------------
LN2 = np.log(2.0)

HALF_LIFE_RA226_H  = 1600.0   * 365.25 * 24.0   # 1600 y
HALF_LIFE_RA225_H  = 14.8     * 24.0             # 14.8 d  (NNDC best value)
HALF_LIFE_AC225_H  = 9.920    * 24.0             # 9.920 d  (NNDC best value)
HALF_LIFE_RA227_H  = 42.2     / 60.0             # 42.2 min
HALF_LIFE_AC227_H  = 21.772   * 365.25 * 24.0    # 21.772 y

# Cross sections (NNDC / JENDL-5)
SIGMA_NGAMMA_THERMAL_CM2 = 12.8e-24   # 12.8 barn thermal (n,γ) — correct, 1/v law applies
SIGMA_N2N_FAST_CM2       = 27e-27     # 27 mb spectrum-averaged fast reactor (JENDL-5)
                                       # ZERO for thermal energies (threshold ~6.42 MeV)

# (n,2n) threshold energy
E_THRESHOLD_N2N_EV = 6.42e6   # 6.42 MeV — Ra-226(n,2n) threshold (ENDF/B-VIII.0)

THERMAL_REFERENCE_EV = 0.025   # thermal energy reference for 1/v scaling
PINN_ENERGY_MIN_EV   = 0.025
PINN_ENERGY_MAX_EV   = 2.0e7   # 20 MeV — covers full fast neutron range


def sigma_scale_one_over_v(energy_ev: float | np.ndarray,
                            reference_ev: float = THERMAL_REFERENCE_EV) -> float | np.ndarray:
    """
    1/v cross-section scaling for (n,γ): σ(E) ∝ 1/v ∝ E^{-1/2}.
    Returns dimensionless factor σ(E)/σ(reference).
    Vectorised for numpy arrays.
    """
    E = np.maximum(np.asarray(energy_ev, dtype=float), 1e-30)
    return np.sqrt(float(reference_ev) / E)


def sigma_scale_threshold_n2n(energy_ev: float | np.ndarray,
                               threshold_ev: float = E_THRESHOLD_N2N_EV,
                               width_ev: float = 5e5) -> float | np.ndarray:
    """
    Smooth threshold model for (n,2n) cross section.
    Uses sigmoid so it is differentiable — important so the ODE data is smooth.
    Returns 0 well below threshold, rises to 1 above.
    Peak cross section is SIGMA_N2N_FAST_CM2 * this factor.
    """
    E = np.asarray(energy_ev, dtype=float)
    return 1.0 / (1.0 + np.exp(-(E - threshold_ev) / width_ev))


class NeutronEnergyGroup(str, Enum):
    THERMAL    = "Thermal"
    EPITHERMAL = "Epithermal"
    FAST       = "Fast"


# Energy group → approximate representative energy in eV for threshold calculation
_ENERGY_GROUP_EV: dict[str, float] = {
    NeutronEnergyGroup.THERMAL.value:    0.025,
    NeutronEnergyGroup.EPITHERMAL.value: 1.0,
    NeutronEnergyGroup.FAST.value:       14.0e6,
}


@dataclass
class IsotopeEnvironment:
    """
    Nuclear parameters for the five-species transmutation + decay model.

    Two competing neutron channels on Ra-226:
        (n,2n)  -> Ra-225  [threshold reaction; needs E > 6.42 MeV]
        (n,γ)   -> Ra-227  [1/v reaction; active at thermal energies]
    """
    phi: float = 1.0e14
    sigma_ra226: float = SIGMA_N2N_FAST_CM2       # (n,2n) reference [cm²]  27 mb
    sigma_ngamma: float = SIGMA_NGAMMA_THERMAL_CM2 # (n,γ) reference  [cm²] 12.8 b
    energy_group: str = NeutronEnergyGroup.FAST.value
    neutron_energy_ev: float | None = None
    target_mass_g: float = 1.0

    lambda_ra226_per_h: float = field(init=False)
    lambda_ra225_per_h: float = field(init=False)
    lambda_ac225_per_h: float = field(init=False)
    lambda_ra227_per_h: float = field(init=False)
    lambda_ac227_per_h: float = field(init=False)

    # Separate energy scales for the two channels
    _ng_energy_scale:  float = field(init=False)   # 1/v factor for (n,γ)
    _n2n_energy_scale: float = field(init=False)   # threshold factor for (n,2n)

    def __post_init__(self) -> None:
        self.lambda_ra226_per_h = LN2 / HALF_LIFE_RA226_H
        self.lambda_ra225_per_h = LN2 / HALF_LIFE_RA225_H
        self.lambda_ac225_per_h = LN2 / HALF_LIFE_AC225_H
        self.lambda_ra227_per_h = LN2 / HALF_LIFE_RA227_H
        self.lambda_ac227_per_h = LN2 / HALF_LIFE_AC227_H

        if self.neutron_energy_ev is not None:
            e_ev = float(self.neutron_energy_ev)
        else:
            e_ev = _ENERGY_GROUP_EV.get(self.energy_group, 14.0e6)

        # (n,γ): 1/v scaling (correct physics)
        self._ng_energy_scale = float(sigma_scale_one_over_v(e_ev))
        # (n,2n): threshold model (correct physics — zero below 6.42 MeV)
        self._n2n_energy_scale = float(sigma_scale_threshold_n2n(e_ev))

    def effective_sigma_n2n(self) -> float:
        """(n,2n) effective cross section [cm²] — zero for thermal neutrons."""
        return float(self.sigma_ra226) * self._n2n_energy_scale

    def effective_sigma_ngamma(self) -> float:
        """(n,γ) effective cross section [cm²] — 1/v scaled."""
        return float(self.sigma_ngamma) * self._ng_energy_scale

    def k_n2n_per_h(self) -> float:
        shielding = float(np.exp(-0.01 * self.target_mass_g))
        return float(self.phi) * shielding * self.effective_sigma_n2n() * 3600.0

    def k_ngamma_per_h(self) -> float:
        shielding = float(np.exp(-0.01 * self.target_mass_g))
        return float(self.phi) * shielding * self.effective_sigma_ngamma() * 3600.0

    # Legacy compatibility
    def effective_sigma_ra226(self) -> float:
        return self.effective_sigma_n2n()

    def transmutation_rate_constant_per_s(self) -> float:
        return float(self.phi) * self.effective_sigma_n2n()

    def transmutation_rate_constant_per_h(self) -> float:
        return self.transmutation_rate_constant_per_s() * 3600.0


def bateman_transmutation_rhs(
    t: float,
    N: np.ndarray,
    k_n2n: float,
    k_ng: float,
    lam226: float,
    lam225: float,
    lam_ac: float,
    lam227: float,
    lam_ac7: float,
) -> np.ndarray:
    """
    RHS for solve_ivp: dN/dt for five-species system.
    Signature is (t, N, *args) — note t comes first (solve_ivp convention).
    """
    N_ra226 = max(float(N[0]), 0.0)
    N_ra225 = max(float(N[1]), 0.0)
    N_ac225 = max(float(N[2]), 0.0)
    N_ra227 = max(float(N[3]), 0.0)
    N_ac227 = max(float(N[4]), 0.0)

    dN_ra226 = -(lam226 + k_n2n + k_ng) * N_ra226
    dN_ra225 =  k_n2n * N_ra226 - lam225 * N_ra225
    dN_ac225 =  lam225 * N_ra225 - lam_ac * N_ac225
    dN_ra227 =  k_ng * N_ra226 - lam227 * N_ra227
    dN_ac227 =  lam227 * N_ra227 - lam_ac7 * N_ac227

    return np.array([dN_ra226, dN_ra225, dN_ac225, dN_ra227, dN_ac227], dtype=float)


def run_simulation(
    env: IsotopeEnvironment,
    t_end_h: float = 200.0,
    n_points: int = 501,
    N_ra0: float = 6.022e23,
    N_ra225_0: float = 0.0,
    N_ac0: float = 0.0,
    N_ra227_0: float = 0.0,
    N_ac227_0: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Integrate the five-species system from t=0 to t=t_end_h using Radau (stiff solver).

    Returns (t_hours, Y) where Y has shape (n_points, 5):
        col 0: N_Ra226, col 1: N_Ra225, col 2: N_Ac225,
        col 3: N_Ra227, col 4: N_Ac227.
    """
    t_hours = np.linspace(0.0, float(t_end_h), int(n_points))
    y0 = np.array([N_ra0, N_ra225_0, N_ac0, N_ra227_0, N_ac227_0], dtype=float)

    k_n2n  = env.k_n2n_per_h()
    k_ng   = env.k_ngamma_per_h()
    lam226 = env.lambda_ra226_per_h
    lam225 = env.lambda_ra225_per_h
    lam_ac = env.lambda_ac225_per_h
    lam227 = env.lambda_ra227_per_h
    lam_ac7= env.lambda_ac227_per_h

    sol = solve_ivp(
        bateman_transmutation_rhs,
        t_span=(0.0, float(t_end_h)),
        y0=y0,
        method="Radau",          # A-stable stiff solver — handles Ra-227 T½=42min correctly
        t_eval=t_hours,
        args=(k_n2n, k_ng, lam226, lam225, lam_ac, lam227, lam_ac7),
        rtol=1e-9,
        atol=1e-12,
        dense_output=False,
    )

    Y = np.maximum(sol.y.T, 0.0)   # (n_points, 5), clip negatives from solver numerics
    return t_hours, Y


def ac225_peak_metrics(
    t_hours: np.ndarray,
    N_ac225: np.ndarray,
) -> tuple[float, float]:
    i_peak = int(np.argmax(N_ac225))
    return float(N_ac225[i_peak]), float(t_hours[i_peak])


def run_simulation_peak(
    env: IsotopeEnvironment,
    t_end_h: float,
    n_points: int | None = None,
    N_ra0: float = 6.022e23,
    N_ac0: float = 0.0,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    if n_points is None:
        n_points = max(400, int(np.clip(t_end_h * 8, 400, 15000)))
    t_h, Y = run_simulation(
        env, t_end_h=t_end_h, n_points=n_points, N_ra0=N_ra0, N_ra225_0=0.0, N_ac0=N_ac0
    )
    N_ac = Y[:, 2]
    max_yield, t_peak = ac225_peak_metrics(t_h, N_ac)
    return max_yield, t_peak, t_h, N_ac


def generate_training_runs(
    n_runs: int = 1000,
    sigma_ra226: float = SIGMA_N2N_FAST_CM2,
    rng: np.random.Generator | None = None,
    N_ra0: float = 6.022e23,
) -> list[dict[str, float]]:
    if rng is None:
        rng = np.random.default_rng()
    rows: list[dict[str, float]] = []
    for _ in range(n_runs):
        log_phi = rng.uniform(13.0, 15.0)
        phi = 10.0**log_phi
        total_time = float(rng.uniform(10.0, 500.0))
        env = IsotopeEnvironment(phi=phi, sigma_ra226=sigma_ra226)
        max_ac225, time_to_peak, _, _ = run_simulation_peak(env, total_time, N_ra0=N_ra0)
        rows.append({
            "phi": phi, "sigma": sigma_ra226,
            "total_time": total_time,
            "max_ac225_yield": max_ac225, "time_to_peak": time_to_peak,
        })
    return rows


TARGETED_PHI_LOG_MIN = 11.0
TARGETED_PHI_LT      = 1e13
TARGETED_TIME_MIN_H  = 500.0
TARGETED_TIME_MAX_H  = 8760.0


def generate_targeted_data(
    n_runs: int = 200,
    sigma_ra226: float = SIGMA_N2N_FAST_CM2,
    rng: np.random.Generator | None = None,
    N_ra0: float = 6.022e23,
) -> list[dict[str, float]]:
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
        rows.append({
            "phi": phi, "sigma": sigma_ra226,
            "total_time": total_time,
            "max_ac225_yield": max_ac225, "time_to_peak": time_to_peak,
        })
    return rows


def append_training_csv(path: str, rows: list[dict[str, float]]) -> None:
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
    sigma_ra226: float = SIGMA_N2N_FAST_CM2,
    rng: np.random.Generator | None = None,
) -> int:
    rng = rng or np.random.default_rng(2026)
    new_rows = generate_targeted_data(n_runs=n_targeted, sigma_ra226=sigma_ra226, rng=rng)
    append_training_csv(csv_path, new_rows)
    return len(new_rows)


def save_training_csv(path: str, rows: list[dict[str, float]]) -> None:
    fieldnames = ["phi", "sigma", "total_time", "max_ac225_yield", "time_to_peak"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def generate_pinn_training_runs(
    n_runs: int = 1500,
    sigma_ra226: float = SIGMA_N2N_FAST_CM2,
    rng: np.random.Generator | None = None,
    N_ra0: float = 6.022e23,
) -> list[dict[str, float]]:
    """
    PINN-oriented dataset: random flux (log-uniform), neutron energy (eV, log-uniform
    between PINN_ENERGY_MIN_EV and PINN_ENERGY_MAX_EV = 20 MeV), and irradiation time.
    Energy range now covers fast neutrons so (n,2n) threshold fires at correct energies.
    """
    if rng is None:
        rng = np.random.default_rng()
    log_e_min = np.log10(PINN_ENERGY_MIN_EV)
    log_e_max = np.log10(PINN_ENERGY_MAX_EV)
    rows: list[dict[str, float]] = []
    for _ in range(n_runs):
        log_phi   = float(rng.uniform(13.0, 15.0))
        phi       = 10.0**log_phi
        time_h    = float(rng.uniform(10.0, 500.0))
        log_e     = float(rng.uniform(log_e_min, log_e_max))
        energy_ev = float(10.0**log_e)

        env = IsotopeEnvironment(phi=phi, sigma_ra226=sigma_ra226, neutron_energy_ev=energy_ev)
        n_pts = max(400, int(np.clip(time_h * 8, 400, 15000)))
        _, Y  = run_simulation(env, t_end_h=time_h, n_points=n_pts, N_ra0=N_ra0)
        end   = Y[-1]

        rows.append({
            "phi": phi, "energy": energy_ev, "time": time_h,
            "N_Ra226": float(end[0]), "N_Ra225": float(end[1]),
            "N_Ac225": float(end[2]), "N_Ra227": float(end[3]),
            "N_Ac227": float(end[4]),
        })
    return rows


def save_pinn_training_csv(path: str, rows: list[dict[str, float]]) -> None:
    fieldnames = ["phi", "energy", "time", "N_Ra226", "N_Ra225", "N_Ac225", "N_Ra227", "N_Ac227"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def plot_yield_heatmap_flux_time(
    rows: list[dict[str, float]],
    outfile: str = "graphs/ac225_yield_heatmap.png",
    n_bins_flux: int = 24,
    n_bins_time: int = 24,
) -> None:
    phi  = np.array([r["phi"] for r in rows], dtype=float)
    tt   = np.array([r["total_time"] for r in rows], dtype=float)
    yld  = np.array([r["max_ac225_yield"] for r in rows], dtype=float)
    phi_edges  = np.logspace(13.0, 15.0, n_bins_flux + 1)
    time_edges = np.linspace(10.0, 500.0, n_bins_time + 1)
    stat, _, _, _ = stats.binned_statistic_2d(phi, tt, yld, statistic="max",
                                               bins=[phi_edges, time_edges])
    Z = np.ma.masked_invalid(stat.T)
    fig, ax = plt.subplots(figsize=(10, 6))
    X, Y_m  = np.meshgrid(phi_edges, time_edges)
    pcm = ax.pcolormesh(X, Y_m, Z, shading="auto", cmap="viridis")
    fig.colorbar(pcm, ax=ax, label=r"max $^{225}$Ac yield (atoms)")
    ax.set_xscale("log")
    ax.set_xlabel(r"Neutron flux $\phi$ (n cm$^{-2}$ s$^{-1}$)")
    ax.set_ylabel("Irradiation time (hours)")
    ax.set_title(r"$^{225}$Ac yield heatmap (max per bin over Monte Carlo runs)")
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    try:
        import graph_provenance
        graph_provenance.record_graph_write(
            pathlib.Path(__file__).resolve().parent,
            pathlib.Path(outfile).resolve(),
            producer="ra226_ac225_transmutation.py",
            run_id=graph_provenance.new_run_id(),
        )
    except Exception:
        pass


def save_results_csv(path, t_hours, N_ra226, N_ra225, N_ac225):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_h", "N_Ra226", "N_Ra225", "N_Ac225"])
        for i in range(len(t_hours)):
            w.writerow([t_hours[i], N_ra226[i], N_ra225[i], N_ac225[i]])


def plot_ac225_growth(t_hours, N_ac225, outfile="graphs/ac225_growth.png"):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t_hours, N_ac225, color="darkgreen", lw=2, label=r"$^{225}$Ac")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel(r"$N_{^{225}\mathrm{Ac}}$ (atoms)")
    ax.set_title(r"Growth of $^{225}$Ac from $^{226}$Ra transmutation")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if outfile:
        fig.savefig(outfile, dpi=150)
        try:
            import graph_provenance
            graph_provenance.record_graph_write(
                pathlib.Path(__file__).resolve().parent,
                pathlib.Path(outfile).resolve(),
                producer="ra226_ac225_transmutation.py",
                run_id=graph_provenance.new_run_id(),
            )
        except Exception:
            pass
    plt.close(fig)


def main() -> None:
    rng = np.random.default_rng(42)
    rows = generate_training_runs(n_runs=1000, sigma_ra226=SIGMA_N2N_FAST_CM2, rng=rng)
    save_training_csv("isotope_training_data.csv", rows)
    plot_yield_heatmap_flux_time(rows, outfile="graphs/ac225_yield_heatmap.png")


def main_pinn() -> None:
    rng = np.random.default_rng(43)
    rows = generate_pinn_training_runs(n_runs=1500, sigma_ra226=SIGMA_N2N_FAST_CM2, rng=rng)
    save_pinn_training_csv("pinn_training_data.csv", rows)
    print("Wrote pinn_training_data.csv with", len(rows), "rows")


def demo_single_run() -> None:
    env = IsotopeEnvironment(phi=1.0e14, sigma_ra226=SIGMA_N2N_FAST_CM2,
                             neutron_energy_ev=14e6)  # 14 MeV fast
    t_h, Y = run_simulation(env, t_end_h=200.0, n_points=501)
    save_results_csv("raw_physics_data.csv", t_h, Y[:, 0], Y[:, 1], Y[:, 2])
    plot_ac225_growth(t_h, Y[:, 2], outfile="graphs/ac225_growth.png")


if __name__ == "__main__":
    cmd = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    if cmd == "augment":
        n = augment_fail_zone_training()
        print(f"Appended {n} targeted rows to isotope_training_data.csv")
    elif cmd == "pinn":
        main_pinn()
    else:
        main()
