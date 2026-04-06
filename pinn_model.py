"""
Physics-informed neural network backbone for isotope transmutation states.

SCALING CONTRACT (must match train.py exactly)
===============================================
Input columns fed to the network (all in [0, 1] after normalisation):

  col 0   t_nn    = t_hours / T_REF_H            (T_REF_H   = 500)
  col 1   phi_nn  = phi / PHI_SCALE              (PHI_SCALE = 1e15)
  col 2   E_nn    = sqrt(E_ref / E_eV)           (E_ref=0.025 eV; 1/v sigma ratio)
  col 3   n226_nn = N_Ra226 / N226_SCALE          (N226_SCALE = 6.022e23)
  col 4   n225_nn = N_Ra225 / N225_SCALE          (N225_SCALE = 1e20)
  col 5   nac_nn  = N_Ac225 / NAC_SCALE           (NAC_SCALE  = 1e20)

Outputs: (n226_nn_pred, n225_nn_pred, nac_nn_pred) in the same normalised space.

OUTPUT (HARD BUDGET)
====================
After the MLP, inventories are rescaled so raw-atom totals never exceed
the batch initial total (empty tank -> all zeros).

LOSS TERMS (Max-Fix v2)
=======================
1. physics_mse       -- Bateman residuals; Ra-225 equation weighted x5
2. data_mse          -- CSV fit (per-species weighted)
3. mass_cons_loss    -- no net atom creation
4. fuel_anchor_loss  -- Ra-226 burnup anchor
5. non_neg_loss      -- penalty for negative pre-clamp activations
6. secular_eq_loss   -- Ac-225 cannot exceed transient-equilibrium ceiling
7. zero_injection    -- empty-tank must stay empty

NUCLEAR DATA (NNDC 2024)
========================
Ra-226 half-life: 1600 +/- 7 years
Ra-225 half-life: 14.9 +/- 0.2 days (beta- to Ac-225)
Ac-225 half-life: 9.920 +/- 0.003 days (alpha)
Reference sigma:  1 barn (tunable; (n,2n) channel schematic)
1/v law:          sigma(E) = sigma_th * sqrt(0.025 eV / E)
"""
from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# -- Physical constants (hours, NNDC verified) --------------------------------
_LN2 = math.log(2.0)

THERMAL_REFERENCE_EV = 0.025

# Ra-226: 1600 y  (NNDC)
DEFAULT_LAMBDA_226_H = _LN2 / (1600.0 * 365.25 * 24.0)
# Ra-225: 14.9 d  (NNDC)
DEFAULT_LAMBDA_225_H = _LN2 / (14.9  * 24.0)
# Ac-225: 9.920 d (NNDC)
DEFAULT_LAMBDA_AC_H  = _LN2 / (9.920 * 24.0)

# Transient equilibrium ratio: N_Ac225 / N_Ra225 at steady state
# = lambda_225 / (lambda_Ac - lambda_225)   (valid when lambda_Ac > lambda_225)
SECULAR_EQ_RATIO = DEFAULT_LAMBDA_225_H / (DEFAULT_LAMBDA_AC_H - DEFAULT_LAMBDA_225_H)

# -- Scaling defaults (MUST match train.py) ------------------------------------
DEFAULT_N226_SCALE = 6.022e23   # ~1 mole
DEFAULT_N225_SCALE = 1e20
DEFAULT_NAC_SCALE  = 1e20
DEFAULT_PHI_SCALE  = 1e15       # max reactor flux -> phi_nn in [0, 1]
DEFAULT_T_REF_H    = 500.0      # max irradiation time -> t_nn in [0, 1]

SECONDS_PER_HOUR   = 3_600.0
N_INPUT_FEATURES   = 6

_ENERGY_EV_CLIP = (1e-12, 1e10)


def neutron_energy_ev_to_feature_torch(energy_ev: torch.Tensor) -> torch.Tensor:
    """Network input col 2: sqrt(E_ref / E) -- matches 1/v sigma ratio."""
    E = energy_ev.clamp(min=_ENERGY_EV_CLIP[0], max=_ENERGY_EV_CLIP[1])
    ref = torch.as_tensor(THERMAL_REFERENCE_EV, dtype=E.dtype, device=E.device)
    return torch.sqrt(ref / E)


def neutron_energy_ev_to_feature_numpy(energy_ev: np.ndarray | float) -> np.ndarray:
    """Same mapping for NumPy pipelines."""
    a = np.asarray(energy_ev, dtype=np.float64)
    a = np.clip(a, _ENERGY_EV_CLIP[0], _ENERGY_EV_CLIP[1])
    return np.sqrt(THERMAL_REFERENCE_EV / a)


# ==============================================================================
# Model
# ==============================================================================
class IsotopePINN(nn.Module):
    """
    MLP with hard initial-condition (IC) constraint::

        N_nn(t) = N_nn_0 + t_nn * NN(x)

    forward_raw: physics training (allows slight negatives for gradient flow).
    forward:     inference with hard budget cap and non-negativity clamp.
    """

    def __init__(
        self,
        hidden_dim:  int   = 128,
        n_hidden:    int   = 4,
        *,
        signed_rate: bool  = True,
        t_zero_eps:  float = 0.001,
    ) -> None:
        super().__init__()
        if n_hidden < 1:
            raise ValueError("n_hidden must be >= 1")

        layers: list[nn.Module] = []
        in_dim = N_INPUT_FEATURES
        for _ in range(n_hidden):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        self.hidden      = nn.Sequential(*layers)
        self.head        = nn.Linear(hidden_dim, 3)
        self.signed_rate = signed_rate
        self.t_zero_eps  = t_zero_eps
        self.rate_abs_max = 10.0

    def forward_raw(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass WITHOUT hard budget cap.

        IC guarantee: at t_nn = 0 the output is exactly n0 (no activation
        attenuation).  The non-negativity penalty in the loss function
        discourages negative outputs during training; forward() hard-clamps
        for inference.
        """
        squeeze = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze = True

        t_nn = x[:, 0:1]
        n0   = x[:, 3:6]

        h    = self.hidden(x)
        raw  = self.head(h)
        rate = raw if self.signed_rate else F.softplus(raw)
        rate = rate.clamp(min=-self.rate_abs_max, max=self.rate_abs_max)

        t_scale = torch.sigmoid(t_nn * 1000.0)
        output = n0 + t_scale * t_nn * rate
        # NO SiLU here: SiLU(x) = x*sigmoid(x) halves small positive x,
        # destroying the IC (n0 -> 0.5*n0 for normalized inventories ~0.01).
        # Non-negativity penalty handles negative predictions during training.
        return output.squeeze(0) if squeeze else output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass WITH non-negativity clamp + hard budget cap (inference safe)."""
        squeeze = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze = True

        smoothed = self.forward_raw(x)
        if smoothed.dim() == 1:
            smoothed = smoothed.unsqueeze(0)

        # Hard non-negativity for inference (atoms cannot be negative)
        smoothed = smoothed.clamp(min=0.0)

        n0 = x[:, 3:6]

        # Hard cap: total predicted atoms <= total initial atoms
        s226 = x.new_tensor(DEFAULT_N226_SCALE)
        s225 = x.new_tensor(DEFAULT_N225_SCALE)
        sac  = x.new_tensor(DEFAULT_NAC_SCALE)
        tot0 = n0[:, 0:1] * s226 + n0[:, 1:2] * s225 + n0[:, 2:3] * sac
        totp = smoothed[:, 0:1] * s226 + smoothed[:, 1:2] * s225 + smoothed[:, 2:3] * sac
        tiny = x.new_tensor(1e-12)
        empty_start = tot0 <= tiny
        excess = totp > tot0 + tiny
        scale = torch.ones_like(totp)
        scale = torch.where(excess & ~empty_start, tot0 / (totp + tiny), scale)
        scale = torch.where(empty_start & (totp > tiny), torch.zeros_like(scale), scale)
        out = smoothed * scale

        return out.squeeze(0) if squeeze else out


# -- Helpers -------------------------------------------------------------------
def _phi_sigma_per_hour(
    phi_raw:      torch.Tensor,
    sigma_cm2:    float,
    energy_scale: torch.Tensor,
) -> torch.Tensor:
    """Transmutation rate k [h^-1] = phi * sigma(E) * 3600."""
    return phi_raw * (sigma_cm2 * energy_scale) * SECONDS_PER_HOUR


def time_derivatives_wrt_t(
    pred:   torch.Tensor,
    inputs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dts: list[torch.Tensor] = []
    for i in range(3):
        (ginp,) = torch.autograd.grad(
            pred[:, i].sum(),
            inputs,
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )
        dts.append(ginp[:, 0:1])
    return dts[0], dts[1], dts[2]


def time_derivatives_finite_diff(
    forward_fn:  Callable[[torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
    pred:   torch.Tensor,
    *,
    eps:         float = 5e-4,
    t_nn_max:    float = 1.0e3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """dN_nn/dt_input by central finite difference (forward diff at t~0)."""
    t0 = inputs[:, 0:1]
    dt = inputs.new_tensor(eps)

    t_low_unclamped = t0 - dt
    t_high_unclamped = t0 + dt

    at_lower_bound = t_low_unclamped <= 0.0

    t_low_clamped = t_low_unclamped.clamp(min=0.0)
    t_high_clamped = t_high_unclamped.clamp(max=t_nn_max)
    denom_central = (t_high_clamped - t_low_clamped).clamp(min=dt * 0.5)

    low_in = torch.cat([t_low_clamped, inputs[:, 1:]], dim=1)
    high_in = torch.cat([t_high_clamped, inputs[:, 1:]], dim=1)
    pred_low = forward_fn(low_in)
    pred_high = forward_fn(high_in)

    d_central = (pred_high - pred_low) / denom_central

    t_fwd = t0 + 2.0 * dt
    t_fwd_clamped = t_fwd.clamp(max=t_nn_max)
    denom_fwd = (t_fwd_clamped - t0).clamp(min=dt)

    fwd_in = torch.cat([t_fwd_clamped, inputs[:, 1:]], dim=1)
    pred_fwd = forward_fn(fwd_in)

    d_fwd = (pred_fwd - pred) / denom_fwd

    d_all = torch.where(at_lower_bound.expand_as(d_central), d_fwd, d_central)
    d_all = d_all.clamp(min=-5.0e3, max=5.0e3)

    return d_all[:, 0:1], d_all[:, 1:2], d_all[:, 2:3]


# ==============================================================================
# Combined loss: physics + data + mass + non-neg + secular eq
# ==============================================================================
def compute_physics_loss(
    model:   IsotopePINN,
    inputs:  torch.Tensor,
    pred:    torch.Tensor,
    targets: torch.Tensor | None = None,
    *,
    sigma_cm2:             float               = 1e-24,
    energy_scale:          torch.Tensor | None = None,
    use_one_over_v_energy: bool                = True,
    lambda_226_h:          float               = DEFAULT_LAMBDA_226_H,
    lambda_225_h:          float               = DEFAULT_LAMBDA_225_H,
    lambda_ac_h:           float               = DEFAULT_LAMBDA_AC_H,
    physics_weight:        float               = 1.0,
    data_weight:           float               = 1.0,
    mass_weight:           float               = 10.0,
    fuel_anchor_weight:    float               = 0.0,
    non_neg_weight:        float               = 50.0,
    secular_eq_weight:     float               = 25.0,
    ra225_physics_weight:  float               = 5.0,
    pred_for_data:         torch.Tensor | None = None,
    data_species_weights:  tuple[float, float, float] = (1.0, 1.0, 1.0),
    n226_scale:            float               = DEFAULT_N226_SCALE,
    n225_scale:            float               = DEFAULT_N225_SCALE,
    nac_scale:             float               = DEFAULT_NAC_SCALE,
    phi_scale:             float               = DEFAULT_PHI_SCALE,
    t_ref_h:               float               = DEFAULT_T_REF_H,
    d_t_input_d_t_hours:   float               = 1.0 / DEFAULT_T_REF_H,
    atom_scale:            float | None        = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Combined PINN loss with Max-Fix enhancements:

    - 1/v energy scaling on cross-section (sigma * sqrt(E_ref/E))
    - Ra-225 Bateman residual weighted by ra225_physics_weight (default 5.0)
    - Non-negativity penalty on pre-clamp predictions
    - Secular equilibrium ceiling on Ac-225 / Ra-225 ratio
    - Mass conservation (no alchemy)
    - Zero-injection (empty tank stays empty)
    """
    if inputs.dim() != 2 or inputs.size(-1) != N_INPUT_FEATURES:
        raise ValueError(
            f"inputs must be (batch, {N_INPUT_FEATURES}): "
            "[t_nn, phi_nn, E, n226_nn_0, n225_nn_0, nac_nn_0]"
        )
    if pred.dim() != 2 or pred.size(-1) != 3 or pred.size(0) != inputs.size(0):
        raise ValueError("pred must be (batch, 3) matching inputs batch dim.")
    pred_data = pred
    if pred_for_data is not None:
        if pred_for_data.shape != pred.shape:
            raise ValueError("pred_for_data must match pred shape (batch, 3).")
        pred_data = pred_for_data
    if not inputs.requires_grad:
        raise ValueError("inputs must have requires_grad=True for autograd.")

    dev   = pred.device
    dtype = pred.dtype

    # -- recover raw phi -------------------------------------------------------
    phi_raw = inputs[:, 1:2] * phi_scale

    # -- 1/v energy scaling: sigma(E) = sigma_th * sqrt(E_ref / E) ------------
    # Column 2 IS sqrt(E_ref/E), so it's the 1/v scaling factor directly.
    if energy_scale is None:
        if use_one_over_v_energy:
            energy_scale = inputs[:, 2:3].clamp(min=1e-20, max=1e6)
        else:
            energy_scale = torch.ones_like(phi_raw)
    if energy_scale.shape != phi_raw.shape:
        raise ValueError("energy_scale must match phi shape (batch, 1).")

    n226 = pred[:, 0:1]
    n225 = pred[:, 1:2]
    nac  = pred[:, 2:3]

    # -- time derivatives: finite difference -> dN_nn/dt_h ---------------------
    d226_nn, d225_nn, dac_nn = time_derivatives_finite_diff(model.forward_raw, inputs, pred)
    c    = torch.as_tensor(d_t_input_d_t_hours, dtype=dtype, device=dev)
    d226 = d226_nn * c
    d225 = d225_nn * c
    dac  = dac_nn  * c

    # -- rate constants --------------------------------------------------------
    k_h    = _phi_sigma_per_hour(phi_raw, sigma_cm2, energy_scale).clamp(min=0.0, max=5.0e3)
    lam226 = torch.as_tensor(lambda_226_h, dtype=dtype, device=dev)
    lam225 = torch.as_tensor(lambda_225_h, dtype=dtype, device=dev)
    lam_ac = torch.as_tensor(lambda_ac_h,  dtype=dtype, device=dev)

    r226_225 = n226_scale / n225_scale
    r225_ac  = n225_scale / nac_scale

    # -- Bateman residuals -----------------------------------------------------
    f1 = d226 + (lam226 + k_h) * n226
    f2 = d225 - (k_h * n226 * r226_225 - lam225 * n225)
    f3 = dac  - (lam225 * n225 * r225_ac - lam_ac * nac)

    tr    = torch.as_tensor(t_ref_h, dtype=dtype, device=dev)
    f_lim = torch.as_tensor(5.0e3, dtype=dtype, device=dev)
    f1 = f1.clamp(min=-f_lim, max=f_lim)
    f2 = f2.clamp(min=-f_lim, max=f_lim)
    f3 = f3.clamp(min=-f_lim, max=f_lim)

    # Ra-225 Bateman residual weighted x5 to fix underprediction
    ra225_w = torch.as_tensor(ra225_physics_weight, dtype=dtype, device=dev)
    physics_mse = (
        (f1 * tr).pow(2) + ra225_w * (f2 * tr).pow(2) + (f3 * tr).pow(2)
    ).mean()

    # -- Ra-226 burnup anchor --------------------------------------------------
    n226_0_in = inputs[:, 3:4]
    s226_mass = torch.as_tensor(n226_scale, dtype=dtype, device=dev)
    raw226_0 = n226_0_in * s226_mass
    t_h = inputs[:, 0:1] * torch.as_tensor(t_ref_h, dtype=dtype, device=dev)
    burn_exp = (-(lam226 + k_h) * t_h).clamp(min=-80.0, max=80.0)
    n226_burn = n226_0_in * torch.exp(burn_exp)
    m_fuel = raw226_0.squeeze(-1) > 1.0e16
    if bool(m_fuel.any().item()):
        fuel_anchor_loss = F.mse_loss(n226[m_fuel], n226_burn[m_fuel])
    else:
        fuel_anchor_loss = torch.zeros((), device=dev, dtype=dtype)

    # -- data loss -------------------------------------------------------------
    if targets is not None:
        if targets.shape != pred.shape:
            raise ValueError("targets must match pred shape (batch, 3).")
        s226t = torch.as_tensor(n226_scale, dtype=dtype, device=dev)
        s225t = torch.as_tensor(n225_scale, dtype=dtype, device=dev)
        sact  = torch.as_tensor(nac_scale,  dtype=dtype, device=dev)
        targets_nn = torch.stack([
            targets[:, 0] / s226t,
            targets[:, 1] / s225t,
            targets[:, 2] / sact,
        ], dim=1)
        se = (pred_data - targets_nn).pow(2)
        w = torch.as_tensor(data_species_weights, dtype=dtype, device=dev).view(1, 3)
        data_mse = (se * w.expand_as(se)).mean()
    else:
        data_mse = torch.zeros((), device=dev, dtype=dtype)

    # -- non-negativity penalty ------------------------------------------------
    # forward_raw can produce slight negatives via SiLU; penalize them smoothly
    non_neg_loss = F.relu(-pred).pow(2).mean()

    # -- secular equilibrium ceiling for Ac-225 --------------------------------
    # Transient eq: N_Ac225 <= (lam225 / (lam_ac - lam225)) * N_Ra225
    # In normalized space (same scale for Ra225, Ac225):
    eq_ratio = lam225 / (lam_ac - lam225)  # ~1.99
    scale_ratio = n225_scale / nac_scale    # 1.0 when scales are equal
    nac_ceiling = eq_ratio * scale_ratio * n225.detach().clamp(min=0.0)
    secular_eq_loss = F.relu(nac - nac_ceiling).pow(2).mean()

    # -- mass conservation loss ------------------------------------------------
    n226_0 = inputs[:, 3:4]
    n225_0 = inputs[:, 4:5]
    nac_0  = inputs[:, 5:6]

    s226 = torch.as_tensor(n226_scale, dtype=dtype, device=dev)
    s225 = torch.as_tensor(n225_scale, dtype=dtype, device=dev)
    sac  = torch.as_tensor(nac_scale,  dtype=dtype, device=dev)

    total_start_raw = n226_0 * s226 + n225_0 * s225 + nac_0 * sac
    n226_d, n225_d, nac_d = pred_data[:, 0:1], pred_data[:, 1:2], pred_data[:, 2:3]
    total_pred_raw = n226_d * s226 + n225_d * s225 + nac_d * sac

    excess_raw = F.relu(total_pred_raw - total_start_raw)
    mass_ref = torch.as_tensor(1.0e15, dtype=dtype, device=dev)
    ratio_empty = excess_raw / (total_start_raw + mass_ref)
    term_log = torch.log1p(ratio_empty.clamp(max=1.0e12)).pow(2)
    meaningful_start_atoms = torch.as_tensor(1.0e20, dtype=dtype, device=dev)
    has_start_m = total_start_raw.squeeze(-1) > meaningful_start_atoms
    term_rel_sq = (
        (excess_raw / total_start_raw.clamp(min=1.0e-30))
        .pow(2)
        .clamp(max=1.0e4)
    )
    term_rel = torch.where(
        has_start_m.unsqueeze(-1),
        term_rel_sq,
        torch.zeros_like(excess_raw),
    )
    mass_cons_loss = (term_log + term_rel).mean()

    # -- zero-injection penalty ------------------------------------------------
    empty_mask = (n226_0.squeeze(-1) < 1e-8) & \
                 (n225_0.squeeze(-1) < 1e-8) & \
                 (nac_0.squeeze(-1)  < 1e-8)

    zero_injection_loss = torch.zeros((), device=dev, dtype=dtype)
    if empty_mask.any():
        nonzero_outputs = (
            n226_d[empty_mask].abs() + n225_d[empty_mask].abs() + nac_d[empty_mask].abs()
        )
        zero_injection_loss = nonzero_outputs.mean()

    # -- combine all losses ----------------------------------------------------
    zero_injection_weight = 100.0 if targets is not None else 200.0

    total = (
        physics_weight      * physics_mse
        + data_weight       * data_mse
        + mass_weight       * mass_cons_loss
        + fuel_anchor_weight * fuel_anchor_loss
        + non_neg_weight    * non_neg_loss
        + secular_eq_weight * secular_eq_loss
        + zero_injection_weight * zero_injection_loss
    )

    return total, {
        "total_loss":          total,
        "data_mse":            data_mse,
        "physics_mse":         physics_mse,
        "mass_cons_loss":      mass_cons_loss,
        "fuel_anchor_loss":    fuel_anchor_loss,
        "non_neg_loss":        non_neg_loss,
        "secular_eq_loss":     secular_eq_loss,
        "zero_injection_loss": zero_injection_loss,
        "f1": f1, "f2": f2, "f3": f3,
    }
