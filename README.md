# IsotopePINN

Physics-informed neural network surrogate for **Actinium-225** production planning in the Ra-226 transmutation chain — relevant to **targeted alpha therapy (TAT)** radiopharmaceutical supply.

**Live demo:** deploy with [Streamlit Community Cloud](docs/STREAMLIT_CLOUD_DEPLOY.md) (set `requirements-streamlit-cloud.txt`).

## Summary

| Item | Detail |
|------|--------|
| **Problem** | Ac-225 is scarce; planning irradiation (flux, energy, time) requires many stiff ODE solves |
| **Approach** | 0D five-species Bateman ODE reference (NNDC/JENDL) + PINN surrogate with physics loss |
| **Validation** | Six independent checks vs ODE; held-out Ac-225 median **~4.5%** relative error |
| **Not validated** | Laboratory reactor data, patient pharmacokinetics, or 3D transport (MCNP/OpenMC) |

## Results (ODE reference validation)

| Check | Result |
|-------|--------|
| Empty-target safety (no production from zero inventory) | PASS |
| Production scenario (14 MeV, full Ra-226 feed) | PASS (<10% Ac-225 vs ODE) |
| Decay-chain ingrowth (Ra-225 → Ac-225, no flux) | PASS |
| Species quality gate | PASS |
| PINN vs ODE correlation | PASS |
| Held-out scenarios (22 cases) | **4.51%** median Ac-225 error |

Full report: [`results/v63_validation_20260530.json`](results/v63_validation_20260530.json)

Weights checksum (SHA-256 prefix): `7c21debe` — file [`weights/pinn_best_weights.pth`](weights/pinn_best_weights.pth)

## Quick start (local)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open **Overview** → **Validation** → **About** in the app.

## Reproduce validation

```bash
pip install -r requirements.txt
python test_single.py
python analysis/validate_predictor.py
python analysis/evaluate_quality_gate.py
python analysis/correlation_check.py
```

Expected: all gates PASS; held-out Ac-225 median ~0.045 vs ODE.

## Training (optional)

Full retrain is GPU-heavy (~4k epochs with physics pretrain). Entry point:

```bash
python train.py
```

Colab-friendly bundle: see `IsotopePINN_Colab_Run.ipynb` in the workspace root or train with env vars documented in `train.py`.

## Deploy (Streamlit Cloud)

1. Push this repo to GitHub (public).
2. Connect at [share.streamlit.io](https://share.streamlit.io/) → main file `app.py`.
3. Requirements file: **`requirements-streamlit-cloud.txt`** (CPU PyTorch).
4. Ensure `weights/pinn_best_weights.pth` and `results/v63_validation_20260530.json` are committed.

Details: [`docs/STREAMLIT_CLOUD_DEPLOY.md`](docs/STREAMLIT_CLOUD_DEPLOY.md)

First load after idle sleep may take ~45–90 seconds (PyTorch + model load).

## Repository layout

| Path | Purpose |
|------|---------|
| `app.py` | Interactive demo (Overview, Screening, Validation, Methods, About) |
| `pinn_model.py`, `train.py` | PINN architecture and training |
| `ra226_ac225_transmutation.py` | Stiff ODE reference (Radau) |
| `test_single.py` | Scenario integrity tests (Trio A/B/C) |
| `analysis/` | Held-out validation, quality gate, correlation |
| `results/` | Validation JSON, loss history |
| `weights/` | Canonical trained checkpoint |
| `docs/DATA_ASSUMPTIONS.md` | Nuclear data sources and modeling scope |

## Modeling scope

This is a **0D lumped** well-mixed target model (scalar flux and energy), not a full reactor or patient dose model. Chemistry, recovery yield, and shipping delays are post-processed in the app, not inside the PINN loss.

## Limitations

- All reported errors are **PINN vs ODE**, not vs experiment.
- Largest errors near **epithermal (~9.5%)** and **threshold (~8.5%)** neutron energies.
- Do not use for regulatory release or clinical dosing without separate assay and qualified review.

## References

Methods and citations are listed in the app **About** tab (Raissi et al. PINN framework; NNDC/JENDL nuclear data).

## License

MIT — see [LICENSE](LICENSE).
