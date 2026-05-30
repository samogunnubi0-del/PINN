# Deploy IsotopePINN to Streamlit Community Cloud

Public URL for reviewers (professors, admissions supplements, demos).

## Required repo files (push to GitHub)

- `app.py`, `pinn_model.py`, `ra226_ac225_transmutation.py`, `graph_provenance.py`
- `weights/pinn_best_weights.pth` (v63, sha256 prefix `7c21debe`)
- `results/v63_validation_20260530.json`
- `analysis/validation/heldout_validation_summary.csv`
- `graphs/isef_parity_restyled.png`, `graphs/pinn_loss_history.png` (optional but recommended)
- `requirements-streamlit-cloud.txt` (not `requirements.txt` on cloud)

Do **not** rely on `kaggle_results/` — that folder is local only.

## Deploy steps

1. Push the [`New folder`](../) project root (contains `app.py`) to a **public** GitHub repo.
2. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/).
3. **New app** → connect GitHub → branch `main`.
4. **Main file path:** `app.py`
5. **Advanced settings → Python:** 3.11
6. **Advanced settings → Requirements file:** `requirements-streamlit-cloud.txt`  
   (CPU PyTorch; much smaller/faster than generic `requirements.txt`.)
7. Deploy. First **build** may take several minutes (PyTorch download). Subsequent **cold starts** after idle sleep are faster if the app uses cached model load (see `get_cached_pinn` in `app.py`).

## Cold start (inactive app)

Free tier apps **sleep after inactivity**. When a judge opens the URL after sleep:

- Container wakes (~30–90s typical after optimizations; was ~5 min before CPU torch + model cache).
- Sidebar shows: *"First open after sleep may take ~45–90 seconds."*
- Model loads **once** per container via `@st.cache_resource` — not on every widget click.

You cannot disable sleep on the free tier without upgrading.

## Optional secrets

App settings → **Secrets**:

```toml
PINN_BENCH_SCENARIOS = "200"
```

Use if a speed-benchmark loop is added later (limits ODE repeats on 1 GB RAM instances).

## Local test before deploy

```powershell
cd "C:\Users\ogunn\Downloads\New folder\New folder"
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\streamlit run app.py
```

First load may show "Loading PINN weights (v63)…". Change a slider — second interaction should **not** reload weights.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Infinite spinner / 5+ min | Confirm Cloud uses `requirements-streamlit-cloud.txt`; check Logs for OOM |
| "No PINN weights found" | Commit `weights/pinn_best_weights.pth` to GitHub |
| Wrong metrics on validation tab | Commit `results/v63_validation_20260530.json` |
| Build fails on torch | Pin `torch==2.2.2+cpu` in cloud requirements file |
