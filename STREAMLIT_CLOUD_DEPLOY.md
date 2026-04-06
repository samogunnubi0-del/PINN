# Deploy to Streamlit Community Cloud (option one)

This gives you a **stable public URL** (e.g. `https://your-app.streamlit.app`) so judges can open the dashboard without your Wi‑Fi.

## What you must do (about 15 minutes)

### 1) Put the project on GitHub

Use **this folder** as the repo root (the one that contains `app.py` and `requirements.txt`).

```powershell
cd "C:\Users\ogunn\Downloads\New folder\New folder"
git init
git add .
git commit -m "Initial commit: Ac-225 PINN dashboard"
```

Create a **new public repository** on GitHub, then:

```powershell
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git branch -M main
git push -u origin main
```

Use **public** for the simplest free Community Cloud deploy. If the repo must stay private, Community Cloud may require a paid workspace—check current Streamlit pricing.

### 2) Deploy on Streamlit Community Cloud

1. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/).
2. **New app** → connect GitHub → pick the repo and `main`.
3. **Main file path:** `app.py`
4. **Advanced settings → Python version:** 3.11 (recommended).
5. **Advanced settings → Requirements file:** choose `requirements-streamlit-cloud.txt`  
   (CPU-friendly **PyTorch** install on Linux; your local `requirements.txt` stays unchanged for Windows.)
6. Deploy. First build can take **several minutes** (PyTorch download).

### 3) Optional secrets (recommended on free tier)

In the deployed app: **Settings → Secrets**, add:

```toml
# Fewer ODE repeats avoids long runs and timeouts on small instances
PINN_BENCH_SCENARIOS = "200"
```

Restart the app after saving.

### 4) Share with judges

Use the **Streamlit URL** and/or generate a QR code pointing to it. No local firewall or LAN setup needed.

## Notes specific to this project

- **Weights and plots** used by the app should stay in the repo (they are already small enough for GitHub).
- **Ephemeral disk:** “Reactor Calibration” / finetune writes disappear when the app sleeps or redeploys—that is normal on Community Cloud.
- **Heavy tab:** Speed benchmark runs many **ordinary differential equation (ODE)** solves; use `PINN_BENCH_SCENARIOS` on cloud.
- If the build fails or the app runs out of memory, try a smaller **PyTorch** pin in `requirements-streamlit-cloud.txt` or upgrade your Streamlit plan.

## config.toml

Local `.streamlit/config.toml` binds `0.0.0.0` for home networks; **Community Cloud overrides** host/port as needed—you do not need to change this file for the cloud.
