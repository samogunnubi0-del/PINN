# Ac-225 Production Intelligence

A **physics-informed neural network (PINN)** that predicts Actinium-225 yields from the
Ra-226 -> Ra-225 -> Ac-225 transmutation chain. Built for targeted alpha therapy production
planning.

## Quick Start

From the repository root (where `app.py` lives), in PowerShell:

```powershell
cd path\to\this-repo
.\venv\Scripts\Activate.ps1
streamlit run app.py
```

Opens at **http://localhost:8501** with 13 interactive tabs.

## What This Is

- A 4-layer MLP trained on 12,000 epochs of physics-informed loss
- Learns Bateman decay equations, mass conservation, and 1/v energy scaling
- Runs **500x+ faster** than traditional ODE solvers
- Includes live reactor calibration via transfer learning (upload CSV, 30-second adapt)

## Trio Validation (all pass)

| Test | Result |
|------|--------|
| A - Empty tank + flux | PASS (no alchemy) |
| B - Ra-226 feed + flux | PASS (Ac-225 vs ODE within ~25%; see `test_single.py`) |
| C - Pure decay | PASS (correct chain, no ghost Ra-226) |

## Key Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit website (13 tabs, live prediction, training, calibration) |
| `pinn_model.py` | PINN architecture + 8-term loss function |
| `train.py` | Training pipeline (12k epochs, diverse ICs, augmentation) |
| `test_single.py` | Trio validation test |
| `ra226_ac225_transmutation.py` | ODE reference simulator |
| `pinn_trained_weights.pth` | Pre-trained model weights |

## Website Tabs

1. **Why This Matters** - Ac-225 supply crisis, targeted alpha therapy
2. **Live Prediction** - Interactive sliders, real-time isotope predictions
3. **Live Training** - Watch the model learn + upload reactor data for calibration
4. **Speed Benchmark** - PINN vs ODE timing (1,000 scenarios)
5. **Dose Calculator** - Atoms to patient doses conversion
6. **Project Timeline** - Animated orbital development history
7. **Training Results** - Loss curves and parity plot
8. **Plot Gallery** - Every visualization explained
9. **Trio Validation** - Physics correctness tests
10. **Struggles & Failures** - Honest bug documentation
11. **Mistakes We Made** - Retrospective on judgment errors
12. **Future Applications** - Clinical, supply chain, digital twin
13. **Technical Details** - Architecture, loss, nuclear data, equations

## Nuclear Data (NNDC)

| Isotope | Half-life | Decay |
|---------|-----------|-------|
| Ra-226 | 1,600 years | Alpha |
| Ra-225 | 14.9 days | Beta- to Ac-225 |
| Ac-225 | 9.920 days | Alpha (4 alphas in chain) |

## Training

Full training (one-time, ~2 hours on i7):

```powershell
.\venv\Scripts\python.exe train.py
```

Test:

```powershell
.\venv\Scripts\python.exe test_single.py
```
