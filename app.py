"""
PINN Isotope Transmutation -- Showcase Website
Nuclear-medicine-themed Streamlit dashboard (13 tabs).
Run: streamlit run app.py
"""
import io
import os
import pathlib
import socket

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import qrcode
import torch
from PIL import Image
from qrcode.constants import ERROR_CORRECT_M

ROOT = pathlib.Path(__file__).parent


def _detect_streamlit_port() -> str:
    raw = os.environ.get("STREAMLIT_SERVER_PORT") or os.environ.get("PORT")
    if raw and str(raw).isdigit():
        return str(raw)
    return "8501"


def _lan_ipv4() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.25)
        s.connect(("8.8.8.8", 53))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def _share_url() -> str:
    return f"http://{_lan_ipv4()}:{_detect_streamlit_port()}"


def _qr_png_bytes(url: str) -> bytes:
    qr = qrcode.QRCode(version=1, error_correction=ERROR_CORRECT_M, box_size=7, border=2)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="#0f172a", back_color="#ffffff").convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Ac-225 PINN | Nuclear Medicine AI",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Theme CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{--t:#0d9488;--d:#0f172a;--g:#10b981;--a:#f59e0b;--r:#ef4444;--cb:#f8fafc;--ce:#e2e8f0}
html,body,[class*="css"]{font-family:'Inter',sans-serif}
.main{padding:0}
.hero{background:linear-gradient(135deg,#0f172a 0%,#1e293b 40%,#0d9488 100%);color:#fff;padding:3rem 2.5rem 2.5rem;border-radius:0 0 24px 24px;margin:-1rem -1rem 2rem -1rem}
.hero h1{font-size:2.6rem;font-weight:800;margin:0 0 .3rem;letter-spacing:-.5px}
.hero .tl{font-size:1.15rem;opacity:.88;margin-bottom:1.2rem;line-height:1.5}
.hb{display:inline-block;background:rgba(255,255,255,.15);border:1px solid rgba(255,255,255,.25);border-radius:999px;padding:.3rem 1rem;font-size:.82rem;font-weight:600;margin-right:.5rem;margin-bottom:.4rem}
.sr{display:flex;gap:1rem;flex-wrap:wrap;margin:1.5rem 0}
.sc{flex:1 1 150px;background:var(--cb);border:1px solid var(--ce);border-radius:14px;padding:1.3rem;text-align:center;transition:box-shadow .2s}
.sc:hover{box-shadow:0 4px 20px rgba(13,148,136,.12)}
.sc .n{font-size:1.7rem;font-weight:800;color:var(--d)}
.sc .l{font-size:.78rem;font-weight:600;text-transform:uppercase;letter-spacing:.5px;color:#64748b;margin-top:.25rem}
.sh{font-size:1.45rem;font-weight:700;color:var(--d);border-left:4px solid var(--t);padding-left:.9rem;margin:2.2rem 0 1rem}
.tc{border-radius:12px;padding:1.2rem 1.4rem;margin-bottom:.8rem;border-left:5px solid}
.tp{background:#ecfdf5;border-color:var(--g)}.tw{background:#fffbeb;border-color:var(--a)}.tf{background:#fef2f2;border-color:var(--r)}
.tc h4{margin:0 0 .4rem;font-size:1rem}.tc p{margin:0;font-size:.88rem;color:#475569;line-height:1.5}
.ic{background:#fff;border:1px solid var(--ce);border-radius:14px;padding:1.5rem;height:100%}
.ic h4{color:var(--t);font-size:1.05rem;margin:.6rem 0 .5rem}.ic p{color:#475569;font-size:.88rem;line-height:1.55}
.ii{font-size:2rem}
.tt{width:100%;border-collapse:collapse;font-size:.88rem}
.tt th{text-align:left;padding:.6rem 1rem;background:#f1f5f9;font-weight:600;border-bottom:2px solid var(--ce)}
.tt td{padding:.6rem 1rem;border-bottom:1px solid var(--ce)}
.ft{text-align:center;padding:2rem 1rem;margin-top:3rem;border-top:1px solid var(--ce);color:#94a3b8;font-size:.82rem}
.ft a{color:var(--t);text-decoration:none}
.bug-card{background:#fff;border:1px solid var(--ce);border-radius:14px;padding:1.5rem;margin-bottom:1.2rem;border-left:5px solid var(--r)}
.bug-card.fixed{border-left-color:var(--g)}
.bug-card h4{margin:0 0 .4rem;font-size:1rem;color:var(--d)}
.bug-card .phase{display:inline-block;background:#f1f5f9;border-radius:6px;padding:.15rem .6rem;font-size:.72rem;font-weight:700;text-transform:uppercase;color:#64748b;margin-bottom:.5rem}
.bug-card p{margin:.3rem 0;font-size:.88rem;color:#475569;line-height:1.5}
.mistake-card{background:#fffbeb;border:1px solid #fde68a;border-radius:14px;padding:1.3rem 1.5rem;margin-bottom:1rem}
.mistake-card h4{margin:0 0 .3rem;font-size:.95rem;color:#92400e}
.mistake-card p{margin:0;font-size:.88rem;color:#78350f;line-height:1.5}
.future-card{background:linear-gradient(135deg,#f0fdfa 0%,#ecfeff 100%);border:1px solid #99f6e4;border-radius:14px;padding:1.5rem;height:100%}
.future-card h4{color:#0f766e;font-size:1.05rem;margin:.5rem 0}.future-card p{color:#475569;font-size:.88rem;line-height:1.55}
.gallery-row{display:flex;gap:1.5rem;margin:1.5rem 0;align-items:flex-start;flex-wrap:wrap}
.gallery-text{flex:1 1 300px}
.gallery-text h4{margin:0 0 .4rem;font-size:1.05rem;color:var(--d)}
.gallery-text p{font-size:.88rem;color:#475569;line-height:1.55;margin:0}
/* Mobile / narrow screens */
@media (max-width: 768px){
  .main .block-container{padding:1rem .75rem 2rem !important;max-width:100% !important}
  header[data-testid="stHeader"]{background:rgba(255,255,255,.96)}
  .hero{padding:1.35rem 1rem 1.5rem !important;margin:-.75rem -.75rem 1.25rem !important;border-radius:0 0 18px 18px !important}
  .hero h1{font-size:1.45rem !important;line-height:1.2 !important}
  .hero .tl{font-size:.95rem !important;margin-bottom:.9rem !important}
  .hb{font-size:.68rem !important;padding:.22rem .65rem !important;margin-right:.35rem !important}
  .sr{gap:.65rem !important;margin:1rem 0 !important}
  .sc{flex:1 1 calc(50% - .35rem) !important;min-width:120px !important;padding:1rem .75rem !important}
  .sc .n{font-size:1.25rem !important}
  .sc .l{font-size:.68rem !important}
  .sh{font-size:1.15rem !important;margin:1.35rem 0 .65rem !important;padding-left:.65rem !important}
  div[data-testid="stTabs"] button p{font-size:.78rem !important;line-height:1.25 !important}
  div[data-testid="stTabs"]{margin-bottom:.5rem}
  .stButton>button{min-height:2.75rem;font-size:1rem}
  .ic{padding:1.1rem !important}
}
@media (max-width: 420px){
  .hero h1{font-size:1.28rem !important}
  div[data-testid="stTabs"] button{padding:.35rem .45rem !important}
  div[data-testid="stTabs"] button p{font-size:.7rem !important}
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Dotted surface background (Canvas 2D animation, fixed behind content)
# ---------------------------------------------------------------------------
components.html("""
<canvas id="dotbg" style="position:fixed;top:0;left:0;width:100vw;height:100vh;z-index:-1;pointer-events:none;opacity:0.12"></canvas>
<script>
(function(){
var c=document.getElementById("dotbg"),g=c.getContext("2d");
var W,H,cols,rows,sep=28,cnt=0;
function sz(){W=window.innerWidth;H=window.innerHeight;c.width=W;c.height=H;cols=Math.ceil(W/sep)+2;rows=Math.ceil(H/sep)+2}
sz();window.addEventListener("resize",sz);
function draw(){
  g.clearRect(0,0,W,H);
  for(var ix=0;ix<cols;ix++){for(var iy=0;iy<rows;iy++){
    var x=ix*sep,y=iy*sep;
    var wave=Math.sin((ix+cnt)*0.15)*4+Math.sin((iy+cnt)*0.2)*4;
    var r=1.2+Math.abs(wave)*0.3;
    var alpha=0.25+Math.abs(wave)*0.06;
    g.fillStyle="rgba(13,148,136,"+alpha+")";
    g.beginPath();g.arc(x,y+wave,r,0,Math.PI*2);g.fill();
  }}
  cnt+=0.04;
  requestAnimationFrame(draw);
}
draw();
})();
</script>
""", height=0, scrolling=False)

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
<h1>⚛️ Ac-225 Production Intelligence</h1>
<p class="tl">
A physics-informed neural network that predicts <b>Actinium-225</b> yields
from the <b>Ra-226 &rarr; Ra-225 &rarr; Ac-225</b> transmutation chain &mdash;
enabling faster, smarter decisions for <b>targeted alpha therapy</b> in cancer treatment.
</p>
<span class="hb">Physics-Informed</span>
<span class="hb">12,000 Epochs</span>
<span class="hb">Mass-Conserving</span>
<span class="hb">NNDC Nuclear Data</span>
<span class="hb">CPU Real-Time</span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="sr">
<div class="sc"><div class="n">51,819</div><div class="l">Parameters</div></div>
<div class="sc"><div class="n">&lt; 2 ms</div><div class="l">Inference</div></div>
<div class="sc"><div class="n">12,000</div><div class="l">Epochs</div></div>
<div class="sc"><div class="n">7,500</div><div class="l">Samples</div></div>
<div class="sc"><div class="n">3 / 3</div><div class="l">Trio Pass</div></div>
</div>
""", unsafe_allow_html=True)

_share = _share_url()
with st.expander("📱 Open on another device (QR) — same Wi‑Fi", expanded=False):
    st.markdown(
        "Scan with a phone camera. **This PC and the phone must be on the same Wi‑Fi.** "
        "The server uses `.streamlit/config.toml` (listens on all interfaces). "
        "If the page does not open, allow **TCP port 8501** in Windows Firewall."
    )
    try:
        st.image(_qr_png_bytes(_share), caption=f"Scan → {_share}", use_container_width=False, width=260)
    except Exception as exc:  # pragma: no cover
        st.warning(f"Install: `pip install 'qrcode[pil]'` — {exc}")
    st.code(_share, language="text")
    st.caption("Tip on small phones: use landscape for the Project Timeline orbit. Copy the link to text it to someone nearby.")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
(tab_why, tab_demo, tab_train_live, tab_speed, tab_dose, tab_timeline, tab_results, tab_gallery,
 tab_trio, tab_struggles, tab_mistakes, tab_future, tab_tech) = st.tabs([
    "🏥 Why This Matters",
    "🔬 Live Prediction",
    "🧠 Live Training",
    "⚡ Speed Benchmark",
    "💊 Dose Calculator",
    "🪐 Project Timeline",
    "📊 Training Results",
    "🖼️ Plot Gallery",
    "✅ Trio Validation",
    "🔥 Struggles & Failures",
    "💡 Mistakes We Made",
    "🚀 Future Applications",
    "⚙️ Technical Details",
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 -- WHY THIS MATTERS
# ═══════════════════════════════════════════════════════════════════════════
with tab_why:
    st.markdown('<div class="sh">The Ac-225 Supply Crisis</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("""
**Actinium-225** is one of the most promising isotopes for **targeted alpha therapy (TAT)**
-- a next-generation cancer treatment that delivers lethal radiation directly to tumor cells
while sparing healthy tissue.

- **4 alpha decays** in the Ac-225 chain deliver ~28 MeV within a few cell diameters
- Clinical trials show responses in **metastatic prostate cancer**, **leukemia**, and **neuroendocrine tumors**
- Global demand: **50--100 Ci/year** projected by 2030; current supply: **< 3 Ci/year**

The bottleneck is **production planning**. Each irradiation campaign costs hundreds of thousands
of dollars. Getting the flux, time, and target composition wrong means wasted beam time and missed patient doses.
        """)
    with c2:
        st.markdown("""
<div class="ic"><div class="ii">🎯</div><h4>Targeted Alpha Therapy</h4>
<p>Ac-225 labeled antibodies seek out cancer cells and destroy them with short-range alpha particles. Unlike chemo, healthy cells are largely unaffected.</p></div>
        """, unsafe_allow_html=True)
        st.markdown("""
<div class="ic" style="margin-top:.8rem"><div class="ii">⚡</div><h4>Where AI Helps</h4>
<p>This PINN replaces minutes-long ODE solves with <b>millisecond</b> predictions, letting engineers sweep thousands of production scenarios in real time.</p></div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="sh">How It Works</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="ic"><div class="ii">📐</div><h4>1. Physics Foundation</h4><p>The Bateman decay equations govern how Ra-226 transmutes to Ra-225 under neutron flux, and how Ra-225 beta-decays to Ac-225. These ODEs are the ground truth.</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="ic"><div class="ii">🧠</div><h4>2. Neural Network</h4><p>A 4-layer MLP is trained on 7,500 ODE trajectories <i>and</i> physics residuals simultaneously. It learns the dynamics, not just data patterns.</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="ic"><div class="ii">🛡️</div><h4>3. Safety Constraints</h4><p>Hard mass-budget cap prevents "alchemy." Non-negativity, secular equilibrium ceiling, and empty-tank penalties ensure physically valid outputs.</p></div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 -- LIVE PREDICTION
# ═══════════════════════════════════════════════════════════════════════════
with tab_demo:
    st.markdown('<div class="sh">Interactive PINN Prediction</div>', unsafe_allow_html=True)
    st.markdown("Adjust the sliders and see predicted isotope inventories in real time.")

    _cal_w = ROOT / "pinn_calibrated_weights.pth"
    WEIGHTS_PATH = _cal_w if _cal_w.is_file() else ROOT / "pinn_trained_weights.pth"
    if _cal_w.is_file():
        st.caption("Using **calibrated** weights (from Reactor Calibration).")
    model_loaded = False
    model = None
    if WEIGHTS_PATH.is_file():
        try:
            from pinn_model import IsotopePINN, neutron_energy_ev_to_feature_numpy
            model = IsotopePINN()
            state = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.eval()
            model_loaded = True
        except Exception as e:
            st.error(f"Could not load model: {e}")
    else:
        st.warning("No trained weights found. Run `python train.py` first.")

    ci, co = st.columns([1, 2])
    with ci:
        st.markdown("#### Irradiation Parameters")
        flux_exp = st.slider("Flux (log10 n/cm^2/s)", 12.0, 15.5, 14.0, 0.1)
        flux = 10.0 ** flux_exp
        st.caption(f"phi = {flux:.2e} n/cm^2/s")
        time_h = st.slider("Irradiation time (hours)", 1.0, 500.0, 200.0, 5.0)
        energy_ev = st.slider("Neutron energy (eV)", 0.01, 1.0, 0.025, 0.005)
        st.markdown("#### Initial Inventories")
        ra226_0 = st.number_input("Ra-226 (atoms)", value=6.022e23, format="%.3e")
        ra225_0 = st.number_input("Ra-225 (atoms)", value=0.0, format="%.3e")
        ac225_0 = st.number_input("Ac-225 (atoms)", value=0.0, format="%.3e")

    with co:
        if model_loaded and model is not None:
            N226S, N225S, NACS, PHIS, TSH = 6.022e23, 1e20, 1e20, 1e15, 500.0
            e_nn = float(neutron_energy_ev_to_feature_numpy(energy_ev))
            x = torch.tensor([[time_h/TSH, flux/PHIS, e_nn, ra226_0/N226S, ra225_0/N225S, ac225_0/NACS]], dtype=torch.float32)
            with torch.no_grad():
                pred = model(x)
            p226, p225, pac = float(pred[0,0]*N226S), float(pred[0,1]*N225S), float(pred[0,2]*NACS)
            st.markdown("#### Predicted Inventories")
            m1, m2, m3 = st.columns(3)
            m1.metric("Ra-226", f"{p226:.4e}", f"{p226-ra226_0:.2e}")
            m2.metric("Ra-225", f"{p225:.4e}", f"{p225-ra225_0:.2e}")
            m3.metric("Ac-225", f"{pac:.4e}", f"{pac-ac225_0:.2e}")
            tot0 = ra226_0 + ra225_0 + ac225_0
            totp = p226 + p225 + pac
            if tot0 > 0:
                st.progress(min(totp / tot0, 1.0))
                st.caption(f"Mass conservation: {totp/tot0*100:.4f}%")
            st.markdown("#### Production Curve")
            times = np.linspace(1.0, float(time_h), 80)
            ac_c = []
            for t in times:
                xt = torch.tensor([[t/TSH, flux/PHIS, e_nn, ra226_0/N226S, ra225_0/N225S, ac225_0/NACS]], dtype=torch.float32)
                with torch.no_grad():
                    p = model(xt)
                ac_c.append(float(p[0,2]*NACS))
            st.line_chart(pd.DataFrame({"Time (h)": times, "Ac-225 (atoms)": ac_c}), x="Time (h)", y="Ac-225 (atoms)", color="#0d9488")
        else:
            st.info("Load trained weights to enable live predictions.")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 -- LIVE TRAINING
# ═══════════════════════════════════════════════════════════════════════════
with tab_train_live:
    st.markdown('<div class="sh">Live Training & Reactor Calibration</div>', unsafe_allow_html=True)

    _train_sub = st.radio(
        "Select mode:",
        ["Physics Fine-Tune (demo)", "Reactor Calibration (upload CSV)"],
        horizontal=True, key="train_sub",
    )

    _N226S, _N225S, _NACS, _PHIS, _TSH = 6.022e23, 1e20, 1e20, 1e15, 500.0
    _wp = ROOT / "pinn_trained_weights.pth"
    _cal_path = ROOT / "pinn_calibrated_weights.pth"

    def _run_finetune(mod, opt, inputs_t, targets_t, n_epochs, container, has_targets=False):
        """Shared training loop with live chart updates."""
        import time as _time
        import math as _math
        from pinn_model import compute_physics_loss as _cpl
        from torch.nn.utils import clip_grad_norm_ as _clip
        losses, phys_l, data_l, mass_l, neg_l = [], [], [], [], []
        prog = container.progress(0)
        chart_ph = container.empty()
        metric_ph = container.empty()
        status_ph = container.empty()
        update_every = max(1, min(3, n_epochs // 50))
        t0 = _time.perf_counter()
        for ep in range(1, n_epochs + 1):
            mod.train()
            opt.zero_grad(set_to_none=True)
            pred = mod.forward_raw(inputs_t)
            tgt = targets_t if has_targets else None
            pred_cap = mod(inputs_t) if has_targets else None
            loss, info = _cpl(
                mod, inputs_t, pred, targets=tgt,
                physics_weight=1000.0, data_weight=80.0 if has_targets else 0.0,
                mass_weight=500.0, fuel_anchor_weight=20.0,
                non_neg_weight=50.0, secular_eq_weight=25.0, ra225_physics_weight=5.0,
                pred_for_data=pred_cap,
                data_species_weights=(1.0, 80.0, 80.0),
                n226_scale=_N226S, n225_scale=_N225S, nac_scale=_NACS,
                phi_scale=_PHIS, d_t_input_d_t_hours=1.0/_TSH,
                use_one_over_v_energy=True,
            )
            if torch.isfinite(loss):
                loss.backward()
                _clip(mod.parameters(), 5.0)
                opt.step()
            lv = float(loss.detach().cpu()) if torch.isfinite(loss) else (losses[-1] if losses else 0)
            pv = float(info["physics_mse"].detach().cpu())
            dv = float(info["data_mse"].detach().cpu())
            mv = float(info["mass_cons_loss"].detach().cpu())
            nv = float(info["non_neg_loss"].detach().cpu())
            losses.append(lv)
            phys_l.append(pv)
            data_l.append(dv)
            mass_l.append(mv)
            neg_l.append(nv)
            prog.progress(ep / n_epochs)

            if ep % update_every == 0 or ep == 1 or ep == n_epochs:
                log_total = [_math.log10(max(v, 1e-20)) for v in losses]
                log_phys = [_math.log10(max(v, 1e-20)) for v in phys_l]
                cols = {"Epoch": list(range(1, len(losses)+1)),
                        "Total Loss (log10)": log_total,
                        "Physics Loss (log10)": log_phys}
                if has_targets:
                    cols["Data Loss (log10)"] = [_math.log10(max(v, 1e-20)) for v in data_l]
                chart_ph.line_chart(
                    pd.DataFrame(cols), x="Epoch",
                    y=["Total Loss (log10)", "Physics Loss (log10)"] + (["Data Loss (log10)"] if has_targets else []),
                    color=["#0ea5e9", "#f59e0b"] + (["#10b981"] if has_targets else []),
                )
                el = _time.perf_counter() - t0
                metric_ph.markdown(
                    f"**Epoch {ep}/{n_epochs}** | Total: **{lv:.2f}** | Physics: {pv:.6f} | "
                    f"Mass: {mv:.2e} | Non-neg: {nv:.2e}"
                    + (f" | Data: {dv:.6f}" if has_targets else "")
                    + f" | {el:.1f}s"
                )
                if ep > 1 and len(losses) > 2:
                    drop = (losses[0] - losses[-1]) / max(losses[0], 1e-10) * 100
                    status_ph.caption(f"Loss reduced by {drop:.1f}% from epoch 1")
        return losses

    # ---- MODE A: Physics fine-tune (demo) ----
    if "Physics" in _train_sub:
        st.markdown("""
Watch the PINN improve on physics constraints **live in your browser**. The model loads the
pre-trained 12k-epoch weights, freezes the physics layers, and fine-tunes the output head.
You'll see the loss curve drop in real time.
        """)
        ft_ep = st.slider("Epochs", 10, 500, 100, 10, key="pft_ep")
        ft_lr = st.select_slider("Learning rate", [1e-5, 5e-5, 1e-4, 5e-4, 1e-3], value=1e-4, key="pft_lr")
        ft_go = st.button("Start Physics Training", type="primary", key="pft_go")

        if ft_go and _wp.is_file():
            from pinn_model import IsotopePINN as _PI, neutron_energy_ev_to_feature_torch as _eft
            from torch.optim import Adam as _Adam
            _mod = _PI(); _mod.load_state_dict(torch.load(_wp, map_location="cpu", weights_only=True))
            for n, p in _mod.named_parameters():
                if "hidden.0" in n or "hidden.1" in n or "hidden.2" in n or "hidden.3" in n:
                    p.requires_grad = False
            trainable = sum(p.numel() for p in _mod.parameters() if p.requires_grad)
            st.caption(f"Frozen early layers. Training {trainable:,} parameters (head + last layers).")
            _opt = _Adam(filter(lambda p: p.requires_grad, _mod.parameters()), lr=float(ft_lr))
            _rng = np.random.default_rng(77)
            _nc = 400
            _th = torch.tensor(_rng.uniform(0.1, 500, _nc), dtype=torch.float32)
            _ph = torch.tensor(10.0**_rng.uniform(13, 15, _nc), dtype=torch.float32)
            _er = torch.tensor(_rng.uniform(0.015, 0.08, _nc), dtype=torch.float32)
            _en = _eft(_er)
            _u = _rng.uniform(0, 1, (_nc, 3)).astype(np.float32)
            _coll = torch.cat([
                (_th/_TSH).unsqueeze(1), (_ph/_PHIS).unsqueeze(1), _en.unsqueeze(1),
                torch.tensor(np.exp(np.log(1.0)+_u[:,0]*(np.log(_N226S*1.1)-np.log(1.0))),dtype=torch.float32).unsqueeze(1)/_N226S,
                torch.tensor(np.exp(np.log(1.0)+_u[:,1]*(np.log(_N225S*10)-np.log(1.0))),dtype=torch.float32).unsqueeze(1)/_N225S,
                torch.tensor(np.exp(np.log(1.0)+_u[:,2]*(np.log(_NACS*10)-np.log(1.0))),dtype=torch.float32).unsqueeze(1)/_NACS,
            ], dim=1).detach()
            _ei = torch.randperm(_nc)[:int(_nc*0.3)]
            _coll[_ei, 3:6] = 0.0; _coll[_ei, 1] = 1.0
            _coll = _coll.requires_grad_(True)
            _train_container = st.container()
            losses = _run_finetune(_mod, _opt, _coll, None, ft_ep, _train_container, has_targets=False)
            st.success(f"Training complete! {ft_ep} epochs. Final loss: {losses[-1]:.4f}")
            torch.save(_mod.state_dict(), ROOT / "pinn_finetuned_weights.pth")
            st.caption("Saved to `pinn_finetuned_weights.pth`")
        elif ft_go:
            st.error("No base weights found. Run `python train.py` first.")

    # ---- MODE B: Reactor calibration (upload CSV) ----
    else:
        st.markdown("""
**Upload reactor measurement data** and the model will calibrate itself to your facility in under a minute.
The base 12k-epoch physics knowledge is preserved; only the last layers adapt to your data.
        """)

        st.markdown("""
<div class="ic">
<h4>Required CSV Format</h4>
<p>Columns (exact names): <code>time</code>, <code>flux</code>, <code>energy</code>,
<code>N_Ra226</code>, <code>N_Ra225</code>, <code>N_Ac225</code><br>
<b>time</b> in hours, <b>flux</b> in n/cm<sup>2</sup>/s, <b>energy</b> in eV,
<b>N_*</b> in atoms (measured inventory at end of irradiation).<br>
Initial condition is assumed to be (6.022e23, 0, 0) unless you add <code>init_N226</code>,
<code>init_N225</code>, <code>init_NAc</code> columns.</p>
</div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader("Upload reactor CSV", type=["csv"], key="cal_csv")
        cal_ep = st.slider("Calibration epochs", 50, 500, 300, 25, key="cal_ep")
        cal_lr = st.select_slider("Learning rate", [1e-5, 5e-5, 1e-4, 5e-4], value=5e-5, key="cal_lr")
        cal_go = st.button("Calibrate Model", type="primary", key="cal_go", disabled=uploaded is None)

        if cal_go and uploaded is not None and _wp.is_file():
            from pinn_model import IsotopePINN as _PI, neutron_energy_ev_to_feature_torch as _eft
            from torch.optim import Adam as _Adam

            # --- CSV auto-fix preprocessor ---
            try:
                df_cal = pd.read_csv(uploaded)
            except Exception:
                uploaded.seek(0)
                try:
                    df_cal = pd.read_csv(uploaded, sep=None, engine="python")
                except Exception as _e:
                    st.error(f"Could not parse CSV: {_e}")
                    df_cal = None

            _fixes_applied = []
            if df_cal is not None:
                df_cal.columns = df_cal.columns.str.strip()

                _col_map = {
                    "time": ["time", "time_h", "time_hr", "time_hours", "hours", "t", "irradiation_time", "t_h"],
                    "flux": ["flux", "phi", "neutron_flux", "flux_phi", "n_flux", "phi_n"],
                    "energy": ["energy", "energy_ev", "e", "ev", "neutron_energy", "e_ev"],
                    "N_Ra226": ["n_ra226", "ra226", "ra-226", "n226", "nra226", "ra_226", "n_ra_226"],
                    "N_Ra225": ["n_ra225", "ra225", "ra-225", "n225", "nra225", "ra_225", "n_ra_225"],
                    "N_Ac225": ["n_ac225", "ac225", "ac-225", "nac225", "nac", "ac_225", "n_ac_225", "actinium"],
                    "init_N226": ["init_n226", "init_ra226", "initial_ra226", "ra226_0", "n226_0", "init_n_ra226"],
                    "init_N225": ["init_n225", "init_ra225", "initial_ra225", "ra225_0", "n225_0", "init_n_ra225"],
                    "init_NAc": ["init_nac", "init_ac225", "initial_ac225", "ac225_0", "nac_0", "init_n_ac225"],
                }
                existing_lower = {c.lower(): c for c in df_cal.columns}
                renames = {}
                for target, aliases in _col_map.items():
                    if target in df_cal.columns:
                        continue
                    for alias in aliases:
                        if alias.lower() in existing_lower:
                            renames[existing_lower[alias.lower()]] = target
                            break
                if renames:
                    df_cal = df_cal.rename(columns=renames)
                    _fixes_applied.append(f"Renamed columns: {renames}")

                req = {"time", "flux", "energy", "N_Ra226", "N_Ra225", "N_Ac225"}
                missing = req - set(df_cal.columns)

                if "energy" in missing and "energy" not in df_cal.columns:
                    df_cal["energy"] = 0.025
                    missing.discard("energy")
                    _fixes_applied.append("Added default energy = 0.025 eV (thermal)")

                if "flux" not in missing and df_cal["flux"].max() < 1e6:
                    _fixes_applied.append(
                        f"WARNING: max flux = {df_cal['flux'].max():.2e}. "
                        f"Expected n/cm^2/s (typically 1e13-1e15). Check units."
                    )

                if "time" not in missing and df_cal["time"].max() > 50000:
                    _fixes_applied.append(
                        f"WARNING: max time = {df_cal['time'].max():.0f}. "
                        f"Expected hours (not seconds). If in seconds, divide by 3600."
                    )

                for col in ["N_Ra226", "N_Ra225", "N_Ac225"]:
                    if col in df_cal.columns:
                        neg_count = (df_cal[col] < 0).sum()
                        if neg_count > 0:
                            df_cal[col] = df_cal[col].clip(lower=0)
                            _fixes_applied.append(f"Clipped {neg_count} negative values in {col} to 0")

                for col in df_cal.columns:
                    if df_cal[col].isna().any():
                        n_na = int(df_cal[col].isna().sum())
                        if col == "energy":
                            df_cal[col] = df_cal[col].fillna(0.025)
                        elif col in ("init_N226", "init_N225", "init_NAc"):
                            df_cal[col] = df_cal[col].fillna(0.0)
                        else:
                            df_cal = df_cal.dropna(subset=[col])
                        _fixes_applied.append(f"Fixed {n_na} missing values in {col}")

                df_cal = df_cal[df_cal.select_dtypes(include=[np.number]).columns.tolist()].dropna()

            if df_cal is not None and _fixes_applied:
                with st.expander("Auto-fixes applied (click to review)", expanded=True):
                    for fix in _fixes_applied:
                        if "WARNING" in fix:
                            st.warning(fix)
                        else:
                            st.caption(f"  {fix}")

            req = {"time", "flux", "energy", "N_Ra226", "N_Ra225", "N_Ac225"}
            if df_cal is None:
                pass
            elif not req.issubset(df_cal.columns):
                missing = req - set(df_cal.columns)
                st.error(
                    f"Still missing required columns after auto-fix: **{missing}**\n\n"
                    f"Found: {sorted(df_cal.columns.tolist())}\n\n"
                    f"Need: `time`, `flux`, `energy`, `N_Ra226`, `N_Ra225`, `N_Ac225`"
                )
            elif len(df_cal) == 0:
                st.error("CSV has no valid rows after cleanup.")
            else:
                st.markdown(f"**Loaded {len(df_cal)} rows** from uploaded CSV.")

                i226 = df_cal["init_N226"].values if "init_N226" in df_cal.columns else np.full(len(df_cal), _N226S)
                i225 = df_cal["init_N225"].values if "init_N225" in df_cal.columns else np.zeros(len(df_cal))
                iac = df_cal["init_NAc"].values if "init_NAc" in df_cal.columns else np.zeros(len(df_cal))

                _e_nn = _eft(torch.tensor(df_cal["energy"].values, dtype=torch.float32))
                _inp = torch.cat([
                    torch.tensor(df_cal["time"].values / _TSH, dtype=torch.float32).unsqueeze(1),
                    torch.tensor(df_cal["flux"].values / _PHIS, dtype=torch.float32).unsqueeze(1),
                    _e_nn.unsqueeze(1),
                    torch.tensor(i226 / _N226S, dtype=torch.float32).unsqueeze(1),
                    torch.tensor(i225 / _N225S, dtype=torch.float32).unsqueeze(1),
                    torch.tensor(iac / _NACS, dtype=torch.float32).unsqueeze(1),
                ], dim=1).detach().requires_grad_(True)
                _tgt = torch.tensor(
                    df_cal[["N_Ra226", "N_Ra225", "N_Ac225"]].values,
                    dtype=torch.float32,
                )

                _mod = _PI()
                _mod.load_state_dict(torch.load(_wp, map_location="cpu", weights_only=True))
                for n, p in _mod.named_parameters():
                    if any(f"hidden.{i}" in n for i in range(6)):
                        p.requires_grad = False
                trainable = sum(p.numel() for p in _mod.parameters() if p.requires_grad)
                total_p = sum(p.numel() for p in _mod.parameters())
                st.caption(f"Frozen {total_p - trainable:,} / {total_p:,} params. Training {trainable:,} (head + last layer).")

                _opt = _Adam(filter(lambda p: p.requires_grad, _mod.parameters()), lr=float(cal_lr))

                st.markdown("#### Before Calibration")
                _mod.eval()
                with torch.no_grad():
                    _pb = _mod(_inp.detach())
                _ac_before = (_pb[:, 2] * _NACS).cpu().numpy()
                _ac_true = _tgt[:, 2].cpu().numpy()
                _mask = _ac_true > 0
                if _mask.any():
                    _mape_before = float(np.mean(np.abs(_ac_before[_mask] - _ac_true[_mask]) / _ac_true[_mask]) * 100)
                    st.metric("Ac-225 MAPE (before)", f"{_mape_before:.1f}%")

                st.markdown("#### Calibrating...")
                _cal_container = st.container()
                losses = _run_finetune(_mod, _opt, _inp, _tgt, cal_ep, _cal_container, has_targets=True)

                _mod.eval()
                with torch.no_grad():
                    _pa = _mod(_inp.detach())
                _ac_after = (_pa[:, 2] * _NACS).cpu().numpy()
                if _mask.any():
                    _mape_after = float(np.mean(np.abs(_ac_after[_mask] - _ac_true[_mask]) / _ac_true[_mask]) * 100)

                st.success(f"Calibration complete! {cal_ep} epochs. Final loss: {losses[-1]:.4f}")

                if _mask.any():
                    m1, m2, m3 = st.columns(3)
                    m1.metric("MAPE Before", f"{_mape_before:.1f}%")
                    m2.metric("MAPE After", f"{_mape_after:.1f}%", f"{_mape_after - _mape_before:+.1f}%")
                    m3.metric("Improvement", f"{_mape_before - _mape_after:.1f}%", "reduction")

                torch.save(_mod.state_dict(), _cal_path)
                st.caption(f"Calibrated weights saved to `{_cal_path.name}`. The Live Prediction tab will use these automatically.")

                st.markdown("#### Per-Sample Comparison")
                _comp = pd.DataFrame({
                    "Time (h)": df_cal["time"].values,
                    "Measured Ac-225": _ac_true,
                    "Before Cal.": _ac_before,
                    "After Cal.": _ac_after,
                })
                st.dataframe(_comp.style.format("{:.4e}"), use_container_width=True)

        elif cal_go and uploaded is None:
            st.warning("Upload a CSV first.")
        elif cal_go:
            st.error("No base weights found.")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 -- SPEED BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════
with tab_speed:
    st.markdown('<div class="sh">PINN vs ODE: Speed Comparison</div>', unsafe_allow_html=True)
    st.markdown("How much faster is the trained neural network compared to solving the differential equations from scratch?")

    _bench_default = 1000
    _bench_raw = None
    try:
        if hasattr(st, "secrets") and "PINN_BENCH_SCENARIOS" in st.secrets:
            _bench_raw = st.secrets["PINN_BENCH_SCENARIOS"]
    except Exception:
        _bench_raw = None
    if _bench_raw is None:
        _bench_raw = os.environ.get("PINN_BENCH_SCENARIOS", str(_bench_default))
    try:
        _bench_n = int(_bench_raw)
    except (TypeError, ValueError):
        _bench_n = _bench_default
    _bench_n = int(np.clip(_bench_n, 50, 5000))
    st.caption(
        f"Benchmark size: **{_bench_n}** scenarios "
        f"(Streamlit **Secrets** key `PINN_BENCH_SCENARIOS`, or env var; default 1000)."
    )

    if st.button(f"Run Benchmark ({_bench_n} scenarios)", type="primary"):
        import time as _time
        from ra226_ac225_transmutation import IsotopeEnvironment, run_simulation
        from pinn_model import IsotopePINN as _PINN, neutron_energy_ev_to_feature_numpy as _efn

        _rng = np.random.default_rng(99)
        N_BENCH = _bench_n
        phis = 10.0 ** _rng.uniform(13, 15, N_BENCH)
        times_h = _rng.uniform(10, 500, N_BENCH)
        energies = _rng.uniform(0.02, 0.05, N_BENCH)

        _N226S, _N225S, _NACS, _PHIS, _TSH = 6.022e23, 1e20, 1e20, 1e15, 500.0

        st.markdown(f"**Running ODE solver** (scipy odeint) on {N_BENCH} scenarios...")
        prog = st.progress(0)
        t0_ode = _time.perf_counter()
        _log_every = max(1, N_BENCH // 10)
        for i in range(N_BENCH):
            env = IsotopeEnvironment(phi=float(phis[i]), sigma_ra226=1e-24, neutron_energy_ev=float(energies[i]))
            run_simulation(env, t_end_h=float(times_h[i]), n_points=201, N_ra0=_N226S)
            if i % _log_every == 0:
                prog.progress(i / N_BENCH)
        t_ode = _time.perf_counter() - t0_ode
        prog.progress(1.0)

        st.markdown(f"**Running PINN** on {N_BENCH} scenarios...")
        _wp = ROOT / "pinn_trained_weights.pth"
        if _wp.is_file():
            _m = _PINN()
            _m.load_state_dict(torch.load(_wp, map_location="cpu", weights_only=True))
            _m.eval()
            rows = []
            for i in range(N_BENCH):
                rows.append([float(times_h[i])/_TSH, float(phis[i])/_PHIS,
                    float(_efn(energies[i])), 1.0, 0.0, 0.0])
            _xb = torch.tensor(rows, dtype=torch.float32)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t0_pinn = _time.perf_counter()
            with torch.no_grad():
                _m(_xb)
            t_pinn = _time.perf_counter() - t0_pinn

            speedup = t_ode / max(t_pinn, 1e-9)
            ode_per = t_ode / N_BENCH * 1000
            pinn_per = t_pinn / N_BENCH * 1000

            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            c1.metric("ODE Total", f"{t_ode:.2f} s", f"{ode_per:.2f} ms/sample")
            c2.metric("PINN Total", f"{t_pinn*1000:.1f} ms", f"{pinn_per:.4f} ms/sample")
            c3.metric("Speedup", f"{speedup:.0f}x", "faster with PINN")

            bench_df = pd.DataFrame({
                "Method": ["ODE (scipy)", "PINN (PyTorch)"],
                "Time per sample (ms)": [ode_per, pinn_per],
            })
            st.bar_chart(bench_df, x="Method", y="Time per sample (ms)", color="#0d9488", horizontal=True)

            st.markdown(f"""
**Result**: The PINN is **{speedup:.0f}x faster** than the ODE solver.

- **ODE**: {ode_per:.2f} ms per scenario (sequential scipy integration)
- **PINN**: {pinn_per:.4f} ms per scenario (batched neural network forward pass)

This means an engineer can sweep **{int(1000/max(t_pinn,1e-9)):,} scenarios per second** with the PINN,
enabling real-time production optimization that is impossible with traditional solvers.
            """)
        else:
            st.warning("No weights found. Train first.")
    else:
        st.info("Click the button above to run a live speed comparison. It takes about 30 seconds.")
        st.markdown("""
**Why this matters:**

Traditional isotope production planning uses codes like **ORIGEN** that solve ODEs for each scenario.
If you need to sweep 10,000 combinations of flux, time, and energy to find the optimal schedule,
that takes **minutes to hours**. The PINN does it in **milliseconds** -- enabling real-time
optimization and interactive dashboards like this one.
        """)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 -- DOSE CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════
with tab_dose:
    st.markdown('<div class="sh">Clinical Dose Estimation</div>', unsafe_allow_html=True)
    st.markdown("Convert predicted Ac-225 atom counts into clinical activity and patient doses.")

    import math as _math
    _LN2 = _math.log(2.0)
    _LAMBDA_AC_S = _LN2 / (9.920 * 24 * 3600)
    _AVOGADRO = 6.022e23
    _CI_PER_BQ = 1.0 / 3.7e10

    st.markdown("""
<div class="ic">
<h4>The Physics</h4>
<p>
Activity (Becquerels) = decay constant (lambda) x number of atoms<br>
lambda(Ac-225) = ln(2) / half-life = ln(2) / (9.920 days) = <b>8.09 x 10<sup>-7</sup> /s</b><br>
1 Curie = 3.7 x 10<sup>10</sup> Bq<br>
Typical patient dose for TAT: <b>50-200 kBq/kg</b> (about <b>100-200 microCi</b> total per treatment)
</p>
</div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    dc1, dc2 = st.columns([1, 2])
    with dc1:
        st.markdown("#### Input")
        ac_atoms = st.number_input("Ac-225 atoms produced", value=1.47e17, format="%.3e",
                                     help="From your PINN prediction or ODE output")
        patient_weight_kg = st.slider("Patient weight (kg)", 40, 120, 70, 5)
        dose_per_kg = st.slider("Dose (kBq/kg)", 10, 300, 100, 10,
                                  help="Typical TAT: 50-200 kBq/kg")

    with dc2:
        activity_bq = _LAMBDA_AC_S * ac_atoms
        activity_ci = activity_bq * _CI_PER_BQ
        activity_mci = activity_ci * 1e3
        activity_uci = activity_ci * 1e6

        total_dose_needed_bq = dose_per_kg * 1000 * patient_weight_kg
        total_dose_needed_atoms = total_dose_needed_bq / _LAMBDA_AC_S
        n_patients = ac_atoms / max(total_dose_needed_atoms, 1)

        st.markdown("#### Produced Activity")
        m1, m2, m3 = st.columns(3)
        m1.metric("Becquerels", f"{activity_bq:.3e} Bq")
        m2.metric("Millicuries", f"{activity_mci:.2f} mCi")
        m3.metric("Microcuries", f"{activity_uci:.0f} uCi")

        st.markdown("#### Patient Doses")
        m4, m5 = st.columns(2)
        m4.metric("Dose per patient", f"{dose_per_kg * patient_weight_kg / 1000:.1f} MBq",
                   f"{dose_per_kg} kBq/kg x {patient_weight_kg} kg")
        m5.metric("Patients treatable", f"{n_patients:.1f}",
                   f"from {ac_atoms:.2e} atoms")

        if n_patients >= 1:
            st.success(f"This production run yields enough Ac-225 for **{int(n_patients)} patient treatments**.")
        else:
            pct = n_patients * 100
            st.warning(f"This yields **{pct:.1f}%** of one patient dose. Increase flux or irradiation time.")

        st.markdown("#### Scale Perspective")
        st.markdown(f"""
| Metric | Value |
|--------|-------|
| Atoms produced | {ac_atoms:.3e} |
| Activity | {activity_mci:.2f} mCi ({activity_bq:.3e} Bq) |
| Per patient need | {total_dose_needed_atoms:.3e} atoms ({total_dose_needed_bq/1e6:.1f} MBq) |
| **Patients served** | **{n_patients:.1f}** |
| Ac-225 mass | {ac_atoms * 225 / _AVOGADRO * 1e9:.4f} nanograms |

*Note: Ac-225 is so potent that a few nanograms can treat multiple patients.
This is why production optimization matters -- small improvements in yield translate directly to lives saved.*
        """)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 6 -- PROJECT TIMELINE (ORBITAL)
# ═══════════════════════════════════════════════════════════════════════════
with tab_timeline:
    st.markdown('<div class="sh">Project Development Timeline</div>', unsafe_allow_html=True)
    st.markdown(
        "Click a node to pause rotation — the orbit **stays put** (nothing jumps to the top). "
        "Details open in the **right panel** so you can still reach every node. Click empty space to spin again."
    )

    ORBITAL_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=5, viewport-fit=cover">
<style>
*{margin:0;padding:0;box-sizing:border-box}
body,html{background:#000;overflow:hidden;font-family:Inter,-apple-system,BlinkMacSystemFont,sans-serif;color:#fff}
#w{width:100%;height:650px;display:flex;flex-direction:row;background:radial-gradient(ellipse 80% 70% at 40% 45%,#0f172a 0%,#020617 55%,#000 100%);position:relative;overflow:hidden}
.cvWrap{flex:1;position:relative;min-width:180px;min-height:0}
canvas{display:block;width:100%;height:100%;touch-action:manipulation}
#dock{width:min(300px,34vw);flex-shrink:0;border-left:1px solid rgba(148,163,184,.12);
  background:linear-gradient(180deg,rgba(15,23,42,.95) 0%,rgba(2,6,23,.98) 100%);
  display:flex;flex-direction:column;align-items:stretch;padding:16px 14px;overflow-y:auto;overflow-x:hidden}
#dockPh{flex:1;display:flex;align-items:center;justify-content:center;text-align:center;padding:12px;
  font-size:12px;color:rgba(148,163,184,.75);line-height:1.5;border:1px dashed rgba(148,163,184,.2);
  border-radius:12px;margin-top:4px}
#card{display:none;background:rgba(15,23,42,.88);border:1px solid rgba(148,163,184,.25);
  border-radius:16px;padding:18px 18px;color:#fff;width:100%;backdrop-filter:blur(20px);
  box-shadow:0 12px 40px rgba(0,0,0,.45),inset 0 1px 0 rgba(255,255,255,.06)}
#card .ph{font-size:10px;text-transform:uppercase;letter-spacing:1.6px;color:#94a3b8;margin-bottom:6px}
#card .tt{font-size:16px;font-weight:700;margin-bottom:8px;letter-spacing:-.3px}
#card .dd{font-size:12px;color:#cbd5e1;line-height:1.65}
#card .badge{margin-top:12px;font-size:10px;font-weight:700;padding:4px 12px;border-radius:999px;display:inline-block;color:#0f172a}
#card .ebar{margin-top:14px;border-top:1px solid rgba(255,255,255,.1);padding-top:12px}
#card .ebar .lbl{font-size:10px;color:#94a3b8;display:flex;justify-content:space-between;margin-bottom:6px}
#card .ebar .track{width:100%;height:5px;background:rgba(255,255,255,.08);border-radius:99px;overflow:hidden}
#card .ebar .fill{height:100%;border-radius:99px;background:linear-gradient(90deg,#38bdf8,#818cf8,#a78bfa)}
#card .conn{margin-top:12px;border-top:1px solid rgba(255,255,255,.1);padding-top:10px}
#card .conn .cl{font-size:10px;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px}
#card .conn button{background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.18);color:#e2e8f0;
  font-size:10px;padding:5px 10px;border-radius:6px;cursor:pointer;margin:3px 4px 3px 0;font-family:inherit;transition:background .15s,border-color .15s}
#card .conn button:hover{background:rgba(255,255,255,.12);border-color:rgba(255,255,255,.35);color:#fff}
.hint{position:absolute;bottom:14px;left:50%;transform:translateX(-50%);font-size:11px;color:rgba(148,163,184,.45);pointer-events:none;white-space:nowrap;text-shadow:0 1px 8px #000}
@media (max-width:640px){
  body,html{overflow:auto;-webkit-overflow-scrolling:touch}
  #w{flex-direction:column;height:auto;min-height:580px}
  .cvWrap{height:min(48vh,380px);min-height:260px;flex:none}
  #dock{width:100%!important;border-left:none;border-top:1px solid rgba(148,163,184,.15);max-height:none;padding:12px}
  .hint{font-size:10px;white-space:normal;text-align:center;max-width:90%;bottom:8px}
}
</style></head><body>
<div id="w">
<div class="cvWrap"><canvas id="c"></canvas>
<div class="hint">Orbit &bull; pause on node &bull; empty space to resume</div></div>
<div id="dock">
<div id="dockPh">Select any phase on the orbit.<br>Details show here without blocking the wheel.</div>
<div id="card">
  <div class="ph" id="cph"></div><div class="tt" id="ctt"></div><div class="dd" id="cdd"></div>
  <div class="badge" id="cbg"></div>
  <div class="ebar"><div class="lbl"><span>Energy Level</span><span id="cen"></span></div>
  <div class="track"><div class="fill" id="cef"></div></div></div>
  <div class="conn" id="ccn"><div class="cl">Connected Nodes</div><div id="cbt"></div></div>
</div>
</div>
</div>
<script>
var N=[
{t:"Problem Definition",p:"Phase 1",d:"Identified the Ac-225 supply crisis for targeted alpha therapy. Defined Ra-226 to Ac-225 transmutation chain.",s:"completed",co:"#10b981",e:100,r:[1]},
{t:"ODE Simulator",p:"Phase 2",d:"Built Bateman equation integrator (scipy odeint). Validated half-lives from NNDC. Generated 1,500 training trajectories.",s:"completed",co:"#0ea5e9",e:95,r:[0,2]},
{t:"PINN Architecture",p:"Phase 3",d:"Designed 4-layer MLP with IC constraint N(0)=N0. Mass budget cap, species-weighted data loss, Bateman residuals.",s:"completed",co:"#7c3aed",e:90,r:[1,3]},
{t:"First Training",p:"Phase 4",d:"Initial 4,000 epochs on i3 CPU. Discovered alchemy bug and Ra-225 underprediction. Added zero-injection penalty.",s:"completed",co:"#f59e0b",e:70,r:[2,4]},
{t:"SiLU Bug Fix",p:"Phase 5",d:"Trio test C revealed SiLU(0.01)=0.005 halved all ICs. Removed SiLU from output layer. Root cause of 50% mass loss.",s:"completed",co:"#ef4444",e:85,r:[3,5]},
{t:"Max Fix v2",p:"Phase 6",d:"1/v energy scaling, Ra-225 physics x5, non-negativity, secular equilibrium ceiling, 30% empty-tank collocation.",s:"completed",co:"#0d9488",e:95,r:[4,6]},
{t:"12k Training",p:"Phase 7",d:"Full 12,000-epoch run on i7 Evo. Loss converged 14k to 37. All trio tests passed. Median error -1.7%.",s:"completed",co:"#10b981",e:100,r:[5,7]},
{t:"Website",p:"Phase 8",d:"Professional Streamlit dashboard with live PINN predictions, speed benchmark, dose calculator, plot gallery.",s:"completed",co:"#0ea5e9",e:100,r:[6]}
];
var cv=document.getElementById("c"),ctx=cv.getContext("2d"),cvWrap=document.querySelector(".cvWrap");
var card=document.getElementById("card"),dockPh=document.getElementById("dockPh");
var W,H,cx,cy,ang=0,sel=-1,autoR=true;
var dpr=Math.min(window.devicePixelRatio||1,2.5);
var lastT=performance.now();
var pulseT=0;
var HIT_R2=26*26;

function sz(){
  W=cvWrap.clientWidth;H=cvWrap.clientHeight;
  cv.width=Math.max(1,Math.floor(W*dpr));cv.height=Math.max(1,Math.floor(H*dpr));
  cv.style.width=W+"px";cv.style.height=H+"px";
  ctx.setTransform(dpr,0,0,dpr,0,0);
  cx=W/2;cy=H/2;
}
sz();window.addEventListener("resize",sz);

function getR(){return Math.min(W,H)*0.30}
function npos(i){var R=getR();var a=i/N.length*Math.PI*2+ang-Math.PI/2;return{x:cx+R*Math.cos(a),y:cy+R*Math.sin(a)}}

function showCard(i){
  var n=N[i];
  document.getElementById("cph").textContent=n.p;
  document.getElementById("ctt").textContent=n.t;
  document.getElementById("cdd").textContent=n.d;
  var bg=document.getElementById("cbg");bg.textContent=n.s.toUpperCase().replace("-"," ");bg.style.background=n.co;
  document.getElementById("cen").textContent=n.e+"%";
  document.getElementById("cef").style.width=n.e+"%";
  var btns=document.getElementById("cbt");btns.innerHTML="";
  var cn=document.getElementById("ccn");
  if(n.r&&n.r.length){cn.style.display="block";n.r.forEach(function(ri){
    var b=document.createElement("button");b.textContent=N[ri].t;
    b.onclick=function(ev){ev.stopPropagation();clickNode(ri)};btns.appendChild(b)})
  }else{cn.style.display="none"}
  dockPh.style.display="none";
  card.style.display="block";
}
function hideCard(){
  card.style.display="none";
  dockPh.style.display="flex";
}

function clickNode(i){
  if(sel===i){sel=-1;autoR=true;hideCard();return}
  sel=i;autoR=false;
  showCard(i);
}

function pickNode(mx,my){
  var best=-1,bestD=1e18;
  for(var i=0;i<N.length;i++){
    var q=npos(i),dx=mx-q.x,dy=my-q.y,d2=dx*dx+dy*dy;
    if(d2<HIT_R2&&d2<bestD){bestD=d2;best=i}
  }
  return best;
}

function drawOrbitRing(R){
  var grd=ctx.createLinearGradient(cx-R,cy,cx+R,cy);
  grd.addColorStop(0,"rgba(45,212,191,.12)");grd.addColorStop(.5,"rgba(129,140,248,.2)");grd.addColorStop(1,"rgba(45,212,191,.12)");
  ctx.strokeStyle=grd;ctx.lineWidth=1.5;ctx.lineCap="round";
  ctx.beginPath();ctx.arc(cx,cy,R,0,Math.PI*2);ctx.stroke();
  ctx.strokeStyle="rgba(255,255,255,.05)";ctx.lineWidth=1;
  ctx.beginPath();ctx.arc(cx,cy,R+3,0,Math.PI*2);ctx.stroke();
}

function frame(ts){
  ts=ts||performance.now();
  var dt=Math.min(0.033,(ts-lastT)/1000);
  lastT=ts;
  pulseT+=dt;

  ctx.clearRect(0,0,W,H);

  var R=getR();
  drawOrbitRing(R);

  var g=ctx.createRadialGradient(cx,cy,0,cx,cy,R*0.45);
  g.addColorStop(0,"rgba(124,58,237,.35)");g.addColorStop(.25,"rgba(59,130,246,.18)");g.addColorStop(.5,"rgba(13,148,136,.08)");g.addColorStop(1,"transparent");
  ctx.fillStyle=g;ctx.beginPath();ctx.arc(cx,cy,R*0.45,0,Math.PI*2);ctx.fill();

  var ping=(Math.sin(ts*0.003)+1)*0.5;
  ctx.strokeStyle="rgba(255,255,255,"+(0.12+0.08*ping)+")";ctx.lineWidth=1.5;ctx.lineCap="round";
  ctx.beginPath();ctx.arc(cx,cy,18+ping*14,0,Math.PI*2);ctx.stroke();

  ctx.fillStyle="rgba(255,255,255,.92)";
  ctx.shadowColor="rgba(56,189,248,.5)";ctx.shadowBlur=12;
  ctx.beginPath();ctx.arc(cx,cy,5.5,0,Math.PI*2);ctx.fill();
  ctx.shadowBlur=0;

  for(var i=0;i<N.length;i++){
    var q=npos(i),n=N[i],isSel=i===sel,isRel=sel>=0&&N[sel].r&&N[sel].r.indexOf(i)>=0;
    var breathe=autoR?1+0.06*Math.sin(pulseT*2.2+i*0.7):1;
    var r=(isSel?15:isRel?10:8)*breathe;
    var grad=ctx.createLinearGradient(cx,cy,q.x,q.y);
    grad.addColorStop(0,"rgba(255,255,255,.04)");grad.addColorStop(1,isSel?"rgba(255,255,255,.1)":isRel?"rgba(255,255,255,.06)":"rgba(255,255,255,.02)");
    ctx.strokeStyle=grad;ctx.lineWidth=isSel?2:1;ctx.lineCap="round";
    ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(q.x,q.y);ctx.stroke();

    if(isSel||isRel){
      var gg=ctx.createRadialGradient(q.x,q.y,0,q.x,q.y,42);
      gg.addColorStop(0,n.co+(isSel?"55":"28"));gg.addColorStop(1,"transparent");
      ctx.fillStyle=gg;ctx.beginPath();ctx.arc(q.x,q.y,42,0,Math.PI*2);ctx.fill();
    }
    ctx.fillStyle=isSel?"#f8fafc":isRel?"rgba(248,250,252,.75)":n.co;
    ctx.beginPath();ctx.arc(q.x,q.y,r+1.5,0,Math.PI*2);ctx.fill();
    ctx.fillStyle=isSel?"#fff":isRel?"rgba(255,255,255,.85)":n.co;
    ctx.beginPath();ctx.arc(q.x,q.y,r,0,Math.PI*2);ctx.fill();
    ctx.strokeStyle=isSel?"#fff":isRel?"rgba(255,255,255,.9)":"rgba(255,255,255,.35)";
    ctx.lineWidth=isSel?2.2:isRel?1.8:1.2;
    ctx.beginPath();ctx.arc(q.x,q.y,r,0,Math.PI*2);ctx.stroke();

    ctx.fillStyle=isSel?"#fff":isRel?"rgba(248,250,252,.9)":"rgba(203,213,225,.85)";
    ctx.font=(isSel?"600 13px":isRel?"500 12px":"500 11px")+" Inter,system-ui,sans-serif";
    ctx.textAlign="center";ctx.textBaseline="top";
    ctx.fillText(n.t,q.x,q.y+r+12);
  }

  if(autoR)ang+=dt*0.21;

  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

cv.addEventListener("click",function(e){
  var rect=cv.getBoundingClientRect(),mx=e.clientX-rect.left,my=e.clientY-rect.top;
  var hit=pickNode(mx,my);
  if(hit>=0)clickNode(hit);
  else{sel=-1;autoR=true;hideCard()}
});
cv.addEventListener("mousemove",function(e){
  var rect=cv.getBoundingClientRect(),mx=e.clientX-rect.left,my=e.clientY-rect.top;
  cv.style.cursor=pickNode(mx,my)>=0?"pointer":"default";
});
</script></body></html>"""
    components.html(ORBITAL_HTML, height=720, scrolling=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 7 -- TRAINING RESULTS
# ═══════════════════════════════════════════════════════════════════════════
with tab_results:
    st.markdown('<div class="sh">Training Performance</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        p = ROOT / "pinn_loss_history.png"
        if p.exists():
            st.image(Image.open(p), caption="Loss convergence over 12,000 epochs", use_container_width=True)
    with c2:
        p = ROOT / "pinn_ac225_pred_vs_true.png"
        if p.exists():
            st.image(Image.open(p), caption="Ac-225 parity: PINN vs ODE (color = log error)", use_container_width=True)

    st.markdown('<div class="sh">Reading the Plots</div>', unsafe_allow_html=True)
    st.markdown("""
- **Loss plot (left)**: Both data (blue) and physics (orange) losses drop steadily over 12k epochs, confirming the network learns real dynamics -- not just curve-fitting.
- **Parity plot (right)**: Each dot is one training sample. Points on the dashed line mean perfect agreement with the ODE. The color shows log-scale error: cooler = lower error. The green shaded band is the 2x envelope.
    """)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 8 -- PLOT GALLERY
# ═══════════════════════════════════════════════════════════════════════════
with tab_gallery:
    st.markdown('<div class="sh">Every Plot Explained</div>', unsafe_allow_html=True)
    st.markdown("Each visualization in this project serves a specific purpose. Here is what they show and why they matter.")

    GALLERY = [
        ("pinn_loss_history.png", "Training Loss History",
         "Shows data MSE (blue) and physics MSE (orange) on a log scale over all 12,000 epochs. Both curves drop steadily and plateau, meaning the network converged. If the physics line stayed flat while data dropped, the network would be curve-fitting without learning real physics."),
        ("pinn_ac225_pred_vs_true.png", "Ac-225 Parity Plot",
         "Every training sample is plotted as (ODE truth, PINN prediction). Points on the y=x line are perfect. The color encodes log-scale error. This proves the model tracks Ac-225 production across the full dynamic range -- from trace amounts to bulk production."),
        ("training_coverage_counts.png", "Training Data Coverage",
         "A heatmap of how many training samples land in each (flux, time) bin. Dark/empty cells are 'holes' where the model has no data and may extrapolate poorly. This guided the decision to add diverse ICs and targeted augmentation."),
        ("feature_importance.png", "Feature Importance (Random Forest)",
         "A separate Random Forest model trained on the same CSV to predict peak Ac-225 yield. This bar chart shows that flux (phi) dominates yield prediction, with irradiation time as a secondary factor. Useful for understanding which knobs matter most."),
        ("training_pairplot.png", "Feature Relationships (Pair Plot)",
         "Scatter matrix of flux, time, and Ac-225 yield. Diagonal shows distributions; off-diagonal shows pairwise relationships. Reveals that yield correlates strongly with flux (log-scale) and shows the training distribution shape."),
        ("ac225_yield_heatmap.png", "Yield Landscape Heatmap",
         "2D map of maximum Ac-225 yield across (flux, time) space. Color = peak atoms produced. Shows the 'sweet spots' for production planning: high flux + moderate time gives the best yield. Operators use maps like this to pick irradiation schedules."),
        ("ac225_growth.png", "Ac-225 Growth Curve (Single Run)",
         "A single ODE integration showing Ac-225 inventory rising over time at fixed flux. This is the reference curve the PINN is trying to learn. The shape (initial lag from Ra-225 buildup, then steady growth) is characteristic of Bateman chains."),
        ("flux_sensitivity.png", "Flux Sensitivity: PINN vs ODE",
         "Ac-225 yield as a function of neutron flux at fixed time (200 h). The solid green line is ODE truth; the dashed blue line is the PINN prediction. Close agreement across two decades of flux confirms the model correctly captures how the main production knob (flux) drives yield."),
        ("error_histogram.png", "Prediction Error Distribution",
         "Histogram of relative error ((PINN - ODE) / ODE) for Ac-225 across all 7,200 training samples. The distribution is tightly centered near zero with a median of about -1.7%, meaning the model is slightly conservative but rarely far off. Most predictions fall within a few percent of the ODE."),
    ]

    for fname, title, explanation in GALLERY:
        fpath = ROOT / fname
        if fpath.exists():
            c1, c2 = st.columns([1, 1])
            with c1:
                st.image(Image.open(fpath), use_container_width=True)
            with c2:
                st.markdown(f'<div class="gallery-text"><h4>{title}</h4><p>{explanation}</p></div>', unsafe_allow_html=True)
            st.divider()

# ═══════════════════════════════════════════════════════════════════════════
# TAB 9 -- TRIO VALIDATION
# ═══════════════════════════════════════════════════════════════════════════
with tab_trio:
    st.markdown('<div class="sh">Trio Validation Test</div>', unsafe_allow_html=True)
    st.markdown("Three scenarios that any nuclear production model **must** pass before it can be trusted.")

    st.markdown("""
<div class="tc tp"><h4>A -- Empty Tank + High Flux</h4>
<p><b>Setup:</b> No initial atoms, reactor at full power (10<sup>15</sup> n/cm<sup>2</sup>/s).<br>
<b>Expected:</b> Zero output -- you cannot create matter from nothing.<br>
<b>Result:</b> PINN outputs exactly 0 for all species. <b>No alchemy.</b></p></div>
    """, unsafe_allow_html=True)
    st.markdown("""
<div class="tc tp"><h4>B -- Full Tank (Ra-226 = 10<sup>22</sup>) + Reactor Flux</h4>
<p><b>Setup:</b> 10<sup>22</sup> Ra-226 atoms, moderate flux, 250 hours.<br>
<b>Expected:</b> Small Ra-226 depletion, Ra-225 and Ac-225 production.<br>
<b>Result:</b> Ra-226 within 0.004% of ODE. Ac-225 within 7% of ODE. <b>Mass conserved.</b></p></div>
    """, unsafe_allow_html=True)
    st.markdown("""
<div class="tc tp"><h4>C -- Pure Decay (Ra-225 only, zero flux)</h4>
<p><b>Setup:</b> 10<sup>18</sup> Ra-225 atoms, reactor off, 48 hours.<br>
<b>Expected:</b> Ra-225 decays to Ac-225 via beta emission. No Ra-226 appears.<br>
<b>Result:</b> Ra-226 stays at 0 (correct). Ra-225 within 5% of ODE. Ac-225 growing correctly.</p></div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sh">Why These Tests Matter</div>', unsafe_allow_html=True)
    st.markdown("""
| Test | What It Catches | Real-World Analog |
|------|----------------|-------------------|
| **A** | "Alchemy" -- AI hallucinating atoms from nothing | False positive in production schedule |
| **B** | Fuel tracking accuracy | Target inventory management |
| **C** | Decay chain fidelity when reactor is off | Storage / cooldown predictions |
    """)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 10 -- STRUGGLES & FAILURES
# ═══════════════════════════════════════════════════════════════════════════
with tab_struggles:
    st.markdown('<div class="sh">Development Log: What Went Wrong</div>', unsafe_allow_html=True)
    st.markdown("Every real project has failures. Here are ours, documented honestly.")

    bugs = [
        ("Phase 3", True, "The SiLU Initial Condition Bug",
         "The output activation function (SiLU) was halving all initial conditions. SiLU(0.01) = 0.005, destroying the physics guarantee that N(t=0) = N0. The trio test showed 50% mass loss at t=0.",
         "Discovered via Trio Test C: Ra-225 started at 1e18 but the PINN output was ~5e17 immediately. Traced to SiLU(x) = x * sigmoid(x) where sigmoid(0.01) = 0.50.",
         "Removed SiLU from the output layer entirely. The IC constraint now gives exact N0 at t=0. Non-negativity penalty handles negatives during training."),
        ("Phase 2", True, "Alchemy / Mass Inflation",
         "Early models created atoms from nothing. An empty tank fed with high neutron flux would produce Ac-225 -- physically impossible without feedstock.",
         "The network found a 'shortcut' that minimized physics loss without respecting conservation. Mass loss in training was one-sided (only penalized creation, not loss).",
         "Added hard budget cap in forward(), zero-injection penalty (weight 100-200), and 30% empty-tank collocation points in every pretrain batch."),
        ("Phase 4", True, "Ra-225 60% Underprediction",
         "The PINN consistently underpredicted Ra-225 by ~60%. The physics loss was dominated by the Ra-226 equation because Ra-226 inventories are 1000x larger.",
         "Ra-226 has scale ~6e23 while Ra-225 peaks at ~1e20. The MSE of the Ra-226 residual dwarfed Ra-225.",
         "Weighted the Ra-225 Bateman residual by 5x in the physics loss. Also added per-species data weights (1, 80, 80) so daughter species get attention."),
        ("Phase 1", True, "Broken Windows venv",
         "The project was moved between machines. The venv pointed at a different user's Python install path and refused to run.",
         "Both .venv and venv folders had hardcoded paths from the original machine (C:\\Users\\ayomi\\...).",
         "Created a fresh venv with the local Python 3.12 install, updated requirements.txt to flexible version ranges, and replaced the broken environment."),
        ("Phase 6", True, "PINN_MEDIUM_TRAIN Silent Epoch Cap",
         "The 'fast training' script secretly set PINN_MEDIUM_TRAIN=1, cutting epochs from 4,500 to 2,300 without clear indication.",
         "The run_train_fast.ps1 launcher set both PINN_FAST_CPU and PINN_MEDIUM_TRAIN. The user thought they were running full training.",
         "Removed PINN_MEDIUM_TRAIN from the fast launcher. PINN_FAST_CPU now only affects hardware (threads, chunk size), never epoch count."),
        ("Phase 5", True, "Trio C: Ra-226 Ghost at Zero Flux",
         "At zero flux with only Ra-225 initially, the PINN predicted ~1e18 Ra-226 appearing from nothing. ODE correctly shows zero Ra-226.",
         "Root cause was the SiLU bug: the network compensated for SiLU attenuation by shifting mass into the Ra-226 slot, which has the largest scale factor.",
         "Fixed by the SiLU removal. After the fix, Ra-226 correctly stays at zero in zero-flux scenarios."),
    ]

    for phase, fixed, title, problem, investigation, fix in bugs:
        cls = "bug-card fixed" if fixed else "bug-card"
        status = "FIXED" if fixed else "OPEN"
        st.markdown(f"""
<div class="{cls}">
<span class="phase">{phase}</span>
<h4>{"✅" if fixed else "🔴"} {title}</h4>
<p><b>Problem:</b> {problem}</p>
<p><b>Investigation:</b> {investigation}</p>
<p><b>Fix:</b> {fix}</p>
</div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 11 -- MISTAKES WE MADE
# ═══════════════════════════════════════════════════════════════════════════
with tab_mistakes:
    st.markdown('<div class="sh">Honest Retrospective</div>', unsafe_allow_html=True)
    st.markdown("These are not bugs -- they are **judgment calls** we got wrong. Documenting them so the next person (or the next project) can skip the detour.")

    mistakes = [
        ("Using SiLU as an output activation without testing on small inputs",
         "SiLU is a great hidden-layer activation. But as an output layer it halves any value near zero because sigmoid(0.01) = 0.50. We assumed 'smooth non-negativity' without checking the actual numeric behavior on our normalized scale (0.01 -- 0.02). A 30-second test in a Python shell would have caught this immediately."),
        ("Not testing with diverse initial conditions until late",
         "The training CSV only contains virgin Ra-226 fuel trajectories. We did not generate inverted ICs (Ra-225 dominant, zero flux, decay-only) until Trio Test C failed. The model had never seen the decay-only regime and could not generalize to it. Lesson: test with out-of-distribution inputs early, not as a final check."),
        ("Trusting training-log mass=0 without running the trio test",
         "The mass conservation loss was one-sided: it only penalized atom creation, not atom loss. Training logs showed mass_cons_loss = 0, which looked perfect. But the trio test revealed 50% mass deficit. The metric was measuring the wrong thing. Lesson: validation tests beat training metrics."),
        ("Starting with TIME_SCALE=100 and PHI_SCALE=1e14 instead of normalizing to [0, 1]",
         "Input normalization matters. With TIME_SCALE=100 and times up to 500h, t_nn could reach 5.0 -- outside the ideal [0, 1] range. Changed to TIME_SCALE=500 and PHI_SCALE=1e15 so both inputs sit in [0, 1]. The network learned faster and more stably after this change."),
    ]

    for title, body in mistakes:
        st.markdown(f"""
<div class="mistake-card">
<h4>{title}</h4>
<p>{body}</p>
</div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 12 -- FUTURE APPLICATIONS
# ═══════════════════════════════════════════════════════════════════════════
with tab_future:
    st.markdown('<div class="sh">Where This Goes Next</div>', unsafe_allow_html=True)
    st.markdown("This prototype demonstrates the core idea. Here is how it extends to real-world impact.")

    futures = [
        ("🏥", "Clinical Production Planning",
         "Hospitals and isotope producers need to decide flux, time, and target composition for each production campaign. A trained PINN can sweep thousands of candidate schedules in seconds instead of hours, letting planners optimize yield and minimize waste."),
        ("📦", "Supply Chain Optimization",
         "Ac-225 has a 10-day half-life -- it decays during shipping. The PINN can model harvest-to-delivery windows and predict how much usable isotope arrives at the clinic. Integrate with logistics to minimize decay losses."),
        ("🖥️", "Digital Twin for Irradiation Facilities",
         "Embed the PINN in a live monitoring dashboard that reads flux sensors and predicts current isotope inventories in real time. Operators see a continuously updated state estimate without waiting for offline analysis."),
        ("🔬", "Transfer to Other Decay Chains",
         "The same PINN architecture works for Mo-99 (Tc-99m generator for diagnostic imaging), Lu-177 (therapeutic beta emitter), and other chains. Swap the Bateman equations and retrain -- the infrastructure is chain-agnostic."),
        ("📊", "Uncertainty Quantification",
         "Train an ensemble of PINNs or add a Bayesian output layer to produce confidence intervals, not just point predictions. Critical for regulatory submissions where error bars matter as much as the central estimate."),
        ("🤝", "Multi-Facility Coordination",
         "If multiple reactors produce Ac-225, the PINN can model each facility's expected output and help coordinate shipments to maximize global supply. A fast surrogate makes multi-site optimization tractable."),
    ]

    cols = st.columns(2)
    for i, (icon, title, body) in enumerate(futures):
        with cols[i % 2]:
            st.markdown(f"""
<div class="future-card">
<div style="font-size:2rem">{icon}</div>
<h4>{title}</h4>
<p>{body}</p>
</div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 13 -- TECHNICAL DETAILS
# ═══════════════════════════════════════════════════════════════════════════
with tab_tech:
    st.markdown('<div class="sh">Architecture</div>', unsafe_allow_html=True)
    st.markdown("""
<table class="tt">
<tr><th>Component</th><th>Detail</th></tr>
<tr><td>Network</td><td>4-layer MLP, 128 hidden units, SiLU activations (hidden only), 51,819 parameters</td></tr>
<tr><td>IC Enforcement</td><td>N(t) = N<sub>0</sub> + t * sigmoid(1000t) * rate(x) -- exact IC at t=0, no output activation</td></tr>
<tr><td>Output Cap</td><td>Hard budget: sum(N_pred) &le; sum(N_0) at inference (no alchemy)</td></tr>
<tr><td>Optimizer</td><td>Adam, LR 10<sup>-3</sup>, ReduceLROnPlateau (patience 500)</td></tr>
<tr><td>Training</td><td>2,000 physics pretrain + 10,000 joint = 12,000 total epochs</td></tr>
</table>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sh">Loss Function (8 Terms)</div>', unsafe_allow_html=True)
    st.markdown("""
| Term | Weight | Purpose |
|------|--------|---------|
| Physics MSE (Bateman residuals) | 2,000 | ODE consistency |
| Ra-225 residual boost | 5x | Fix underprediction |
| Data MSE (CSV fit) | 50 | Match training trajectories |
| Mass conservation | 350 | No net atom creation |
| Fuel anchor (Ra-226 burnup) | 100 | Correct depletion rate |
| Non-negativity | 50 | Atoms cannot go negative |
| Secular equilibrium ceiling | 25 | Ac-225 &le; transient eq ratio |
| Zero-injection | 100-200 | Empty tank stays empty |
    """)

    st.markdown('<div class="sh">Nuclear Data (NNDC)</div>', unsafe_allow_html=True)
    st.markdown("""
| Isotope | Half-life | Decay Mode | Source |
|---------|-----------|------------|--------|
| Ra-226 | 1,600 years | Alpha | NNDC |
| Ra-225 | 14.9 days | Beta- to Ac-225 | NNDC |
| Ac-225 | 9.920 days | Alpha (4 alphas in chain) | NNDC |
    """)

    st.markdown('<div class="sh">Bateman Equations</div>', unsafe_allow_html=True)
    st.latex(r"\frac{dN_{226}}{dt} = -(\lambda_{226} + k) \, N_{226}")
    st.latex(r"\frac{dN_{225}}{dt} = k \, N_{226} \frac{S_{226}}{S_{225}} - \lambda_{225} \, N_{225}")
    st.latex(r"\frac{dN_{Ac}}{dt} = \lambda_{225} \, N_{225} \frac{S_{225}}{S_{Ac}} - \lambda_{Ac} \, N_{Ac}")
    st.markdown(r"Where $k = \phi \cdot \sigma \cdot \sqrt{0.025/E} \cdot 3600$ (1/v energy scaling, per hour)")

    st.markdown('<div class="sh">Input Normalization</div>', unsafe_allow_html=True)
    st.markdown("""
| Input | Raw Unit | Scale | Normalized Range |
|-------|----------|-------|-----------------|
| Time | hours | / 500 | [0, 1] |
| Flux | n/cm^2/s | / 10^15 | [0, 1] |
| Energy | eV | sqrt(0.025/E) | ~[0.16, 1.6] |
| Ra-226 | atoms | / 6.022e23 | [0, ~1] |
| Ra-225 | atoms | / 1e20 | [0, ~5] |
| Ac-225 | atoms | / 1e20 | [0, ~2] |
    """)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("""
<div class="ft">
<b>PINN Isotope Transmutation Model</b> -- Physics-informed AI for Ac-225 production planning<br>
Built with PyTorch + Streamlit &nbsp;|&nbsp;
<a href="https://en.wikipedia.org/wiki/Actinium-225" target="_blank">About Ac-225</a> &nbsp;|&nbsp;
<a href="https://en.wikipedia.org/wiki/Bateman_equation" target="_blank">Bateman Equations</a> &nbsp;|&nbsp;
<a href="https://www.nndc.bnl.gov/" target="_blank">NNDC Nuclear Data</a>
</div>
""", unsafe_allow_html=True)
