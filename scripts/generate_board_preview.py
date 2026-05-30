"""
Build a self-contained tri-fold poster HTML with images embedded as base64.

This fixes the "pictures don't show" problem: the PNGs are baked into the HTML,
so it renders identically no matter where the file is opened (no path issues).

Run:  python scripts/generate_board_preview.py
Out:   poster/board_preview.html   (open in any browser; Print -> Save as PDF)
"""
from __future__ import annotations

import base64
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GRAPHS = ROOT / "graphs"
OUT = ROOT / "poster" / "board_preview.html"


def _b64(path: Path) -> str:
    if not path.is_file():
        return ""
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    fig_mass = _b64(GRAPHS / "isef_mass_conservation.png")
    fig_parity = _b64(GRAPHS / "isef_parity_restyled.png")
    fig_eval = _b64(GRAPHS / "isef_isotope_evolution.png")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>IsotopePINN — Tri-Fold Board</title>
<style>
  :root {{
    --ink:#0f172a; --muted:#475569; --accent:#1d4ed8; --accent-soft:#dbeafe;
    --pass:#059669; --pass-soft:#ecfdf5; --border:#cbd5e1; --panel:#f8fafc;
  }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family:"Segoe UI",Arial,sans-serif; background:#b8bfc9; color:var(--ink); padding:24px; }}

  .toolbar {{
    max-width:1400px; margin:0 auto 16px; background:#1e293b; color:#fff;
    padding:12px 18px; border-radius:8px; display:flex; gap:16px; align-items:center; flex-wrap:wrap;
  }}
  .toolbar button {{ background:var(--accent); color:#fff; border:0; padding:9px 16px; border-radius:6px; cursor:pointer; font-size:14px; }}
  .toolbar span {{ font-size:13px; opacity:.9; }}

  .board {{
    max-width:1400px; margin:0 auto; background:#fff;
    display:grid; grid-template-columns:1fr 1fr 1fr;
    box-shadow:0 20px 50px rgba(0,0,0,.3); border-radius:6px; overflow:hidden;
  }}
  .panel {{ padding:26px 24px; border-right:2px dashed var(--border); }}
  .panel:last-child {{ border-right:none; }}
  .panel-left {{ background:var(--panel); }}
  .panel-right {{ background:var(--panel); }}

  .title-block {{ text-align:center; margin-bottom:18px; }}
  .title-block h1 {{ font-size:24px; line-height:1.2; color:var(--ink); }}
  .title-block .name {{ font-size:18px; font-weight:700; margin-top:10px; }}
  .title-block .school {{ font-size:13px; color:var(--muted); margin-top:2px; }}

  h2 {{
    font-size:16px; font-weight:800; text-transform:uppercase; letter-spacing:.05em;
    color:var(--accent); border-bottom:2px solid var(--accent-soft);
    padding-bottom:5px; margin:18px 0 10px;
  }}
  .panel-center h2:first-of-type {{ margin-top:8px; }}
  p, li {{ font-size:14px; line-height:1.5; margin-bottom:9px; }}
  ul, ol {{ padding-left:20px; }}
  li {{ margin-bottom:6px; }}

  .box {{ border-radius:8px; padding:12px 14px; margin:10px 0; font-size:14px; line-height:1.45; }}
  .box-rq {{ background:var(--accent-soft); border:2px solid var(--accent); font-weight:600; }}
  .box-chain {{ background:var(--pass-soft); border:2px solid var(--pass); font-size:13px; }}
  .box-chain b {{ font-size:14px; }}

  .steps {{ list-style:none; padding:0; counter-reset:s; }}
  .steps li {{ counter-increment:s; position:relative; padding-left:34px; margin-bottom:11px; font-size:13.5px; }}
  .steps li::before {{
    content:counter(s); position:absolute; left:0; top:1px;
    width:22px; height:22px; background:var(--accent); color:#fff;
    border-radius:50%; font-size:12px; font-weight:700; text-align:center; line-height:22px;
  }}

  table {{ width:100%; border-collapse:collapse; font-size:13px; margin:8px 0; }}
  th, td {{ border:1px solid var(--border); padding:6px 8px; text-align:left; }}
  th {{ background:var(--accent-soft); }}
  .pass {{ color:var(--pass); font-weight:700; }}

  figure {{ margin:12px 0; }}
  figure img {{ width:100%; height:auto; border:1px solid #888; border-radius:4px; display:block; }}
  figcaption {{ font-size:12px; color:var(--muted); margin-top:6px; line-height:1.35; }}
  figcaption b {{ color:var(--ink); }}
  .fig-row {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }}

  .badge {{ display:inline-block; background:var(--pass); color:#fff; font-size:12px; font-weight:700; padding:3px 9px; border-radius:4px; margin:0 4px 6px 0; }}

  .footer {{
    max-width:1400px; margin:0 auto; background:#eef2f7; border-radius:0 0 6px 6px;
    display:grid; grid-template-columns:2fr 1fr; gap:20px; padding:18px 24px;
    box-shadow:0 10px 30px rgba(0,0,0,.2);
  }}
  .footer h3 {{ font-size:14px; text-transform:uppercase; letter-spacing:.05em; border-bottom:1px solid #333; margin-bottom:6px; }}
  .footer p {{ font-size:12px; line-height:1.4; }}
  .demo a {{ color:var(--accent); word-break:break-all; font-size:12px; }}

  @media print {{
    body {{ background:#fff; padding:0; }}
    .toolbar {{ display:none; }}
    .board, .footer {{ box-shadow:none; max-width:100%; }}
    @page {{ size:48in 36in landscape; margin:.3in; }}
  }}
  @media (max-width:900px) {{
    .board {{ grid-template-columns:1fr; }}
    .panel {{ border-right:none; border-bottom:2px dashed var(--border); }}
    .footer {{ grid-template-columns:1fr; }}
  }}
</style>
</head>
<body>

<div class="toolbar">
  <strong>IsotopePINN tri-fold board</strong>
  <span>Pictures are embedded — they always show. Print → Save as PDF (48×36 landscape) for the real board.</span>
  <button type="button" onclick="window.print()">Print / Save PDF</button>
</div>

<div class="board">

  <!-- LEFT -->
  <div class="panel panel-left">
    <h2>Background</h2>
    <p>Actinium-225 is a scarce alpha-emitting radiopharmaceutical used in
       <b>targeted alpha therapy (TAT)</b> for cancer. Supply limits clinical trials and patient access.</p>
    <p>Planning irradiation — neutron flux, energy, and time — requires solving a stiff
       <b>five-isotope transmutation chain</b> many times. ODE integrators are accurate but too slow
       for large parameter sweeps.</p>
    <div class="box box-chain">
      <b>Five-species chain (0D)</b><br>
      Ra-226 → Ra-225 → Ac-225 &nbsp;(product)<br>
      Ra-226 → Ra-227 → Ac-227 &nbsp;(impurity)
    </div>

    <h2>Research Question</h2>
    <div class="box box-rq">
      Can a physics-informed neural network accurately and rapidly predict Ac-225 inventory across
      diverse irradiation scenarios compared to a trusted Bateman ODE reference?
    </div>

    <h2>Hypothesis</h2>
    <p>If Bateman physics is embedded in the architecture and training loss, the PINN will match the
       ODE within <b>10%</b> on held-out scenarios while running orders-of-magnitude faster than
       sequential ODE integration.</p>

    <h2>Expected Outcomes</h2>
    <ul>
      <li>Six independent validation gates PASS</li>
      <li>Held-out Ac-225 median error &lt; 10% vs ODE</li>
      <li>Strongest accuracy: thermal &amp; 14 MeV regimes</li>
      <li>Largest errors: epithermal (~9.5%) &amp; threshold (~8.5%)</li>
    </ul>
  </div>

  <!-- CENTER -->
  <div class="panel panel-center">
    <div class="title-block">
      <h1>Computational Surrogate for Ac-225 Production Planning in Targeted Alpha Therapy</h1>
      <div class="name">Samuel Ogunnubi</div>
      <div class="school">Anne Arundel Community College · Dual Enrollment · May 2026</div>
    </div>

    <h2>Methodology</h2>
    <ol class="steps">
      <li><b>Reference:</b> 0D five-species Bateman ODE (NNDC/ENSDF half-lives, JENDL cross sections), stiff Radau solver generates training targets.</li>
      <li><b>Coverage:</b> Scenarios across thermal, epithermal, threshold (~6.4 MeV), and 14 MeV energies; virgin and recycled inventories.</li>
      <li><b>Surrogate:</b> Physics-informed NN with semi-analytic Bateman backbone and bounded corrections.</li>
      <li><b>Training:</b> 600-epoch physics pretrain + 3,400-epoch joint (v63 weights); mass conservation in loss.</li>
      <li><b>Validation:</b> Six independent gates + 22 held-out scenarios (seed 42).</li>
      <li><b>Demo:</b> Streamlit app for live PINN vs ODE and parameter screening.</li>
    </ol>

    <figure>
      <img src="{fig_mass}" alt="Mass conservation" />
      <figcaption><b>Figure 1 — Mass conservation.</b> Five-species atom budget drift (ppm) vs time;
        PINN stays within ±10 ppm training band (virgin Ra-226, φ = 1×10¹⁴, 14 MeV).</figcaption>
    </figure>
  </div>

  <!-- RIGHT -->
  <div class="panel panel-right">
    <h2>Results</h2>
    <p><span class="badge">6/6 PASS</span><span class="badge">4.51% held-out</span> Weights v63 · sha256 <code>7c21debe</code></p>
    <table>
      <tr><th>Validation check</th><th>Result</th></tr>
      <tr><td>Empty-target safety</td><td class="pass">PASS</td></tr>
      <tr><td>Production (14 MeV)</td><td class="pass">PASS (9.9%)</td></tr>
      <tr><td>Decay-chain ingrowth</td><td class="pass">PASS</td></tr>
      <tr><td>Species quality gate</td><td class="pass">PASS</td></tr>
      <tr><td>PINN vs ODE correlation</td><td class="pass">PASS</td></tr>
      <tr><td>Held-out Ac-225 (22)</td><td class="pass">4.51% median</td></tr>
    </table>

    <div class="fig-row">
      <figure>
        <img src="{fig_parity}" alt="Parity" />
        <figcaption><b>Fig 2 — Parity.</b> PINN vs ODE, 22 held-out; 4.51% median.</figcaption>
      </figure>
      <figure>
        <img src="{fig_eval}" alt="Evolution" />
        <figcaption><b>Fig 3 — Evolution.</b> Ac-225 vs time; PINN tracks ODE.</figcaption>
      </figure>
    </div>

    <h2>Results &amp; Conclusions</h2>
    <p>The PINN passed <b>6/6 gates</b> with <b>4.51%</b> held-out Ac-225 error vs ODE — enabling rapid
       screening impractical with repeated stiff solves. Strongest: thermal / 14 MeV; weakest:
       epithermal and ~6.4 MeV threshold.</p>
    <p><b>Limitations:</b> Validated vs ODE only — not reactor or clinical data. 0D model; not patient
       dosing or 3D transport (MCNP/OpenMC).</p>
  </div>
</div>

<div class="footer">
  <div>
    <h3>Key References</h3>
    <p>Raissi, M., Perdikaris, P., &amp; Karniadakis, G. E. (2019). Physics-informed neural networks.
       <i>Journal of Computational Physics</i>, 378, 686–707.</p>
    <p>NNDC/NuDat decay data; JENDL-4.0 Ra-226 cross sections; DOE Isotope Program (Ac-225 supply).</p>
    <h3 style="margin-top:10px;">Acknowledgements</h3>
    <p>Adult sponsor and science fair mentor. Faculty reviewers (pending). GPU training via Colab/Kaggle.</p>
  </div>
  <div class="demo">
    <h3>Live Demo</h3>
    <p><a href="https://lhyjrhmwzxqfpuuwsux7zh.streamlit.app">lhyjrhmwzxqfpuuwsux7zh.streamlit.app</a></p>
    <p><a href="https://github.com/samogunnubi0-del/PINN">github.com/samogunnubi0-del/PINN</a></p>
    <p style="color:#475569;margin-top:6px;">First load ~1 min if asleep.</p>
  </div>
</div>

</body>
</html>"""

    OUT.write_text(html, encoding="utf-8")
    kb = OUT.stat().st_size // 1024
    missing = [n for n, b in (("mass", fig_mass), ("parity", fig_parity), ("evolution", fig_eval)) if not b]
    note = f" (missing: {', '.join(missing)})" if missing else " (all 3 figures embedded)"
    print(f"Saved {OUT.relative_to(ROOT)} ({kb} KB){note}")


if __name__ == "__main__":
    main()
