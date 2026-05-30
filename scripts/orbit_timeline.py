"""Interactive orbital research timeline for the Streamlit Debug tab.

build_orbital_timeline_html(initial_index) returns a self-contained HTML/JS
canvas widget. Improvements over the original orbit:
  - drifting starfield + nebula backdrop
  - animated comet that traces the orbit ring
  - progress arc highlighting completed phases up to the selection
  - smooth eased "snap-to-front" rotation when a node is selected
  - curved bezier connectors between related nodes
  - glowing center hub showing the active phase
  - hover tooltips, sequence numbers inside nodes
  - auto-play guided tour (play/pause) + keyboard nav
"""

from __future__ import annotations

import json

ORBIT_NODES = [
    {
        "t": "Problem Definition",
        "p": "Phase 1",
        "d": "Identified the Ac-225 supply crisis for targeted alpha therapy. Defined the Ra-226 -> Ac-225 transmutation chain and training domain.",
        "s": "completed",
        "co": "#10b981",
        "e": 100,
        "r": [1],
    },
    {
        "t": "ODE Simulator",
        "p": "Phase 2",
        "d": "Built stiff Bateman integrator (Radau). Validated half-lives from NNDC/JENDL. Generated ~1,500 ODE trajectories as ground truth.",
        "s": "completed",
        "co": "#0ea5e9",
        "e": 95,
        "r": [0, 2],
    },
    {
        "t": "PINN Architecture",
        "p": "Phase 3",
        "d": "4-layer MLP with hard IC N(0)=N0, species-weighted data loss, Bateman residuals, mass-budget cap, and zero-injection collocation.",
        "s": "completed",
        "co": "#7c3aed",
        "e": 90,
        "r": [1, 3],
    },
    {
        "t": "First Training",
        "p": "Phase 4",
        "d": "Initial 4k epochs on CPU. Found alchemy under high flux and Ra-225 underprediction -> led to physics-only pretrain + empty-tank penalties.",
        "s": "completed",
        "co": "#f59e0b",
        "e": 70,
        "r": [2, 4],
    },
    {
        "t": "SiLU Bug Fix",
        "p": "Phase 5",
        "d": "Trio C showed SiLU(0.01)=0.005 halving all ICs - systematic 50% mass loss. Removed SiLU from output heads.",
        "s": "completed",
        "co": "#ef4444",
        "e": 85,
        "r": [3, 5],
    },
    {
        "t": "Bateman Backbone",
        "p": "Phase 6",
        "d": "Integral tanh could not encode the chain. Semi-analytic Ra-225/Ac-225 backbone + bounded NN correction; stiff substepped Ra-227/Ac-227 path.",
        "s": "completed",
        "co": "#8b5cf6",
        "e": 92,
        "r": [4, 6],
    },
    {
        "t": "Physics Calibration",
        "p": "Phase 7",
        "d": "1/v energy scaling, Ra-225 physics x5, ngamma cross-section fix, 30% empty-tank collocation, gradient balancing (physics vs data).",
        "s": "completed",
        "co": "#0d9488",
        "e": 95,
        "r": [5, 7],
    },
    {
        "t": "Kaggle Cloud",
        "p": "Phase 8",
        "d": "GPU training on Kaggle (600 physics pretrain + 3400 joint). Auto-sync scripts keep local and cloud weights aligned.",
        "s": "completed",
        "co": "#10b981",
        "e": 100,
        "r": [6, 8],
    },
    {
        "t": "Float64 Precision",
        "p": "Phase 9",
        "d": "PINN_FLOAT64=1 for flux spanning 12 orders of magnitude. Stabilized gradients and held-out metrics on A100 runs.",
        "s": "completed",
        "co": "#8b5cf6",
        "e": 98,
        "r": [7, 9],
    },
    {
        "t": "Graph & Sync Fix",
        "p": "Phase 10",
        "d": "Fixed parity plot IC columns (~184% false error -> ~5.5%). Curated poster figures; Kaggle zip staging (--dir-mode zip) preserved training CSV.",
        "s": "completed",
        "co": "#0ea5e9",
        "e": 100,
        "r": [8, 10],
    },
    {
        "t": "Clinical Portal",
        "p": "Phase 11",
        "d": "Streamlit triage portal: 10k-scenario grid, Physics Evidence tab (not curve-fit parity alone), validation gates, layman translations.",
        "s": "completed",
        "co": "#10b981",
        "e": 100,
        "r": [9, 11],
    },
    {
        "t": "v63 ISEF Ship",
        "p": "Phase 12",
        "d": "Official Kaggle v63 weights (sha256 7c21debe...): 6/6 PASS, ~4.5% held-out Ac-225. Rejected 12k ablation (7.3% held-out). Poster + demo ready.",
        "s": "completed",
        "co": "#34d399",
        "e": 100,
        "r": [10],
    },
]

ORBIT_PHASE_COUNT = len(ORBIT_NODES)


_TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=5, viewport-fit=cover">
<style>
*{margin:0;padding:0;box-sizing:border-box}
body,html{background:#000;overflow:hidden;font-family:system-ui,-apple-system,sans-serif;color:#fff}
#w{width:100%;height:700px;display:flex;flex-direction:row;
  background:radial-gradient(ellipse 80% 70% at 38% 42%,#101a33 0%,#060a18 55%,#000 100%);
  position:relative;overflow:hidden}
.cvWrap{flex:1;position:relative;min-width:180px;min-height:0}
canvas{display:block;width:100%;height:100%;touch-action:manipulation}
#tip{position:absolute;pointer-events:none;display:none;z-index:5;
  background:rgba(2,6,23,.92);border:1px solid rgba(148,163,184,.35);border-radius:8px;
  padding:6px 10px;font-size:11px;font-weight:600;color:#e2e8f0;white-space:nowrap;
  box-shadow:0 6px 18px rgba(0,0,0,.5);transform:translate(-50%,-130%)}
#ctrl{position:absolute;top:12px;left:12px;display:flex;gap:8px;z-index:6}
#ctrl button{background:rgba(15,23,42,.8);border:1px solid rgba(148,163,184,.28);color:#cbd5e1;
  width:36px;height:36px;border-radius:10px;cursor:pointer;font-size:14px;display:flex;
  align-items:center;justify-content:center;transition:all .15s;backdrop-filter:blur(6px)}
#ctrl button:hover{background:rgba(30,41,59,.95);border-color:rgba(56,189,248,.6);color:#fff;transform:translateY(-1px)}
#ctrl button.on{border-color:#38bdf8;color:#38bdf8;box-shadow:0 0 12px rgba(56,189,248,.4)}
#dock{width:min(320px,36vw);flex-shrink:0;border-left:1px solid rgba(148,163,184,.12);
  background:linear-gradient(180deg,rgba(15,23,42,.96) 0%,rgba(2,6,23,.99) 100%);
  display:flex;flex-direction:column;align-items:stretch;padding:16px 14px;overflow-y:auto;overflow-x:hidden}
#phaseCtr{font-size:10px;text-transform:uppercase;letter-spacing:1.4px;color:#64748b;margin-bottom:6px;text-align:center}
#progWrap{height:4px;background:rgba(255,255,255,.07);border-radius:99px;overflow:hidden;margin-bottom:12px}
#progBar{height:100%;width:0;border-radius:99px;background:linear-gradient(90deg,#38bdf8,#818cf8,#34d399);transition:width .5s ease}
#dockPh{flex:1;display:flex;align-items:center;justify-content:center;text-align:center;padding:12px;
  font-size:12px;color:rgba(148,163,184,.75);line-height:1.5;border:1px dashed rgba(148,163,184,.2);
  border-radius:12px;margin-top:4px}
#card{display:none;background:rgba(15,23,42,.9);border:1px solid rgba(148,163,184,.25);
  border-radius:16px;padding:18px;color:#fff;width:100%;backdrop-filter:blur(20px);
  box-shadow:0 12px 40px rgba(0,0,0,.45),inset 0 1px 0 rgba(255,255,255,.06);
  animation:cardIn .35s cubic-bezier(.2,.9,.3,1)}
@keyframes cardIn{from{opacity:0;transform:translateY(10px) scale(.98)}to{opacity:1;transform:none}}
#card .ph{font-size:10px;text-transform:uppercase;letter-spacing:1.6px;color:#94a3b8;margin-bottom:6px}
#card .tt{font-size:16px;font-weight:700;margin-bottom:8px;letter-spacing:-.3px;line-height:1.25}
#card .dd{font-size:12px;color:#cbd5e1;line-height:1.65}
#card .badge{margin-top:12px;font-size:10px;font-weight:700;padding:4px 12px;border-radius:999px;display:inline-block;color:#0f172a}
#card .ebar{margin-top:14px;border-top:1px solid rgba(255,255,255,.1);padding-top:12px}
#card .ebar .lbl{font-size:10px;color:#94a3b8;display:flex;justify-content:space-between;margin-bottom:6px}
#card .ebar .track{width:100%;height:5px;background:rgba(255,255,255,.08);border-radius:99px;overflow:hidden}
#card .ebar .fill{height:100%;border-radius:999px;background:linear-gradient(90deg,#38bdf8,#818cf8,#a78bfa);transition:width .45s ease}
#card .nav{display:flex;gap:8px;margin-top:14px}
#card .nav button{flex:1;background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.18);color:#e2e8f0;
  font-size:11px;padding:7px 8px;border-radius:8px;cursor:pointer;font-family:inherit;transition:all .15s}
#card .nav button:hover{background:rgba(56,189,248,.18);border-color:rgba(56,189,248,.5);color:#fff}
#card .conn{margin-top:12px;border-top:1px solid rgba(255,255,255,.1);padding-top:10px}
#card .conn .cl{font-size:10px;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px}
#card .conn button{background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.18);color:#e2e8f0;
  font-size:10px;padding:5px 10px;border-radius:6px;cursor:pointer;margin:3px 4px 3px 0;font-family:inherit;transition:all .15s}
#card .conn button:hover{background:rgba(255,255,255,.12);border-color:rgba(255,255,255,.35);color:#fff}
.hint{position:absolute;bottom:14px;left:50%;transform:translateX(-50%);font-size:11px;color:rgba(148,163,184,.5);pointer-events:none;white-space:nowrap;text-shadow:0 1px 8px #000}
@media (max-width:640px){
  body,html{overflow:auto;-webkit-overflow-scrolling:touch}
  #w{flex-direction:column;height:auto;min-height:620px}
  .cvWrap{height:min(52vh,420px);min-height:300px;flex:none}
  #dock{width:100%!important;border-left:none;border-top:1px solid rgba(148,163,184,.15);max-height:none;padding:12px}
  .hint{font-size:10px;white-space:normal;text-align:center;max-width:92%;bottom:8px}
}
</style></head><body>
<div id="w">
<div class="cvWrap">
  <canvas id="c"></canvas>
  <div id="tip"></div>
  <div id="ctrl">
    <button id="btnTour" title="Auto-tour through phases">&#9654;</button>
    <button id="btnReset" title="Reset view">&#8635;</button>
  </div>
  <div class="hint">Click node &bull; hover for title &bull; &larr; &rarr; keys &bull; empty space resumes spin</div>
</div>
<div id="dock">
  <div id="phaseCtr"></div>
  <div id="progWrap"><div id="progBar"></div></div>
  <div id="dockPh">Select any phase on the orbit, press &#9654; for a guided tour, or use &larr; &rarr; keys.</div>
  <div id="card">
    <div class="ph" id="cph"></div><div class="tt" id="ctt"></div><div class="dd" id="cdd"></div>
    <div class="badge" id="cbg"></div>
    <div class="ebar"><div class="lbl"><span>Maturity</span><span id="cen"></span></div>
    <div class="track"><div class="fill" id="cef"></div></div></div>
    <div class="nav"><button id="cprev">&larr; Prev</button><button id="cnext">Next &rarr;</button></div>
    <div class="conn" id="ccn"><div class="cl">Connected Nodes</div><div id="cbt"></div></div>
  </div>
</div>
</div>
<script>
var N=__NODES__;
var initSel=__INIT__;
var cv=document.getElementById("c"),ctx=cv.getContext("2d"),cvWrap=document.querySelector(".cvWrap");
var card=document.getElementById("card"),dockPh=document.getElementById("dockPh"),phaseCtr=document.getElementById("phaseCtr");
var progBar=document.getElementById("progBar"),tip=document.getElementById("tip");
var btnTour=document.getElementById("btnTour"),btnReset=document.getElementById("btnReset");
var W,H,cx,cy,ang=0,sel=-1,hov=-1,autoR=true;
var dpr=Math.min(window.devicePixelRatio||1,2.5);
var lastT=performance.now(),pulseT=0,cometT=0;
var HIT_R2=34*34;
var angVel=0.16,targetAngVel=0.16;
var snapActive=false,snapFrom=0,snapTo=0,snapT=0;
var tourOn=false,tourAcc=0,tourGap=3.2;
var stars=[];

function rand(a,b){return a+Math.random()*(b-a)}
function buildStars(){
  stars=[];var n=Math.round((W*H)/9000);
  for(var i=0;i<n;i++){stars.push({x:Math.random(),y:Math.random(),r:rand(.4,1.5),tw:rand(0,6.28),sp:rand(.4,1.4)});}
}
function sz(){
  W=cvWrap.clientWidth;H=cvWrap.clientHeight;
  cv.width=Math.max(1,Math.floor(W*dpr));cv.height=Math.max(1,Math.floor(H*dpr));
  cv.style.width=W+"px";cv.style.height=H+"px";
  ctx.setTransform(dpr,0,0,dpr,0,0);
  cx=W/2;cy=H/2;buildStars();
}
sz();window.addEventListener("resize",sz);

function getR(){return Math.min(W,H)*0.30}
function baseAngle(i){return i/N.length*Math.PI*2-Math.PI/2}
function npos(i){var R=getR();var a=baseAngle(i)+ang;return{x:cx+R*Math.cos(a),y:cy+R*Math.sin(a),a:a}}

// ease the selected node so it rotates to the top-front of the ring
function angleTo(i){
  var want=-Math.PI/2;            // top of circle
  var cur=baseAngle(i)+ang;
  var twoPi=Math.PI*2;
  var delta=((want-cur)%twoPi+twoPi)%twoPi; if(delta>Math.PI)delta-=twoPi;
  return ang+delta;
}
function startSnap(i){snapActive=true;snapFrom=ang;snapTo=angleTo(i);snapT=0;}

function updatePhaseCtr(i){
  if(i<0){phaseCtr.textContent="Phase - of "+N.length;progBar.style.width="0%";return;}
  phaseCtr.textContent="Phase "+(i+1)+" of "+N.length;
  progBar.style.width=(100*(i+1)/N.length)+"%";
}

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
    b.onclick=function(ev){ev.stopPropagation();selectNode(ri)};btns.appendChild(b);})
  }else{cn.style.display="none"}
  dockPh.style.display="none";card.style.display="block";
  // restart entrance animation
  card.style.animation="none";void card.offsetWidth;card.style.animation="";
  updatePhaseCtr(i);
}
function hideCard(){card.style.display="none";dockPh.style.display="flex";updatePhaseCtr(-1);}

function selectNode(i){
  sel=i;autoR=false;startSnap(i);showCard(i);
}
function toggleNode(i){
  if(sel===i){sel=-1;autoR=true;snapActive=false;hideCard();return;}
  selectNode(i);
}
function stepNode(delta){var next=sel<0?0:(sel+delta+N.length)%N.length;selectNode(next);}

function pickNode(mx,my){
  var best=-1,bestD=1e18;
  for(var i=0;i<N.length;i++){
    var q=npos(i),dx=mx-q.x,dy=my-q.y,d2=dx*dx+dy*dy;
    if(d2<HIT_R2&&d2<bestD){bestD=d2;best=i}
  }
  return best;
}

function setTour(on){
  tourOn=on;tourAcc=0;btnTour.classList.toggle("on",on);
  btnTour.innerHTML=on?"&#10073;&#10073;":"&#9654;";
  if(on){if(sel<0)selectNode(0);}
}

function drawStars(ts){
  for(var i=0;i<stars.length;i++){var s=stars[i];
    var a=0.35+0.45*Math.sin(ts*0.001*s.sp+s.tw);
    ctx.globalAlpha=Math.max(0,a);
    ctx.fillStyle="#cbd5e1";
    ctx.beginPath();ctx.arc(s.x*W,s.y*H,s.r,0,Math.PI*2);ctx.fill();
  }
  ctx.globalAlpha=1;
}
function drawOrbitRing(R){
  var grd=ctx.createLinearGradient(cx-R,cy,cx+R,cy);
  grd.addColorStop(0,"rgba(45,212,191,.12)");grd.addColorStop(.5,"rgba(129,140,248,.2)");grd.addColorStop(1,"rgba(45,212,191,.12)");
  ctx.strokeStyle=grd;ctx.lineWidth=1.5;ctx.lineCap="round";
  ctx.beginPath();ctx.arc(cx,cy,R,0,Math.PI*2);ctx.stroke();
  ctx.strokeStyle="rgba(255,255,255,.05)";ctx.lineWidth=1;
  ctx.beginPath();ctx.arc(cx,cy,R+3,0,Math.PI*2);ctx.stroke();
}
function drawProgressArc(R){
  if(sel<0)return;
  // arc from first node to the selected node along the ring
  var a0=baseAngle(0)+ang, a1=baseAngle(sel)+ang;
  ctx.strokeStyle="rgba(52,211,153,.55)";ctx.lineWidth=3.5;ctx.lineCap="round";
  ctx.shadowColor="rgba(52,211,153,.6)";ctx.shadowBlur=10;
  ctx.beginPath();ctx.arc(cx,cy,R,a0,a1);ctx.stroke();ctx.shadowBlur=0;
}
function drawComet(R,ts){
  var a=cometT;
  var x=cx+R*Math.cos(a),y=cy+R*Math.sin(a);
  for(var k=0;k<14;k++){
    var aa=a-k*0.045;
    var tx=cx+R*Math.cos(aa),ty=cy+R*Math.sin(aa);
    ctx.globalAlpha=(1-k/14)*0.5;
    ctx.fillStyle="#7dd3fc";
    ctx.beginPath();ctx.arc(tx,ty,Math.max(.5,2.4*(1-k/14)),0,Math.PI*2);ctx.fill();
  }
  ctx.globalAlpha=1;
  ctx.fillStyle="#e0f2fe";ctx.shadowColor="#38bdf8";ctx.shadowBlur=14;
  ctx.beginPath();ctx.arc(x,y,3,0,Math.PI*2);ctx.fill();ctx.shadowBlur=0;
}
function drawConnectors(){
  if(sel<0)return;var n=N[sel];if(!n.r)return;
  var p=npos(sel);
  n.r.forEach(function(ri){
    var q=npos(ri);
    var mx=(p.x+q.x)/2,my=(p.y+q.y)/2;
    // bow the control point toward the center for a nice arc
    var ccx=mx+(cx-mx)*0.35,ccy=my+(cy-my)*0.35;
    var grad=ctx.createLinearGradient(p.x,p.y,q.x,q.y);
    grad.addColorStop(0,n.co+"cc");grad.addColorStop(1,N[ri].co+"55");
    ctx.strokeStyle=grad;ctx.lineWidth=2;ctx.lineCap="round";
    ctx.beginPath();ctx.moveTo(p.x,p.y);ctx.quadraticCurveTo(ccx,ccy,q.x,q.y);ctx.stroke();
  });
}
function drawHub(ts){
  var ping=(Math.sin(ts*0.003)+1)*0.5;
  var R=getR();
  var g=ctx.createRadialGradient(cx,cy,0,cx,cy,R*0.5);
  g.addColorStop(0,"rgba(124,58,237,.32)");g.addColorStop(.3,"rgba(59,130,246,.16)");g.addColorStop(.6,"rgba(13,148,136,.07)");g.addColorStop(1,"transparent");
  ctx.fillStyle=g;ctx.beginPath();ctx.arc(cx,cy,R*0.5,0,Math.PI*2);ctx.fill();
  ctx.strokeStyle="rgba(255,255,255,"+(0.12+0.08*ping)+")";ctx.lineWidth=1.5;
  ctx.beginPath();ctx.arc(cx,cy,18+ping*14,0,Math.PI*2);ctx.stroke();
  if(sel>=0){
    var n=N[sel];
    ctx.fillStyle="#f8fafc";ctx.textAlign="center";
    ctx.font="700 22px system-ui,sans-serif";ctx.textBaseline="alphabetic";
    ctx.fillText((sel+1),cx,cy-2);
    ctx.fillStyle="rgba(203,213,225,.85)";ctx.font="600 9px system-ui,sans-serif";
    ctx.textBaseline="top";
    var lbl=n.t.length>20?n.t.slice(0,18)+"...":n.t;
    ctx.fillText(lbl.toUpperCase(),cx,cy+10);
  }else{
    ctx.fillStyle="rgba(255,255,255,.92)";ctx.shadowColor="rgba(56,189,248,.5)";ctx.shadowBlur=12;
    ctx.beginPath();ctx.arc(cx,cy,5.5,0,Math.PI*2);ctx.fill();ctx.shadowBlur=0;
  }
}

function frame(ts){
  ts=ts||performance.now();
  var dt=Math.min(0.033,(ts-lastT)/1000);lastT=ts;pulseT+=dt;cometT+=dt*0.9;
  if(cvWrap.clientWidth!==W||cvWrap.clientHeight!==H)sz();

  // rotation update: snap easing, or free spin with eased velocity
  if(snapActive){
    snapT+=dt/0.6; if(snapT>=1){snapT=1;snapActive=false;}
    var e=1-Math.pow(1-snapT,3);
    ang=snapFrom+(snapTo-snapFrom)*e;
  }else{
    targetAngVel=autoR?0.16:0.0;
    angVel+=(targetAngVel-angVel)*Math.min(1,dt*4);
    ang+=dt*angVel;
  }

  // auto-tour stepping
  if(tourOn){
    tourAcc+=dt;
    if(tourAcc>=tourGap){tourAcc=0;stepNode(1);if(sel===N.length-1){/* loop */}}
  }

  ctx.clearRect(0,0,W,H);
  drawStars(ts);
  var R=getR();
  drawOrbitRing(R);
  drawProgressArc(R);
  drawComet(R,ts);
  drawConnectors();

  for(var i=0;i<N.length;i++){
    var q=npos(i),n=N[i];
    var isSel=i===sel, isRel=sel>=0&&N[sel].r&&N[sel].r.indexOf(i)>=0, isHov=i===hov;
    var breathe=autoR?1+0.06*Math.sin(pulseT*2.2+i*0.7):1;
    var r=(isSel?16:isRel?10:isHov?11:8)*breathe;

    // spoke
    var grad=ctx.createLinearGradient(cx,cy,q.x,q.y);
    grad.addColorStop(0,"rgba(255,255,255,.04)");grad.addColorStop(1,isSel?"rgba(255,255,255,.12)":isRel?"rgba(255,255,255,.06)":"rgba(255,255,255,.02)");
    ctx.strokeStyle=grad;ctx.lineWidth=isSel?2:1;ctx.lineCap="round";
    ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(q.x,q.y);ctx.stroke();

    if(isSel||isRel||isHov){
      var gg=ctx.createRadialGradient(q.x,q.y,0,q.x,q.y,46);
      gg.addColorStop(0,n.co+(isSel?"66":"30"));gg.addColorStop(1,"transparent");
      ctx.fillStyle=gg;ctx.beginPath();ctx.arc(q.x,q.y,46,0,Math.PI*2);ctx.fill();
    }
    // node body
    ctx.fillStyle=isSel?"#0b1220":"#0b1220";
    ctx.beginPath();ctx.arc(q.x,q.y,r+2,0,Math.PI*2);ctx.fill();
    ctx.fillStyle=isSel?"#ffffff":n.co;
    ctx.beginPath();ctx.arc(q.x,q.y,r,0,Math.PI*2);ctx.fill();
    ctx.strokeStyle=isSel?"#ffffff":isRel||isHov?"rgba(255,255,255,.9)":"rgba(255,255,255,.35)";
    ctx.shadowColor=n.co;ctx.shadowBlur=isSel?14:isHov?8:0;
    ctx.lineWidth=isSel?2.4:isRel||isHov?1.8:1.2;
    ctx.beginPath();ctx.arc(q.x,q.y,r,0,Math.PI*2);ctx.stroke();ctx.shadowBlur=0;

    // sequence number inside node
    ctx.fillStyle=isSel?n.co:"#0b1220";
    ctx.font="700 "+(isSel?12:9)+"px system-ui,sans-serif";
    ctx.textAlign="center";ctx.textBaseline="middle";
    ctx.fillText((i+1),q.x,q.y+0.5);

    // outside label (skip if hub already shows it for selected)
    if(W>420&&!isSel){
      ctx.fillStyle=isRel?"rgba(248,250,252,.9)":isHov?"#fff":"rgba(203,213,225,.78)";
      ctx.font=(isRel||isHov?"600 11px":"500 10px")+" system-ui,sans-serif";
      ctx.textBaseline="top";
      var lbl=n.t.length>18?n.t.slice(0,16)+"...":n.t;
      ctx.fillText(lbl,q.x,q.y+r+9);
    }
  }
  drawHub(ts);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

function evtPos(e){var rect=cv.getBoundingClientRect();return{mx:e.clientX-rect.left,my:e.clientY-rect.top};}
cv.addEventListener("click",function(e){
  var p=evtPos(e),hit=pickNode(p.mx,p.my);
  if(hit>=0)toggleNode(hit);else{sel=-1;autoR=true;snapActive=false;hideCard();}
});
cv.addEventListener("mousemove",function(e){
  var p=evtPos(e),hit=pickNode(p.mx,p.my);hov=hit;
  if(hit>=0){
    cv.style.cursor="pointer";
    var q=npos(hit);
    tip.style.display="block";tip.textContent=N[hit].p+" - "+N[hit].t;
    tip.style.left=q.x+"px";tip.style.top=q.y+"px";
  }else{cv.style.cursor="default";tip.style.display="none";}
});
cv.addEventListener("mouseleave",function(){hov=-1;tip.style.display="none";});

document.getElementById("cprev").onclick=function(ev){ev.stopPropagation();stepNode(-1);};
document.getElementById("cnext").onclick=function(ev){ev.stopPropagation();stepNode(1);};
btnTour.onclick=function(){setTour(!tourOn);};
btnReset.onclick=function(){sel=-1;autoR=true;snapActive=false;setTour(false);hideCard();};

window.addEventListener("keydown",function(e){
  if(e.key==="ArrowLeft"){e.preventDefault();stepNode(-1);}
  else if(e.key==="ArrowRight"){e.preventDefault();stepNode(1);}
  else if(e.key===" "){e.preventDefault();setTour(!tourOn);}
  else if(e.key==="Escape"){sel=-1;autoR=true;snapActive=false;setTour(false);hideCard();}
});

updatePhaseCtr(-1);
if(initSel>=0&&initSel<N.length){selectNode(initSel);}
</script></body></html>"""


def build_orbital_timeline_html(initial_index: int = -1) -> str:
    """Return self-contained HTML for the spinning orbit timeline."""
    init = max(-1, min(int(initial_index), ORBIT_PHASE_COUNT - 1))
    return _TEMPLATE.replace("__NODES__", json.dumps(ORBIT_NODES)).replace("__INIT__", str(init))
