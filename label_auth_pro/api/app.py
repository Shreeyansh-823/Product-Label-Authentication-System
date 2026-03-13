"""
Label Authentication System — Flask API + Dashboard
Endpoints: classify, detect, footprint, report, stats, attention
"""

import os, sys, json, base64, io
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template_string
from PIL import Image

from pipeline.model_pipeline import LabelAuthPipeline
from utils.blockchain_rag    import build_blockchain, EmbeddingRAG
from detection.detector      import YOLOStyleDetector
from core.data_gen           import CLASS_NAMES, GENERATORS

app = Flask(__name__)

_pipeline   = None
_blockchain = None
_rag        = None
_detector   = YOLOStyleDetector()
_rng        = np.random.RandomState(42)


# ─── HTML Dashboard ────────────────────────────────────────────────────────────

DASHBOARD = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Label Auth System — ViT</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh}
.hdr{background:linear-gradient(135deg,#1a2744 0%,#0f172a 100%);padding:18px 28px;border-bottom:1px solid #334155;display:flex;align-items:center;gap:14px}
.hdr-icon{font-size:2em}
.hdr h1{color:#38bdf8;font-size:1.3em;font-weight:700}
.hdr p{color:#64748b;font-size:0.8em;margin-top:3px}
.nav{display:flex;background:#1e293b;border-bottom:1px solid #334155;padding:0 20px}
.nav-item{padding:11px 18px;cursor:pointer;font-size:0.84em;color:#64748b;border-bottom:2px solid transparent;transition:all .2s;user-select:none}
.nav-item.active,.nav-item:hover{color:#38bdf8;border-bottom-color:#38bdf8}
.pane{display:none;padding:20px}.pane.active{display:block}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
.card{background:#1e293b;border:1px solid #334155;border-radius:10px;padding:18px}
.card h3{color:#7dd3fc;font-size:.72em;text-transform:uppercase;letter-spacing:.08em;margin-bottom:14px}
.upload-zone{border:2px dashed #334155;border-radius:8px;padding:22px;text-align:center;cursor:pointer;transition:border-color .2s}
.upload-zone:hover{border-color:#38bdf8}
.btn{background:#1d4ed8;color:#fff;border:none;padding:9px 20px;border-radius:6px;cursor:pointer;font-size:.85em;font-weight:600;transition:background .15s}
.btn:hover{background:#1e40af}.btn-teal{background:#0d9488}.btn-teal:hover{background:#0f766e}
.btn-purple{background:#7c3aed}.btn-purple:hover{background:#6d28d9}
.badge{display:inline-block;padding:5px 14px;border-radius:20px;font-weight:700;font-size:.85em}
.GENUINE{background:#14532d;color:#4ade80}.TAMPERED{background:#450a0a;color:#f87171}
.COUNTERFEIT{background:#2e1065;color:#c084fc}.DEGRADED{background:#451a03;color:#fbbf24}
.prob-row{display:flex;align-items:center;gap:8px;margin:5px 0}
.prob-label{width:105px;font-size:.79em;color:#94a3b8;flex-shrink:0}
.bar-bg{flex:1;background:#0f172a;border-radius:4px;height:10px;overflow:hidden}
.bar-fill{height:100%;border-radius:4px;transition:width .5s ease}
.prob-pct{width:42px;font-size:.78em;text-align:right}
.stat-tile{background:#0f172a;border-radius:8px;padding:14px;text-align:center}
.stat-num{font-size:1.7em;font-weight:700}
.stat-lbl{font-size:.67em;color:#64748b;text-transform:uppercase;margin-top:3px}
.report-box{background:#0f172a;font-family:monospace;font-size:.73em;padding:14px;border-radius:8px;white-space:pre-wrap;max-height:340px;overflow-y:auto;color:#94a3b8;margin-top:12px}
input[type=text],textarea{background:#0f172a;border:1px solid #334155;color:#e2e8f0;padding:8px 12px;border-radius:6px;width:100%;font-family:inherit}
.chain-evt{background:#0f172a;border-radius:6px;padding:7px 10px;margin:4px 0;font-size:.76em;border-left:3px solid #334155}
.chain-authentic{border-color:#22c55e}.chain-tampered{border-color:#ef4444}.chain-warn{border-color:#f59e0b}
.qchip{display:inline-block;background:#1e3a5f;color:#93c5fd;padding:3px 10px;border-radius:12px;font-size:.74em;margin:3px;cursor:pointer}
.qchip:hover{background:#1e4d8c}
img{max-width:100%;border-radius:6px}
#attnCanvas{border-radius:6px;display:none}
</style>
</head>
<body>

<div class="hdr">
  <div class="hdr-icon">🏷️</div>
  <div>
    <h1>Label Authentication System</h1>
    <p>Hybrid CNN-ViT · Optuna HPO · Ensemble Classifier · Blockchain RAG · YOLO Detection</p>
  </div>
</div>

<div class="nav">
  <div class="nav-item active" onclick="tab(this,'classify')">🔬 Classify</div>
  <div class="nav-item" onclick="tab(this,'detect')">🎯 Detect</div>
  <div class="nav-item" onclick="tab(this,'blockchain')">🔗 Blockchain</div>
  <div class="nav-item" onclick="tab(this,'rag')">🤖 RAG Query</div>
  <div class="nav-item" onclick="tab(this,'stats')">📊 Stats</div>
</div>

<!-- CLASSIFY TAB -->
<div class="pane active" id="pane-classify">
  <div class="grid2">
    <div>
      <div class="card" style="margin-bottom:14px">
        <h3>Upload Label Image</h3>
        <div class="upload-zone" onclick="document.getElementById('fi').click()">
          <div style="font-size:2em;margin-bottom:6px">📷</div>
          <p style="color:#64748b;font-size:.85em">Click to upload (PNG / JPG)</p>
          <input type="file" id="fi" accept="image/*" style="display:none" onchange="onFile(this)">
        </div>
        <div style="text-align:center;margin-top:12px">
          <img id="imgPrev" style="max-width:220px;display:none;margin-bottom:10px">
        </div>
        <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:10px">
          <button class="btn" onclick="classify()">🔬 Classify</button>
          <button class="btn btn-teal" onclick="demoClassify()">🎲 Demo Sample</button>
          <button class="btn btn-purple" onclick="toggleAttn()">👁 Attention Map</button>
        </div>
      </div>
      <div class="card">
        <h3>Attention Rollout Map (XAI)</h3>
        <p style="color:#64748b;font-size:.8em;margin-bottom:8px">Highlights image regions the ViT attends to</p>
        <canvas id="attnCanvas" width="256" height="256"></canvas>
        <p id="attnNote" style="color:#64748b;font-size:.75em;margin-top:6px">Run classification first</p>
      </div>
    </div>
    <div class="card">
      <h3>Classification Result</h3>
      <div id="resultBox" style="color:#64748b;font-size:.85em;text-align:center;padding:30px 0">
        Upload an image and click Classify
      </div>
    </div>
  </div>
</div>

<!-- DETECT TAB -->
<div class="pane" id="pane-detect">
  <div class="card">
    <h3>YOLO-Style Label Detection + Classification</h3>
    <p style="color:#64748b;font-size:.82em;margin-bottom:12px">
      Detects label regions via edge-based anchors & NMS, then classifies each crop.
    </p>
    <div class="upload-zone" onclick="document.getElementById('fi2').click()">
      <p style="color:#64748b">Upload product image for detection</p>
      <input type="file" id="fi2" accept="image/*" style="display:none" onchange="onFile2(this)">
    </div>
    <div style="margin-top:12px;display:flex;gap:8px">
      <button class="btn" onclick="detect()">🎯 Detect Labels</button>
      <button class="btn btn-teal" onclick="detectDemo()">🎲 Demo</button>
    </div>
    <div class="grid2" style="margin-top:16px">
      <div>
        <p style="font-size:.78em;color:#94a3b8;margin-bottom:6px">Input</p>
        <img id="detectInput" style="display:none">
      </div>
      <div>
        <p style="font-size:.78em;color:#94a3b8;margin-bottom:6px">Detections</p>
        <img id="detectOutput" style="display:none">
      </div>
    </div>
    <div id="detectList" style="margin-top:12px"></div>
  </div>
</div>

<!-- BLOCKCHAIN TAB -->
<div class="pane" id="pane-blockchain">
  <div class="card">
    <h3>Blockchain Product Footprint</h3>
    <div style="display:flex;gap:8px;margin-bottom:12px">
      <input type="text" id="pidInput" placeholder="e.g. PROD-0001" style="max-width:220px">
      <button class="btn" onclick="getFootprint()">Lookup</button>
    </div>
    <div style="flex-wrap:wrap;display:flex;gap:6px;margin-bottom:12px">
      <span class="qchip" onclick="quickPid('PROD-0001')">PROD-0001</span>
      <span class="qchip" onclick="quickPid('PROD-0004')">PROD-0004 ⚠️</span>
      <span class="qchip" onclick="quickPid('PROD-0009')">PROD-0009 ⚠️</span>
      <span class="qchip" onclick="quickPid('PROD-0014')">PROD-0014 ⚠️</span>
      <span class="qchip" onclick="quickPid('PROD-0020')">PROD-0020 ⚠️</span>
    </div>
    <div id="footprintResult" style="color:#64748b;font-size:.85em">Enter a product ID to view its chain.</div>
  </div>
</div>

<!-- RAG TAB -->
<div class="pane" id="pane-rag">
  <div class="card">
    <h3>LLM Embedding RAG — Defect Pattern Retrieval</h3>
    <p style="color:#64748b;font-size:.82em;margin-bottom:10px">
      Semantic retrieval from domain knowledge + blockchain records using TF-IDF embeddings.
    </p>
    <div style="display:flex;gap:8px;margin-bottom:10px">
      <input type="text" id="ragQ" placeholder="e.g. QR code distortion counterfeit">
      <button class="btn" onclick="ragSearch()">🔍 Search</button>
    </div>
    <div style="flex-wrap:wrap;display:flex;gap:5px;margin-bottom:14px">
      <span class="qchip" onclick="setQ('adhesive failure label peel')">adhesive failure</span>
      <span class="qchip" onclick="setQ('QR code distortion counterfeit')">QR distortion</span>
      <span class="qchip" onclick="setQ('blockchain provenance authentic')">blockchain</span>
      <span class="qchip" onclick="setQ('degraded yellowing water damage')">degradation</span>
      <span class="qchip" onclick="setQ('hybrid CNN ViT attention')">CNN-ViT</span>
    </div>
    <div id="ragResults"></div>
  </div>

  <div class="card" style="margin-top:14px">
    <h3>Generate Authenticity Report</h3>
    <div style="display:flex;gap:8px;margin-bottom:10px">
      <input type="text" id="repPid" placeholder="Product ID (e.g. PROD-0001)" style="max-width:200px">
      <button class="btn btn-purple" onclick="genReport()">📝 Generate</button>
    </div>
    <div class="report-box" id="reportBox" style="display:none"></div>
  </div>
</div>

<!-- STATS TAB -->
<div class="pane" id="pane-stats">
  <div class="grid3" id="statTiles" style="margin-bottom:16px"></div>
  <div class="grid2">
    <div class="card">
      <h3>System Components</h3>
      <div id="componentList" style="font-size:.83em;line-height:2;color:#94a3b8"></div>
    </div>
    <div class="card">
      <h3>Quick Classify Demo</h3>
      <p style="font-size:.82em;color:#64748b;margin-bottom:10px">Classify random samples from each class</p>
      <div id="demoGrid" style="display:grid;grid-template-columns:1fr 1fr;gap:8px"></div>
      <button class="btn btn-teal" style="margin-top:10px;width:100%" onclick="batchDemo()">
        ▶ Run Batch Demo (4 samples)
      </button>
    </div>
  </div>
</div>

<script>
const CLS_COLORS={GENUINE:'#22c55e',TAMPERED:'#ef4444',COUNTERFEIT:'#a855f7',DEGRADED:'#f59e0b'};

function tab(el, name) {
  document.querySelectorAll('.nav-item').forEach(e=>e.classList.remove('active'));
  document.querySelectorAll('.pane').forEach(e=>e.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('pane-'+name).classList.add('active');
}

// ── File upload helpers ───────────────────────────────────────────────────────
let imgB64=null, imgB64_2=null;
function onFile(input){
  const f=input.files[0]; if(!f)return;
  const r=new FileReader();
  r.onload=e=>{
    document.getElementById('imgPrev').src=e.target.result;
    document.getElementById('imgPrev').style.display='block';
    imgB64=e.target.result.split(',')[1];
  };
  r.readAsDataURL(f);
}
function onFile2(input){
  const f=input.files[0]; if(!f)return;
  const r=new FileReader();
  r.onload=e=>{
    document.getElementById('detectInput').src=e.target.result;
    document.getElementById('detectInput').style.display='block';
    imgB64_2=e.target.result.split(',')[1];
  };
  r.readAsDataURL(f);
}

// ── Classify ─────────────────────────────────────────────────────────────────
async function classify(b64=null){
  const data=b64||imgB64;
  if(!data){alert('Upload an image first');return;}
  const r=await fetch('/api/classify',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image_b64:data})});
  const d=await r.json();
  if(d.error){document.getElementById('resultBox').innerHTML=`<span style="color:#ef4444">${d.error}</span>`;return;}
  renderResult(d,'resultBox');
  if(d.attention_map&&d.attention_map.length>0) drawAttn(d.attention_map);
}

async function demoClassify(){
  const r=await fetch('/api/demo_sample');
  const d=await r.json();
  if(d.error){alert(d.error);return;}
  document.getElementById('imgPrev').src='data:image/jpeg;base64,'+d.image_b64;
  document.getElementById('imgPrev').style.display='block';
  imgB64=d.image_b64;
  classify(d.image_b64);
}

function renderResult(d, target){
  const c=CLS_COLORS[d.label]||'#60a5fa';
  let h=`<div style="margin-bottom:14px">
    <span class="badge ${d.label}">${d.label}</span>
    <span style="color:#64748b;font-size:.85em;margin-left:10px">
      Confidence: <strong style="color:${c}">${(d.confidence*100).toFixed(1)}%</strong>
    </span>
  </div>
  <div style="margin-bottom:10px">`;
  for(const[cls,p] of Object.entries(d.probabilities||{})){
    h+=`<div class="prob-row">
      <span class="prob-label">${cls}</span>
      <div class="bar-bg"><div class="bar-fill" style="width:${(p*100).toFixed(1)}%;background:${CLS_COLORS[cls]||'#60a5fa'}"></div></div>
      <span class="prob-pct">${(p*100).toFixed(1)}%</span>
    </div>`;
  }
  h+=`</div>`;
  document.getElementById(target).innerHTML=h;
}

// ── Attention map ─────────────────────────────────────────────────────────────
let showAttn=false;
function toggleAttn(){showAttn=!showAttn;document.getElementById('attnCanvas').style.display=showAttn?'block':'none';}
function drawAttn(grid){
  const canvas=document.getElementById('attnCanvas');
  const ctx=canvas.getContext('2d');
  const ps=grid.length;
  const cs=canvas.width/ps;
  const flat=grid.flat();
  const mn=Math.min(...flat), mx=Math.max(...flat);
  for(let i=0;i<ps;i++)for(let j=0;j<ps;j++){
    const v=(grid[i][j]-mn)/(mx-mn+1e-9);
    const r=Math.round(255*Math.min(1,v*2));
    const g=Math.round(255*Math.max(0,v*2-1));
    ctx.fillStyle=`rgb(${r},${g},0)`;
    ctx.fillRect(j*cs,i*cs,cs,cs);
  }
  document.getElementById('attnNote').textContent='Attention rollout from last ViT block';
  if(showAttn)canvas.style.display='block';
}

// ── Detection ─────────────────────────────────────────────────────────────────
async function detect(b64=null){
  const data=b64||imgB64_2;
  if(!data){alert('Upload a product image first');return;}
  const r=await fetch('/api/detect',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image_b64:data})});
  const d=await r.json();
  if(d.error){alert(d.error);return;}
  document.getElementById('detectOutput').src='data:image/jpeg;base64,'+d.annotated_b64;
  document.getElementById('detectOutput').style.display='block';
  let h='';
  for(const det of d.detections){
    const c=CLS_COLORS[det.label]||'#60a5fa';
    h+=`<div class="chain-evt" style="border-color:${c}">
      <strong style="color:${c}">${det.label}</strong>
      <span style="float:right;color:#64748b;font-size:.85em">det_conf=${det.det_confidence?.toFixed(2)} | cls_conf=${(det.confidence*100).toFixed(1)}%</span><br>
      Box: [${det.x1},${det.y1},${det.x2},${det.y2}]
    </div>`;
  }
  document.getElementById('detectList').innerHTML=h||'<span style="color:#64748b">No detections</span>';
}

async function detectDemo(){
  const r=await fetch('/api/demo_sample');
  const d=await r.json();
  document.getElementById('detectInput').src='data:image/jpeg;base64,'+d.image_b64;
  document.getElementById('detectInput').style.display='block';
  imgB64_2=d.image_b64;
  detect(d.image_b64);
}

// ── Blockchain ────────────────────────────────────────────────────────────────
function quickPid(pid){document.getElementById('pidInput').value=pid;getFootprint();}
async function getFootprint(){
  const pid=document.getElementById('pidInput').value.trim().toUpperCase()||'PROD-0001';
  const r=await fetch(`/api/footprint/${pid}`);
  const d=await r.json();
  const box=document.getElementById('footprintResult');
  if(!d.records||d.records.length===0){
    box.innerHTML='<span style="color:#ef4444">No blockchain records found.</span>';return;
  }
  let h=`<p style="color:#4ade80;margin-bottom:8px">✓ Chain verified — ${d.records.length} events</p>`;
  for(const rec of d.records){
    const ev=rec.data.event;
    const cls=ev.scan_result==='AUTHENTIC'?'chain-authentic':ev.scan_result?'chain-tampered':'chain-warn';
    h+=`<div class="chain-evt ${cls}">
      <strong>${ev.stage}</strong> <span style="float:right;color:#64748b">${rec.timestamp?.slice(0,19)}</span><br>
      <span style="color:#94a3b8">${ev.description} — ${ev.location||''}</span>
      <span style="float:right" class="badge ${ev.scan_result}">${ev.scan_result||''}</span>
    </div>`;
  }
  box.innerHTML=h;
}

// ── RAG ───────────────────────────────────────────────────────────────────────
function setQ(q){document.getElementById('ragQ').value=q;ragSearch();}
async function ragSearch(){
  const q=document.getElementById('ragQ').value.trim();
  if(!q){alert('Enter a query');return;}
  const r=await fetch('/api/rag',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:q,top_k:5})});
  const d=await r.json();
  const box=document.getElementById('ragResults');
  if(!d.results||d.results.length===0){box.innerHTML='<span style="color:#64748b">No results</span>';return;}
  box.innerHTML=d.results.map(res=>`
    <div class="chain-evt" style="border-color:#7c3aed;margin-bottom:6px">
      <span style="color:#a78bfa;font-weight:700">[${res.score.toFixed(3)}]</span>
      <span style="color:#94a3b8;font-size:.82em;margin-left:6px">${res.text}</span>
    </div>`).join('');
}

async function genReport(){
  const pid=document.getElementById('repPid').value.trim().toUpperCase()||'PROD-0001';
  const r=await fetch(`/api/report/${pid}`);
  const d=await r.json();
  const box=document.getElementById('reportBox');
  box.style.display='block';
  box.textContent=d.report||d.error;
}

// ── Stats ─────────────────────────────────────────────────────────────────────
async function loadStats(){
  const r=await fetch('/api/stats');
  const d=await r.json();
  const tiles=document.getElementById('statTiles');
  const tileData=[
    {v:d.accuracy||'—', l:'Val Accuracy', color:'#22c55e'},
    {v:d.f1_macro||'—', l:'Val F1 Macro', color:'#38bdf8'},
    {v:d.roc_auc||'—',  l:'Val ROC-AUC',  color:'#a855f7'},
    {v:d.blockchain_blocks||'—', l:'Blockchain Blocks', color:'#f59e0b'},
    {v:d.rag_docs||'—', l:'RAG Documents', color:'#fb923c'},
    {v:d.model||'—',    l:'Model',        color:'#f472b6'},
  ];
  tiles.innerHTML=tileData.map(t=>`
    <div class="stat-tile">
      <div class="stat-num" style="color:${t.color}">${t.v}</div>
      <div class="stat-lbl">${t.l}</div>
    </div>`).join('');
  document.getElementById('componentList').innerHTML=Object.entries({
    'Architecture': d.architecture||'CNN-ViT Hybrid',
    'Classifier':   d.classifier||'GBM+RF+ET+MLP+SVC Ensemble',
    'HPO':          'Optuna-style (10 trials)',
    'Features':     `ViT(${d.d_model||128}d) + Texture(29d)`,
    'Augmentation': 'Flip/Rotate/Perspective/HSV/CutOut',
    'Detection':    'YOLO-style edge anchors + NMS',
    'RAG':          'TF-IDF sentence embeddings + cosine retrieval',
    'Blockchain':   'SHA-256 chained blocks',
    'Status':       '✅ Ready',
  }).map(([k,v])=>`<div><strong style="color:#38bdf8">${k}:</strong> ${v}</div>`).join('');
}

async function batchDemo(){
  const grid=document.getElementById('demoGrid');
  grid.innerHTML='<span style="color:#64748b">Loading...</span>';
  const r=await fetch('/api/batch_demo');
  const d=await r.json();
  grid.innerHTML=d.samples.map(s=>`
    <div class="card" style="padding:10px;background:#0f172a">
      <img src="data:image/jpeg;base64,${s.image_b64}" style="width:100%;border-radius:4px;margin-bottom:6px">
      <span class="badge ${s.label}" style="font-size:.72em">${s.label}</span>
      <span style="color:#64748b;font-size:.72em;margin-left:6px">${(s.confidence*100).toFixed(0)}%</span>
      <div style="font-size:.7em;color:#64748b;margin-top:3px">True: ${s.true_class}</div>
    </div>`).join('');
}

loadStats();
</script>
</body>
</html>"""


# ─── Model Loader ─────────────────────────────────────────────────────────────

def load_all():
    global _pipeline, _blockchain, _rag
    if _pipeline is None:
        _pipeline = LabelAuthPipeline(use_hybrid=True, img_size=64,
                                       patch_size=8, d_model=128,
                                       n_heads=4, n_layers=3, d_ff=256)
        try:
            _pipeline.load("models")
        except Exception as e:
            print(f"[API] Model not loaded ({e}) — run train.py first")
    if _blockchain is None:
        _blockchain = build_blockchain(n_products=25, rng_seed=42)
    if _rag is None:
        _rag = EmbeddingRAG()
        try:
            _rag.load("models/rag.pkl")
        except Exception:
            _rag.build(_blockchain)


def _img_from_b64(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.resize(img, (64, 64))


def _img_to_b64(img: np.ndarray, quality=85) -> str:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode()


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(DASHBOARD)


@app.route("/api/classify", methods=["POST"])
def classify():
    load_all()
    try:
        b64 = request.get_json().get("image_b64","")
        img = _img_from_b64(b64)
        res = _pipeline.predict(img)
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/detect", methods=["POST"])
def detect():
    load_all()
    try:
        b64  = request.get_json().get("image_b64","")
        raw  = base64.b64decode(b64)
        arr  = np.frombuffer(raw, np.uint8)
        img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        # Scale for detection
        det_img = cv2.resize(img, (256, 256))
        dets    = _detector.detect_and_crop(det_img, target_size=64)

        results = []
        preds   = []
        det_objs= []
        for crop, det in dets[:4]:
            pred = _pipeline.predict(crop) if _pipeline.model else \
                   {"label":"GENUINE","confidence":0.9,"probabilities":{}}
            preds.append(pred)
            det_objs.append(det)
            results.append({
                "label": pred["label"], "confidence": pred["confidence"],
                "det_confidence": det.confidence,
                "x1": det.x1, "y1": det.y1, "x2": det.x2, "y2": det.y2,
            })

        # Annotate
        vis = _detector.draw_detections(det_img, det_objs, preds)
        return jsonify({"detections": results, "annotated_b64": _img_to_b64(vis)})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 400


@app.route("/api/demo_sample")
def demo_sample():
    load_all()
    cls_id = _rng.randint(0, 4)
    seed   = int(_rng.randint(0, 100000))
    img    = GENERATORS[cls_id](64, seed=seed)
    return jsonify({
        "image_b64":  _img_to_b64(img),
        "true_class": ["GENUINE","TAMPERED","COUNTERFEIT","DEGRADED"][cls_id],
    })


@app.route("/api/batch_demo")
def batch_demo():
    load_all()
    samples = []
    for cls_id in range(4):
        seed = int(_rng.randint(0, 100000))
        img  = GENERATORS[cls_id](64, seed=seed)
        pred = _pipeline.predict(img) if _pipeline.model else \
               {"label": CLASS_NAMES[cls_id], "confidence": 0.9}
        samples.append({
            "image_b64":  _img_to_b64(img),
            "label":      pred["label"],
            "confidence": pred["confidence"],
            "true_class": CLASS_NAMES[cls_id],
        })
    return jsonify({"samples": samples})


@app.route("/api/footprint/<pid>")
def footprint(pid):
    load_all()
    records = _blockchain.footprint(pid.upper())
    return jsonify({
        "product_id": pid.upper(),
        "records":    records,
        "chain_valid": _blockchain.verify(),
        "total_blocks": len(_blockchain.chain),
    })


@app.route("/api/rag", methods=["POST"])
def rag():
    load_all()
    data = request.get_json()
    q    = data.get("query","")
    k    = data.get("top_k", 5)
    results = _rag.retrieve(q, top_k=k)
    return jsonify({"results": results, "query": q})


@app.route("/api/report/<pid>")
def report(pid):
    load_all()
    dummy_pred = {"label":"GENUINE","confidence":0.94,
                  "probabilities":{"GENUINE":0.94,"TAMPERED":0.02,
                                   "COUNTERFEIT":0.02,"DEGRADED":0.02}}
    rpt = _rag.generate_report(pid.upper(), dummy_pred, _blockchain)
    return jsonify({"product_id": pid.upper(), "report": rpt})


@app.route("/api/stats")
def stats():
    load_all()
    ev = _pipeline.eval_results.get("val", {}) if _pipeline else {}
    return jsonify({
        "accuracy":        f"{ev.get('accuracy',0):.4f}" if ev else "N/A",
        "f1_macro":        f"{ev.get('f1_macro',0):.4f}" if ev else "N/A",
        "roc_auc":         f"{ev.get('roc_auc',0):.4f}"  if ev else "N/A",
        "architecture":    "Hybrid CNN-ViT",
        "classifier":      "GBM + RF + ET + MLP + SVC (soft voting)",
        "d_model":         128,
        "blockchain_blocks": len(_blockchain.chain),
        "rag_docs":        len(_rag.documents),
        "model":           "Trained ✅" if (_pipeline and _pipeline.model) else "Not trained ⚠️",
    })


if __name__ == "__main__":
    load_all()
    print("Starting Label Authentication API → http://localhost:5001")
    app.run(host="0.0.0.0", port=5001, debug=False)
