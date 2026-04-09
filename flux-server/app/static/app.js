/* ═══════════════════════════════════════════════════════════
   Neural Creation Studio — Multi-Modal AI Frontend
   Image · Video · Music · Animation
   ═══════════════════════════════════════════════════════════ */

const RESOLUTION_PRESETS = [
  { label: "Square",    width: 1024, height: 1024 },
  { label: "Portrait",  width: 896,  height: 1152 },
  { label: "Landscape", width: 1152, height: 896  },
  { label: "Wide",      width: 1344, height: 768  },
];

const VIDEO_FRAME_PRESETS = [
  { label: "1s",  frames: 16 },
  { label: "2s",  frames: 33 },
  { label: "3s",  frames: 49 },
  { label: "5s",  frames: 81 },
];

const GENRE_PILLS = [
  "Pop","Rock","Hip Hop","Electronic","Jazz","R&B",
  "Classical","Lo-fi","Ambient","Cinematic","Metal","Country",
];

const state = {
  activeTab: "image",
  models: [], videoModels: [], musicModels: [], animModels: [],
  health: null, categories: [],
  // Image
  history: [], loading: false, error: "", result: null, lightbox: null,
  loraUploadStatus: null,
  form: {
    prompt: "", negative_prompt: "", model_name: "flux-1-dev",
    width: 1024, height: 1024, num_inference_steps: 28,
    guidance_scale: 3.5, seed: "", lora_name: "None", lora_scale: 0.85,
  },
  loras: ["None"],
  // Video
  videoForm: {
    prompt: "", negative_prompt: "", model_name: "ltx-video",
    resolution: "480p", num_frames: 33, fps: 16,
    guidance_scale: 5.0, num_inference_steps: 30, seed: "",
    source_image_b64: null, lora_name: "None", lora_scale: 1.0,
  },
  videoSourceName: "",
  videoLoras: ["None"],
  videoLoraUploadStatus: null,
  // Music
  musicForm: {
    prompt: "", model_name: "audioldm2", duration_seconds: 30,
    lyrics: "", genre: "", bpm: "", seed: "",
  },
  // Animation
  animForm: {
    model_name: "echomimic", expression_scale: 1.0, pose_style: 0,
    use_enhancer: false, source_image_b64: null, audio_b64: null,
  },
  animSourceName: "", animAudioName: "",
  // Jobs
  jobs: [],
  queueStats: { queued: 0, processing: 0, completed: 0, failed: 0, total: 0 },
  // Active SSE connections: jobId -> EventSource
  _sseConnections: {},
};

const root = document.getElementById("root");

/* ─── Utilities ──────────────────────────────────── */

function escapeHtml(v) {
  return String(v ?? "")
    .replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")
    .replace(/"/g,"&quot;").replace(/'/g,"&#39;");
}
function formatMs(ms)  { return ms  ? `${(ms/1000).toFixed(2)}s` : "—"; }
function formatDur(s)  { if (!s) return "—"; const m=Math.floor(s/60),r=Math.round(s%60); return m>0?`${m}m ${r}s`:`${r}s`; }
function activeModel() { return state.models.find(m=>m.name===state.form.model_name)||null; }
function imageSrc(r)   { return r?.image_base64 ? `data:image/png;base64,${r.image_base64}` : ""; }

function statusBadge(status) {
  const map={
    queued:     {cls:"badge-queued",    icon:"◎", text:"Queued"},
    processing: {cls:"badge-processing",icon:"⟳", text:"Processing"},
    completed:  {cls:"badge-done",      icon:"✓", text:"Done"},
    failed:     {cls:"badge-fail",      icon:"✕", text:"Failed"},
    cancelled:  {cls:"badge-cancel",    icon:"—", text:"Cancelled"},
  };
  const s=map[status]||{cls:"",icon:"?",text:status};
  return `<span class="status-badge ${s.cls}">${s.icon} ${s.text}</span>`;
}

function fileToBase64(file) {
  return new Promise((res,rej)=>{
    const r=new FileReader();
    r.onload=()=>res(r.result.split(",")[1]);
    r.onerror=rej;
    r.readAsDataURL(file);
  });
}

function progressBar(pct, status) {
  if (status !== "processing") return "";
  const safe = Math.max(0, Math.min(100, pct||0));
  return `
    <div class="job-progress" title="${safe.toFixed(0)}% complete">
      <div class="job-progress-bar" style="width:${safe}%"></div>
    </div>
    <div class="job-progress-label">${safe.toFixed(0)}%</div>`;
}

/* ─── API ─────────────────────────────────────────── */

async function fetchJson(url, opts) {
  const r = await fetch(url, opts);
  const d = await r.json();
  return { response: r, data: d };
}

async function loadHealth() {
  try {
    const { data } = await fetchJson("/health");
    state.health = data;
  } catch { state.health = null; }
  render();
}

async function loadModels() {
  try {
    const { data } = await fetchJson("/models");
    state.models      = (data.models||[]).filter(m=>m.category==="image");
    state.videoModels = (data.models||[]).filter(m=>m.category==="video");
    state.musicModels = (data.models||[]).filter(m=>m.category==="music");
    state.animModels  = (data.models||[]).filter(m=>m.category==="animation");
    state.categories  = data.categories||[];
    const cur = data.current_model || state.models[0]?.name || "flux-1-dev";
    const info = state.models.find(m=>m.name===cur);
    state.form.model_name = cur;
    if (info) {
      state.form.num_inference_steps = info.default_steps ?? state.form.num_inference_steps;
      state.form.guidance_scale = info.default_guidance_scale ?? state.form.guidance_scale;
    }
    await Promise.all([loadLoras(cur), loadVideoLoras()]);
  } catch { state.models=[]; render(); }
}

async function loadLoras(modelName) {
  try {
    const { data } = await fetchJson(`/loras?model_name=${encodeURIComponent(modelName)}`);
    state.loras = ["None", ...(data.loras||[])];
    state.form.lora_name  = state.loras.includes(state.form.lora_name) ? state.form.lora_name : "None";
    state.form.lora_scale = data.recommended_scale ?? state.form.lora_scale;
  } catch { state.loras=["None"]; }
  render();
}

async function loadVideoLoras() {
  try {
    const { data } = await fetchJson("/api/video/loras");
    state.videoLoras = ["None", ...(data.loras||[])];
    if (!state.videoLoras.includes(state.videoForm.lora_name)) {
      state.videoForm.lora_name = "None";
    }
  } catch { state.videoLoras=["None"]; }
  render();
}

async function loadJobs() {
  try {
    const { data } = await fetchJson("/api/jobs?limit=30");
    state.jobs = data.jobs||[];
  } catch { state.jobs=[]; }
  try {
    const { data } = await fetchJson("/api/queue/status");
    state.queueStats = data;
  } catch {}
}

/* ─── Image ──────────────────────────────────────── */

function applyModel(name) {
  const info = state.models.find(m=>m.name===name);
  state.form.model_name = name;
  state.form.lora_name  = "None";
  if (info) {
    state.form.num_inference_steps = info.default_steps ?? state.form.num_inference_steps;
    state.form.guidance_scale = info.default_guidance_scale ?? state.form.guidance_scale;
  }
  render(); loadLoras(name);
}

function applyResolution(w, h) { state.form.width=w; state.form.height=h; render(); }

function downloadImage() {
  if (!state.result?.image_base64) return;
  const a = document.createElement("a");
  a.href = `data:image/png;base64,${state.result.image_base64}`;
  a.download = `image_${state.result.seed_used||Date.now()}.png`;
  a.click();
}

async function uploadLora(file) {
  state.loraUploadStatus = "uploading"; render();
  const form = new FormData(); form.append("file", file);
  try {
    const r = await fetch("/loras/upload", { method:"POST", body:form });
    const d = await r.json();
    state.loraUploadStatus = r.ok ? "ok:"+d.filename : "err:"+(d?.detail||"Upload failed");
    if (r.ok) await loadLoras(state.form.model_name);
  } catch { state.loraUploadStatus = "err:Network error"; }
  render();
  setTimeout(()=>{ state.loraUploadStatus=null; render(); }, 4500);
}

async function uploadVideoLora(file) {
  state.videoLoraUploadStatus = "uploading"; render();
  const form = new FormData(); form.append("file", file);
  try {
    const r = await fetch("/api/video/loras/upload", { method:"POST", body:form });
    const d = await r.json();
    state.videoLoraUploadStatus = r.ok ? "ok:"+d.filename : "err:"+(d?.detail||"Upload failed");
    if (r.ok) await loadVideoLoras();
  } catch { state.videoLoraUploadStatus = "err:Network error"; }
  render();
  setTimeout(()=>{ state.videoLoraUploadStatus=null; render(); }, 4500);
}

async function onGenerate(e) {
  e.preventDefault();
  state.error=""; state.loading=true; render();
  const payload = {
    prompt: state.form.prompt,
    negative_prompt: state.form.negative_prompt||null,
    model_name: state.form.model_name,
    width: Number(state.form.width),
    height: Number(state.form.height),
    num_inference_steps: Number(state.form.num_inference_steps),
    guidance_scale: Number(state.form.guidance_scale),
    seed: state.form.seed===""?null:Number(state.form.seed),
    lora_name: !state.form.lora_name||state.form.lora_name==="None"?null:state.form.lora_name,
    lora_scale: Number(state.form.lora_scale),
    use_refiner: false,
  };
  try {
    const { response, data } = await fetchJson("/generate-ui",{
      method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload),
    });
    if (!response.ok) { state.error=data?.detail||"Generation failed."; }
    else {
      const r = {...data, model_name:state.form.model_name, prompt:state.form.prompt};
      state.result = r;
      state.history = [r,...state.history].slice(0,16);
    }
  } catch { state.error="Request failed. Check server/network."; }
  state.loading=false; render(); loadHealth();
}

/* ─── Video ──────────────────────────────────────── */

async function onVideoGenerate(e) {
  e.preventDefault(); state.error="";
  const f = state.videoForm;
  const payload = {
    prompt: f.prompt,
    negative_prompt: f.negative_prompt||null,
    model_name: f.model_name,
    resolution: f.resolution,
    num_frames: Number(f.num_frames),
    fps: Number(f.fps),
    guidance_scale: Number(f.guidance_scale),
    num_inference_steps: Number(f.num_inference_steps),
    seed: f.seed===""?null:Number(f.seed),
    source_image_b64: f.source_image_b64||null,
    lora_name: !f.lora_name||f.lora_name==="None"?null:f.lora_name,
    lora_scale: Number(f.lora_scale),
  };
  try {
    const { response, data } = await fetchJson("/api/video/generate",{
      method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload),
    });
    if (!response.ok) { state.error=data?.detail||"Video generation failed."; render(); return; }
    startJobSSE(data.job_id);
  } catch { state.error="Request failed. Check server/network."; }
  render();
  await loadJobs(); render();
}

/* ─── Music ──────────────────────────────────────── */

async function onMusicGenerate(e) {
  e.preventDefault(); state.error="";
  const f = state.musicForm;
  const payload = {
    prompt: f.prompt, model_name: f.model_name,
    duration_seconds: Number(f.duration_seconds),
    lyrics: f.lyrics||null, genre: f.genre||null,
    bpm: f.bpm===""?null:Number(f.bpm),
    seed: f.seed===""?null:Number(f.seed),
  };
  try {
    const { response, data } = await fetchJson("/api/music/generate",{
      method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload),
    });
    if (!response.ok) { state.error=data?.detail||"Music generation failed."; render(); return; }
    startJobSSE(data.job_id);
  } catch { state.error="Request failed. Check server/network."; }
  render(); await loadJobs(); render();
}

/* ─── Animation ──────────────────────────────────── */

async function onAnimGenerate(e) {
  e.preventDefault(); state.error="";
  const f = state.animForm;
  if (!f.source_image_b64||!f.audio_b64) {
    state.error="Please upload both a face image and an audio file."; render(); return;
  }
  const payload = {
    model_name: f.model_name, source_image_b64: f.source_image_b64,
    audio_b64: f.audio_b64, expression_scale: Number(f.expression_scale),
    pose_style: Number(f.pose_style), use_enhancer: f.use_enhancer,
  };
  try {
    const { response, data } = await fetchJson("/api/animation/generate",{
      method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload),
    });
    if (!response.ok) { state.error=data?.detail||"Animation failed."; render(); return; }
    startJobSSE(data.job_id);
  } catch { state.error="Request failed. Check server/network."; }
  render(); await loadJobs(); render();
}

/* ─── SSE Job Streaming ──────────────────────────── */

function startJobSSE(jobId) {
  if (state._sseConnections[jobId]) return;
  const es = new EventSource(`/api/jobs/${jobId}/stream`);
  state._sseConnections[jobId] = es;

  es.onmessage = (evt) => {
    try {
      const msg = JSON.parse(evt.data);
      // Upsert job in state
      const idx = state.jobs.findIndex(j=>j.job_id===jobId);
      const current = idx>=0 ? {...state.jobs[idx]} : { job_id: jobId };
      current.status   = msg.status || current.status;
      current.progress = msg.progress ?? current.progress ?? 0;
      if (msg.result)  current.result = msg.result;
      if (msg.error)   current.error_message = msg.error;
      if (idx>=0) state.jobs[idx]=current; else state.jobs.unshift(current);

      const terminal = ["completed","failed","cancelled"].includes(msg.status);
      if (terminal) {
        es.close();
        delete state._sseConnections[jobId];
        loadJobs();
      }
      render();
    } catch {}
  };

  es.onerror = () => {
    es.close();
    delete state._sseConnections[jobId];
    // Fallback: poll once
    setTimeout(() => startJobPolling(jobId), 2000);
  };
}

/* ─── Polling fallback ───────────────────────────── */

const _pollingTimers = {};

function startJobPolling(jobId) {
  if (_pollingTimers[jobId] || state._sseConnections[jobId]) return;
  _pollingTimers[jobId] = setInterval(async()=>{
    try {
      const { data } = await fetchJson(`/api/jobs/${jobId}`);
      const idx = state.jobs.findIndex(j=>j.job_id===jobId);
      if (idx>=0) state.jobs[idx]=data; else state.jobs.unshift(data);
      if (["completed","failed","cancelled"].includes(data.status)) {
        clearInterval(_pollingTimers[jobId]);
        delete _pollingTimers[jobId];
        loadJobs();
      }
      render();
    } catch {
      clearInterval(_pollingTimers[jobId]);
      delete _pollingTimers[jobId];
    }
  }, 2500);
}

async function cancelJob(jobId) {
  try {
    await fetch(`/api/jobs/${jobId}`,{method:"DELETE"});
    if (_pollingTimers[jobId]) { clearInterval(_pollingTimers[jobId]); delete _pollingTimers[jobId]; }
    if (state._sseConnections[jobId]) { state._sseConnections[jobId].close(); delete state._sseConnections[jobId]; }
    await loadJobs(); render();
  } catch {}
}

function setTab(t) { state.activeTab=t; state.error=""; render(); }

/* ═══════════════════════════════════════════════════
   RENDER
   ═══════════════════════════════════════════════════ */

function render() {
  const cur = activeModel();
  const curImg = imageSrc(state.result);
  const tab = state.activeTab;

  root.innerHTML = `
    <div class="app-shell">
      <div class="ambient ambient-left"></div>
      <div class="ambient ambient-right"></div>
      <div class="app">

        <!-- HERO -->
        <section class="hero premium-card">
          <div class="hero-copy">
            <div class="eyebrow">Neural Creation Studio</div>
            <h1>Multi-modal AI creation platform.</h1>
            <p class="subtitle">Generate images, videos, music, and animations — all from one interface.</p>
          </div>
          <div class="hero-status premium-card inner-card">
            <div class="status-top">
              <span class="dot ${state.health?.status==="healthy"?"ok":""}"></span>
              <span>${escapeHtml(state.health?.status==="healthy"?"System Ready":(state.health?.status||"Starting…"))}</span>
            </div>
            <div class="status-grid">
              <div><label>Active Model</label><strong>${escapeHtml(state.health?.current_model||"—")}</strong></div>
              <div><label>Queue</label><strong>${state.queueStats.queued} queued · ${state.queueStats.processing} running</strong></div>
              <div><label>VRAM</label><strong>${state.health?.vram_used_gb!=null?`${state.health.vram_used_gb.toFixed(1)} / ${state.health.vram_total_gb?.toFixed(0)} GB`:"—"}</strong></div>
            </div>
          </div>
        </section>

        <!-- TAB NAV -->
        <nav class="tab-nav">
          <button class="tab-btn ${tab==="image"?"active":""}" data-tab="image"><span class="tab-icon">◆</span> Image</button>
          <button class="tab-btn ${tab==="video"?"active":""}" data-tab="video"><span class="tab-icon">▶</span> Video</button>
          <button class="tab-btn ${tab==="music"?"active":""}" data-tab="music"><span class="tab-icon">♫</span> Music</button>
          <button class="tab-btn ${tab==="animation"?"active":""}" data-tab="animation"><span class="tab-icon">◎</span> Animation</button>
          <button class="tab-btn ${tab==="queue"?"active":""}" data-tab="queue"><span class="tab-icon">☰</span> Jobs
            <span class="job-count">${state.jobs.filter(j=>j.status==="queued"||j.status==="processing").length||""}</span>
          </button>
        </nav>

        ${tab==="image"     ? renderImageTab(cur,curImg) : ""}
        ${tab==="video"     ? renderVideoTab()           : ""}
        ${tab==="music"     ? renderMusicTab()           : ""}
        ${tab==="animation" ? renderAnimationTab()       : ""}
        ${tab==="queue"     ? renderQueueTab()           : ""}

        ${state.error ? `<div class="global-error">⚠ ${escapeHtml(state.error)}</div>` : ""}
      </div>
      ${state.lightbox ? `<div class="lightbox" id="lightbox"><img src="${state.lightbox}" alt="Preview" /></div>` : ""}
    </div>`;

  bindEvents();
}

/* ─── IMAGE TAB ──────────────────────────────────── */

function renderImageTab(currentModel, currentImage) {
  return `
    <section class="model-strip">
      ${state.models.map(m=>`
        <button type="button" class="model-card premium-card ${state.form.model_name===m.name?"active":""}" data-model="${escapeHtml(m.name)}">
          <div class="model-name">${escapeHtml(m.name)}</div>
          <div class="model-summary">${escapeHtml(m.summary||m.description||"Ready")}</div>
        </button>`).join("")}
    </section>

    <div class="layout">
      <form class="premium-card control-panel" id="generate-form">
        <div class="section-heading"><div><div class="eyebrow">Image Generation</div><h2>Compose</h2></div></div>

        <div class="field"><label>Prompt</label>
          <textarea id="prompt" placeholder="Cinematic portrait, neon cityscape, hyperrealistic..." required>${escapeHtml(state.form.prompt)}</textarea></div>
        <div class="field"><label>Negative Prompt</label>
          <textarea id="negative_prompt" class="compact" placeholder="blurry, low quality, watermark">${escapeHtml(state.form.negative_prompt)}</textarea></div>

        <div class="field-group three-col">
          <div class="field"><label>Model</label>
            <select id="model_name">${state.models.map(m=>`<option value="${escapeHtml(m.name)}" ${state.form.model_name===m.name?"selected":""}>${escapeHtml(m.name)}</option>`).join("")}</select></div>
          <div class="field"><label>LoRA</label>
            <div class="lora-row">
              <select id="lora_name">${state.loras.map(l=>`<option value="${escapeHtml(l)}" ${state.form.lora_name===l?"selected":""}>${escapeHtml(l)}</option>`).join("")}</select>
              <label class="upload-lora-btn ${state.loraUploadStatus==="uploading"?"uploading":""}" title="Upload .safetensors LoRA">
                <input type="file" id="lora-file-input" accept=".safetensors" style="display:none" />
                ${state.loraUploadStatus==="uploading"?"…":"⬆"}
              </label>
            </div>
            ${state.loraUploadStatus&&state.loraUploadStatus!=="uploading"?`<div class="lora-upload-msg ${state.loraUploadStatus.startsWith("ok")?"ok":"err"}">${escapeHtml(state.loraUploadStatus.startsWith("ok:")?"✓ Uploaded: "+state.loraUploadStatus.slice(3):state.loraUploadStatus.slice(4))}</div>`:""}
          </div>
          <div class="field"><label>LoRA Scale</label>
            <input id="lora_scale" type="number" min="0" max="2" step="0.05" value="${escapeHtml(state.form.lora_scale)}" /></div>
        </div>

        <div class="field"><label>Resolution</label>
          <div class="preset-row">${RESOLUTION_PRESETS.map(p=>{
            const a=state.form.width===p.width&&state.form.height===p.height;
            return `<button type="button" class="preset-chip ${a?"active":""}" data-width="${p.width}" data-height="${p.height}">${escapeHtml(p.label)}<small>${p.width}×${p.height}</small></button>`;
          }).join("")}</div></div>

        <div class="field-group four-col">
          <div class="field"><label>W</label><input id="width" type="number" min="256" max="2048" step="8" value="${escapeHtml(state.form.width)}" /></div>
          <div class="field"><label>H</label><input id="height" type="number" min="256" max="2048" step="8" value="${escapeHtml(state.form.height)}" /></div>
          <div class="field"><label>Steps</label><input id="num_inference_steps" type="number" min="${currentModel?.min_steps||1}" max="${currentModel?.max_steps||50}" value="${escapeHtml(state.form.num_inference_steps)}" /></div>
          <div class="field"><label>CFG</label><input id="guidance_scale" type="number" min="0" max="20" step="0.5" value="${escapeHtml(state.form.guidance_scale)}" /></div>
        </div>

        <div class="field-group two-col align-end">
          <div class="field"><label>Seed</label><input id="seed" value="${escapeHtml(state.form.seed)}" placeholder="blank = random" /></div>
          <button class="primary-button" type="submit" ${state.loading?"disabled":""}>${state.loading?"Generating…":"Generate Image"}</button>
        </div>

        ${currentModel?`<div class="model-note"><strong>${escapeHtml(currentModel.name)}</strong><span>${escapeHtml(currentModel.description||"")}</span></div>`:""}
      </form>

      <section class="output-column">
        <div class="premium-card output-panel">
          <div class="section-heading compact-heading">
            <div><div class="eyebrow">Output</div><h2>Preview</h2></div>
            <div class="output-actions">
              ${currentImage&&!state.loading?`<button type="button" class="download-btn" id="download-btn">⬇ Download</button>`:""}
              <div class="metric-inline"><span>${escapeHtml(state.result?.model_name||state.health?.current_model||"No model")}</span></div>
            </div>
          </div>
          <div class="canvas" id="canvas">
            ${state.loading?`<div class="loading-spinner"><div class="spinner"></div><span>Generating image…</span></div>`
              :currentImage?`<img src="${currentImage}" alt="Generated" />`
              :`<span class="empty-state">Your image will appear here.</span>`}
          </div>
          <div class="result-grid">
            <div class="result-card"><label>Seed</label><strong>${escapeHtml(state.result?.seed_used??"-")}</strong></div>
            <div class="result-card"><label>Time</label><strong>${escapeHtml(formatMs(state.result?.inference_time_ms))}</strong></div>
            <div class="result-card"><label>Size</label><strong>${state.result?`${state.form.width}×${state.form.height}`:"—"}</strong></div>
          </div>
        </div>

        <div class="premium-card history-panel">
          <div class="section-heading compact-heading"><div><div class="eyebrow">History</div><h2>Recent generations (${state.history.length})</h2></div></div>
          <div class="history-grid">
            ${state.history.length===0?`<div class="history-empty">No generations yet.</div>`
              :state.history.map((item,i)=>`
                <button type="button" class="history-item" data-history-index="${i}">
                  <img src="data:image/png;base64,${item.image_base64}" alt="${escapeHtml(item.prompt||"")}" />
                  <div class="history-meta"><strong>${escapeHtml(item.model_name)}</strong><span>${escapeHtml(formatMs(item.inference_time_ms))}</span></div>
                </button>`).join("")}
          </div>
        </div>
      </section>
    </div>`;
}

/* ─── VIDEO TAB ──────────────────────────────────── */

function renderVideoTab() {
  const f = state.videoForm;
  const latestVideoJob = state.jobs.find(j=>j.job_type==="video"&&
    (j.status==="completed"||j.status==="processing"||j.status==="queued"));

  return `
    <div class="layout">
      <form class="premium-card control-panel" id="video-form">
        <div class="section-heading"><div><div class="eyebrow">Video Generation</div><h2>Text to Video</h2></div></div>

        <div class="field"><label>Prompt</label>
          <textarea id="v-prompt" placeholder="A camera slowly pans across a futuristic city at golden hour…" required>${escapeHtml(f.prompt)}</textarea></div>

        <div class="field"><label>Negative Prompt</label>
          <textarea id="v-neg" class="compact" placeholder="blurry, static, low motion, worst quality">${escapeHtml(f.negative_prompt||"")}</textarea></div>

        <div class="field"><label>Model</label>
          <select id="v-model">${state.videoModels.map(m=>`
            <option value="${escapeHtml(m.name)}" ${f.model_name===m.name?"selected":""}>${escapeHtml(m.name)} — ${escapeHtml(m.description||"")}</option>`).join("")}
          </select></div>

        <div class="field-group two-col">
          <div class="field"><label>Resolution</label>
            <select id="v-resolution">
              <option value="480p" ${f.resolution==="480p"?"selected":""}>480p (848×480)</option>
              <option value="720p" ${f.resolution==="720p"?"selected":""}>720p (1280×720)</option>
            </select></div>
          <div class="field"><label>FPS</label>
            <select id="v-fps">
              <option value="16" ${f.fps==16?"selected":""}>16 fps</option>
              <option value="24" ${f.fps==24?"selected":""}>24 fps</option>
            </select></div>
        </div>

        <div class="field"><label>Duration</label>
          <div class="preset-row">${VIDEO_FRAME_PRESETS.map(p=>`
            <button type="button" class="preset-chip ${Number(f.num_frames)===p.frames?"active":""}" data-vframes="${p.frames}">
              ${p.label}<small>${p.frames} frames</small></button>`).join("")}</div></div>

        <div class="field-group two-col">
          <div class="field"><label>Steps</label><input id="v-steps" type="number" min="10" max="50" value="${f.num_inference_steps}" /></div>
          <div class="field"><label>Guidance</label><input id="v-guidance" type="number" min="0" max="20" step="0.5" value="${f.guidance_scale}" /></div>
        </div>

        <div class="field"><label>Seed</label>
          <input id="v-seed" value="${escapeHtml(f.seed)}" placeholder="blank = random" /></div>

        <!-- Video LoRA section -->
        <div class="field"><label>Video LoRA</label>
          <div class="lora-row">
            <select id="v-lora">${state.videoLoras.map(l=>`<option value="${escapeHtml(l)}" ${f.lora_name===l?"selected":""}>${escapeHtml(l)}</option>`).join("")}</select>
            <label class="upload-lora-btn ${state.videoLoraUploadStatus==="uploading"?"uploading":""}" title="Upload video LoRA (.safetensors)">
              <input type="file" id="v-lora-file-input" accept=".safetensors" style="display:none" />
              ${state.videoLoraUploadStatus==="uploading"?"…":"⬆"}
            </label>
          </div>
          ${state.videoLoraUploadStatus&&state.videoLoraUploadStatus!=="uploading"?`<div class="lora-upload-msg ${state.videoLoraUploadStatus.startsWith("ok")?"ok":"err"}">${escapeHtml(state.videoLoraUploadStatus.startsWith("ok:")?"✓ Uploaded: "+state.videoLoraUploadStatus.slice(3):state.videoLoraUploadStatus.slice(4))}</div>`:""}
        </div>
        <div class="field"><label>LoRA Scale</label>
          <input id="v-lora-scale" type="number" min="0" max="2" step="0.05" value="${f.lora_scale}" /></div>

        <div class="field"><label>Image-to-Video source (optional)</label>
          <div class="dropzone" id="v-dropzone">
            <input type="file" id="v-source-img" accept="image/*" style="display:none" />
            ${f.source_image_b64
              ?`<div class="dropzone-preview"><img src="data:image/png;base64,${f.source_image_b64}" alt="Source" /><button type="button" class="dropzone-clear" id="v-clear-img">✕</button></div>`
              :`<div class="dropzone-label" id="v-dropzone-label"><span class="dropzone-icon">🖼</span><span>Drop or click to upload source image</span></div>`}
          </div></div>

        <button class="primary-button" type="submit">▶ Generate Video</button>
      </form>

      <section class="output-column">
        <div class="premium-card output-panel">
          <div class="section-heading compact-heading">
            <div><div class="eyebrow">Output</div><h2>Video Preview</h2></div>
            ${latestVideoJob?.result?.video_url?`<a class="download-btn" href="${latestVideoJob.result.video_url}" download>⬇ Download</a>`:""}
          </div>
          ${renderVideoResult(latestVideoJob)}
        </div>
        ${renderRecentJobs("video")}
      </section>
    </div>`;
}

function renderVideoResult(job) {
  if (!job) return `<div class="canvas"><span class="empty-state">Submit a prompt to generate a video.</span></div>`;
  if (job.status==="completed"&&job.result?.video_url) {
    return `
      <div class="video-container">
        <video controls autoplay loop playsinline src="${job.result.video_url}"></video>
      </div>
      <div class="result-grid">
        <div class="result-card"><label>Duration</label><strong>${formatDur(job.result.duration_seconds)}</strong></div>
        <div class="result-card"><label>Frames</label><strong>${job.result.num_frames||"—"}</strong></div>
        <div class="result-card"><label>Time</label><strong>${formatMs(job.result.inference_time_ms)}</strong></div>
      </div>`;
  }
  if (job.status==="processing"||job.status==="queued") {
    const pct = job.progress||0;
    const label = job.status==="queued"?"Waiting in queue…":`Generating video… ${pct.toFixed(0)}%`;
    return `
      <div class="canvas">
        <div class="loading-spinner">
          <div class="spinner"></div>
          <span>${label}</span>
          <div class="progress-track"><div class="progress-fill" style="width:${pct}%"></div></div>
        </div>
      </div>`;
  }
  if (job.status==="failed") {
    return `<div class="canvas"><span class="empty-state error-state">Generation failed: ${escapeHtml(job.error_message||"Unknown error")}</span></div>`;
  }
  return `<div class="canvas"><span class="empty-state">Submit a prompt to generate a video.</span></div>`;
}

/* ─── MUSIC TAB ──────────────────────────────────── */

function renderMusicTab() {
  const f = state.musicForm;
  const latest = state.jobs.find(j=>j.job_type==="music"&&
    (j.status==="completed"||j.status==="processing"||j.status==="queued"));

  return `
    <div class="layout">
      <form class="premium-card control-panel" id="music-form">
        <div class="section-heading"><div><div class="eyebrow">Music Generation</div><h2>Create Music</h2></div></div>

        <div class="field"><label>Prompt</label>
          <textarea id="m-prompt" placeholder="Upbeat electronic track with synth pads and deep bass…" required>${escapeHtml(f.prompt)}</textarea></div>

        <div class="field"><label>Model</label>
          <select id="m-model">${state.musicModels.map(m=>`
            <option value="${escapeHtml(m.name)}" ${f.model_name===m.name?"selected":""}>${escapeHtml(m.name)} — ${escapeHtml(m.description||"")}</option>`).join("")}
          </select></div>

        <div class="field"><label>Genre</label>
          <div class="style-row">${GENRE_PILLS.map(g=>`
            <button type="button" class="style-chip ${f.genre===g?"active":""}" data-genre="${escapeHtml(g)}">${escapeHtml(g)}</button>`).join("")}</div></div>

        <div class="field-group three-col">
          <div class="field"><label>Duration (s)</label><input id="m-duration" type="number" min="5" max="300" value="${f.duration_seconds}" /></div>
          <div class="field"><label>BPM</label><input id="m-bpm" type="number" min="40" max="240" value="${escapeHtml(f.bpm)}" placeholder="auto" /></div>
          <div class="field"><label>Seed</label><input id="m-seed" value="${escapeHtml(f.seed)}" placeholder="random" /></div>
        </div>

        ${f.model_name==="ace-step"?`
          <div class="field"><label>Lyrics (ACE-Step)</label>
            <textarea id="m-lyrics" class="compact" placeholder="[Verse]&#10;Your lyrics here…&#10;[Chorus]&#10;The hook…">${escapeHtml(f.lyrics)}</textarea></div>`:""}

        <button class="primary-button" type="submit">♫ Generate Music</button>
      </form>

      <section class="output-column">
        <div class="premium-card output-panel">
          <div class="section-heading compact-heading">
            <div><div class="eyebrow">Output</div><h2>Audio Preview</h2></div>
            ${latest?.result?.audio_url?`<a class="download-btn" href="${latest.result.audio_url}" download>⬇ Download</a>`:""}
          </div>
          ${latest?.status==="completed"&&latest.result?.audio_url?`
            <div class="audio-container">
              <audio controls src="${latest.result.audio_url}"></audio>
              <div class="result-grid" style="margin-top:14px">
                <div class="result-card"><label>Duration</label><strong>${formatDur(latest.result.duration_seconds)}</strong></div>
                <div class="result-card"><label>Sample Rate</label><strong>${latest.result.sample_rate||"—"} Hz</strong></div>
                <div class="result-card"><label>Time</label><strong>${formatMs(latest.result.inference_time_ms)}</strong></div>
              </div>
            </div>`
          :latest?.status==="processing"||latest?.status==="queued"?`
            <div class="canvas"><div class="loading-spinner"><div class="spinner"></div>
              <span>${latest.status==="queued"?"Waiting in queue…":"Generating music…"}</span>
              ${progressBar(latest.progress, latest.status)}</div></div>`
          :`<div class="canvas"><span class="empty-state">Submit a prompt to generate music.</span></div>`}
        </div>
        ${renderRecentJobs("music")}
      </section>
    </div>`;
}

/* ─── ANIMATION TAB ──────────────────────────────── */

function renderAnimationTab() {
  const f = state.animForm;
  const latest = state.jobs.find(j=>j.job_type==="animation"&&
    (j.status==="completed"||j.status==="processing"||j.status==="queued"));

  return `
    <div class="layout">
      <form class="premium-card control-panel" id="anim-form">
        <div class="section-heading"><div><div class="eyebrow">Animation</div><h2>Talking Head</h2></div></div>

        <div class="field"><label>Model</label>
          <select id="a-model">${state.animModels.map(m=>`
            <option value="${escapeHtml(m.name)}" ${f.model_name===m.name?"selected":""}>${escapeHtml(m.name)} — ${escapeHtml(m.description||"")}</option>`).join("")}
          </select></div>

        <div class="field"><label>Face Image</label>
          <div class="dropzone" id="a-img-dropzone">
            <input type="file" id="a-source-img" accept="image/*" style="display:none" />
            ${f.source_image_b64
              ?`<div class="dropzone-preview"><img src="data:image/png;base64,${f.source_image_b64}" alt="Face" /><button type="button" class="dropzone-clear" id="a-clear-img">✕</button></div>`
              :`<div class="dropzone-label"><span class="dropzone-icon">👤</span><span>Upload a clear face photo</span></div>`}
          </div></div>

        <div class="field"><label>Audio File</label>
          <div class="dropzone" id="a-audio-dropzone">
            <input type="file" id="a-source-audio" accept="audio/*,.wav,.mp3,.ogg" style="display:none" />
            ${f.audio_b64
              ?`<div class="dropzone-file"><span>🎵 ${escapeHtml(state.animAudioName||"Audio loaded")}</span><button type="button" class="dropzone-clear" id="a-clear-audio">✕</button></div>`
              :`<div class="dropzone-label"><span class="dropzone-icon">🎙</span><span>Upload speech or song audio (WAV/MP3)</span></div>`}
          </div></div>

        <div class="field-group two-col">
          <div class="field"><label>Expression Scale</label><input id="a-expr" type="number" min="0.1" max="3.0" step="0.1" value="${f.expression_scale}" /></div>
          <div class="field"><label>Pose Style</label><input id="a-pose" type="number" min="0" max="46" value="${f.pose_style}" /></div>
        </div>

        <button class="primary-button" type="submit" ${!f.source_image_b64||!f.audio_b64?"disabled":""}>◎ Generate Animation</button>
      </form>

      <section class="output-column">
        <div class="premium-card output-panel">
          <div class="section-heading compact-heading">
            <div><div class="eyebrow">Output</div><h2>Animation Preview</h2></div>
            ${latest?.result?.video_url?`<a class="download-btn" href="${latest.result.video_url}" download>⬇ Download</a>`:""}
          </div>
          ${latest?.status==="completed"&&latest.result?.video_url?`
            <div class="video-container"><video controls autoplay loop playsinline src="${latest.result.video_url}"></video></div>
            <div class="result-grid">
              <div class="result-card"><label>Duration</label><strong>${formatDur(latest.result.duration_seconds)}</strong></div>
              <div class="result-card"><label>Time</label><strong>${formatMs(latest.result.inference_time_ms)}</strong></div>
            </div>`
          :latest?.status==="processing"||latest?.status==="queued"?`
            <div class="canvas"><div class="loading-spinner"><div class="spinner"></div>
              <span>${latest.status==="queued"?"Waiting…":"Generating animation…"}</span></div></div>`
          :`<div class="canvas"><span class="empty-state">Upload a face image and audio to animate.</span></div>`}
        </div>
        ${renderRecentJobs("animation")}
      </section>
    </div>`;
}

/* ─── QUEUE TAB ──────────────────────────────────── */

function renderQueueTab() {
  const s = state.queueStats;
  return `
    <div class="queue-layout">
      <div class="premium-card queue-stats-card">
        <div class="section-heading"><div><div class="eyebrow">System Status</div><h2>Job Queue</h2></div></div>
        <div class="queue-stats-grid">
          <div class="queue-stat"><div class="queue-stat-value">${s.queued}</div><div class="queue-stat-label">Queued</div></div>
          <div class="queue-stat processing"><div class="queue-stat-value">${s.processing}</div><div class="queue-stat-label">Processing</div></div>
          <div class="queue-stat completed"><div class="queue-stat-value">${s.completed}</div><div class="queue-stat-label">Completed</div></div>
          <div class="queue-stat failed"><div class="queue-stat-value">${s.failed}</div><div class="queue-stat-label">Failed</div></div>
        </div>
      </div>

      <div class="premium-card queue-list-card">
        <div class="section-heading compact-heading"><div><div class="eyebrow">Recent Jobs</div><h2>All Activity</h2></div></div>
        ${state.jobs.length===0?`<div class="empty-state" style="padding:24px">No jobs yet. Generate something to see activity here.</div>`:`
          <div class="job-list">
            ${state.jobs.map(job=>`
              <div class="job-item ${job.status}">
                <div class="job-item-header">
                  <span class="job-type-badge">${escapeHtml(job.job_type)}</span>
                  ${statusBadge(job.status)}
                  <span class="job-model">${escapeHtml(job.model_name||"")}</span>
                  <span class="job-time">${escapeHtml(job.processing_time_ms?formatMs(job.processing_time_ms):"")}</span>
                  ${job.status==="queued"?`<button class="job-cancel-btn" data-cancel="${escapeHtml(job.job_id)}">Cancel</button>`:""}
                </div>
                ${job.status==="processing"?progressBar(job.progress,"processing"):""}
                ${job.status==="completed"&&job.result?.video_url?`<a class="job-result-link" href="${job.result.video_url}" target="_blank">▶ View Video</a>`:""}
                ${job.status==="completed"&&job.result?.audio_url?`<a class="job-result-link" href="${job.result.audio_url}" target="_blank">♫ Play Audio</a>`:""}
                ${job.status==="failed"&&job.error_message?`<div class="job-error">${escapeHtml(job.error_message)}</div>`:""}
              </div>`).join("")}
          </div>`}
      </div>
    </div>`;
}

/* ─── Shared: recent jobs panel ──────────────────── */

function renderRecentJobs(type) {
  const jobs = state.jobs.filter(j=>j.job_type===type).slice(0,5);
  if (!jobs.length) return "";
  return `
    <div class="premium-card queue-list-card">
      <div class="section-heading compact-heading"><div><div class="eyebrow">Activity</div><h2>Recent ${type} jobs</h2></div></div>
      <div class="job-list compact">
        ${jobs.map(job=>`
          <div class="job-item ${job.status}">
            <div class="job-item-header">
              ${statusBadge(job.status)}
              <span class="job-model">${escapeHtml(job.model_name||"")}</span>
              <span class="job-time">${job.processing_time_ms?formatMs(job.processing_time_ms):""}</span>
              ${job.status==="queued"?`<button class="job-cancel-btn" data-cancel="${escapeHtml(job.job_id)}">Cancel</button>`:""}
            </div>
            ${job.status==="processing"?progressBar(job.progress,"processing"):""}
            ${job.status==="completed"&&job.result?.video_url?`<a class="job-result-link" href="${job.result.video_url}" target="_blank">▶ Play</a>`:""}
            ${job.status==="completed"&&job.result?.audio_url?`<a class="job-result-link" href="${job.result.audio_url}" target="_blank">♫ Play</a>`:""}
            ${job.status==="failed"?`<div class="job-error">${escapeHtml(job.error_message||"Failed")}</div>`:""}
          </div>`).join("")}
      </div>
    </div>`;
}

/* ─── Event Binding ──────────────────────────────── */

function bindEvents() {
  // Tab navigation
  document.querySelectorAll("[data-tab]").forEach(btn=>{
    btn.addEventListener("click", ()=>setTab(btn.dataset.tab));
  });

  // Model cards
  document.querySelectorAll("[data-model]").forEach(card=>{
    card.addEventListener("click", ()=>applyModel(card.dataset.model));
  });

  // Resolution presets
  document.querySelectorAll("[data-width]").forEach(btn=>{
    btn.addEventListener("click", ()=>applyResolution(Number(btn.dataset.width),Number(btn.dataset.height)));
  });

  // Video frame presets
  document.querySelectorAll("[data-vframes]").forEach(btn=>{
    btn.addEventListener("click", ()=>{ state.videoForm.num_frames=Number(btn.dataset.vframes); render(); });
  });

  // Genre chips
  document.querySelectorAll("[data-genre]").forEach(btn=>{
    btn.addEventListener("click", ()=>{ state.musicForm.genre=state.musicForm.genre===btn.dataset.genre?"":btn.dataset.genre; render(); });
  });

  // Image form sync
  bindInput("prompt",   v=>state.form.prompt=v);
  bindInput("negative_prompt", v=>state.form.negative_prompt=v);
  bindInput("model_name", v=>{ state.form.model_name=v; loadLoras(v); });
  bindInput("lora_name", v=>state.form.lora_name=v);
  bindInput("lora_scale", v=>state.form.lora_scale=v);
  bindInput("width",  v=>state.form.width=v);
  bindInput("height", v=>state.form.height=v);
  bindInput("num_inference_steps", v=>state.form.num_inference_steps=v);
  bindInput("guidance_scale", v=>state.form.guidance_scale=v);
  bindInput("seed", v=>state.form.seed=v);

  // Image generate
  const gf = document.getElementById("generate-form");
  if (gf) gf.addEventListener("submit", onGenerate);

  // Download image
  const dlBtn = document.getElementById("download-btn");
  if (dlBtn) dlBtn.addEventListener("click", downloadImage);

  // Image LoRA upload
  const loraInput = document.getElementById("lora-file-input");
  if (loraInput) loraInput.addEventListener("change", e=>{ if(e.target.files[0]) uploadLora(e.target.files[0]); });

  // Lightbox
  const canvas = document.getElementById("canvas");
  if (canvas) canvas.addEventListener("click", ()=>{
    if(state.result?.image_base64) { state.lightbox=imageSrc(state.result); render(); }
  });
  const lb = document.getElementById("lightbox");
  if (lb) lb.addEventListener("click", ()=>{ state.lightbox=null; render(); });

  // History items
  document.querySelectorAll("[data-history-index]").forEach(btn=>{
    btn.addEventListener("click", ()=>{
      const i=Number(btn.dataset.historyIndex);
      state.lightbox=`data:image/png;base64,${state.history[i].image_base64}`;
      render();
    });
  });

  // Video form sync
  bindInput("v-prompt",   v=>state.videoForm.prompt=v);
  bindInput("v-neg",      v=>state.videoForm.negative_prompt=v);
  bindInput("v-model",    v=>state.videoForm.model_name=v);
  bindInput("v-resolution",v=>state.videoForm.resolution=v);
  bindInput("v-fps",      v=>state.videoForm.fps=Number(v));
  bindInput("v-steps",    v=>state.videoForm.num_inference_steps=v);
  bindInput("v-guidance", v=>state.videoForm.guidance_scale=v);
  bindInput("v-seed",     v=>state.videoForm.seed=v);
  bindInput("v-lora",     v=>state.videoForm.lora_name=v);
  bindInput("v-lora-scale",v=>state.videoForm.lora_scale=v);

  // Video LoRA upload
  const vLoraInput = document.getElementById("v-lora-file-input");
  if (vLoraInput) vLoraInput.addEventListener("change", e=>{ if(e.target.files[0]) uploadVideoLora(e.target.files[0]); });

  // Video source image
  const vSrcInput = document.getElementById("v-source-img");
  const vDrop = document.getElementById("v-dropzone");
  if (vSrcInput) vSrcInput.addEventListener("change", async e=>{
    if(e.target.files[0]) {
      state.videoForm.source_image_b64 = await fileToBase64(e.target.files[0]);
      state.videoSourceName = e.target.files[0].name;
      render();
    }
  });
  if (vDrop) {
    vDrop.addEventListener("click", e=>{ if(!e.target.classList.contains("dropzone-clear")) document.getElementById("v-source-img")?.click(); });
    vDrop.addEventListener("dragover", e=>e.preventDefault());
    vDrop.addEventListener("drop", async e=>{
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file?.type.startsWith("image/")) {
        state.videoForm.source_image_b64 = await fileToBase64(file);
        state.videoSourceName = file.name;
        render();
      }
    });
  }
  const vClearImg = document.getElementById("v-clear-img");
  if (vClearImg) vClearImg.addEventListener("click", e=>{ e.stopPropagation(); state.videoForm.source_image_b64=null; state.videoSourceName=""; render(); });

  // Video form submit
  const vf = document.getElementById("video-form");
  if (vf) vf.addEventListener("submit", onVideoGenerate);

  // Music form sync
  bindInput("m-prompt",  v=>state.musicForm.prompt=v);
  bindInput("m-model",   v=>state.musicForm.model_name=v);
  bindInput("m-duration",v=>state.musicForm.duration_seconds=v);
  bindInput("m-bpm",     v=>state.musicForm.bpm=v);
  bindInput("m-seed",    v=>state.musicForm.seed=v);
  bindInput("m-lyrics",  v=>state.musicForm.lyrics=v);
  const mf = document.getElementById("music-form");
  if (mf) mf.addEventListener("submit", onMusicGenerate);

  // Animation form
  bindInput("a-model",v=>state.animForm.model_name=v);
  bindInput("a-expr", v=>state.animForm.expression_scale=v);
  bindInput("a-pose", v=>state.animForm.pose_style=v);

  const aSrcInput = document.getElementById("a-source-img");
  const aDrop = document.getElementById("a-img-dropzone");
  if (aSrcInput) aSrcInput.addEventListener("change", async e=>{
    if(e.target.files[0]) { state.animForm.source_image_b64=await fileToBase64(e.target.files[0]); render(); }
  });
  if (aDrop) aDrop.addEventListener("click", e=>{ if(!e.target.classList.contains("dropzone-clear")) document.getElementById("a-source-img")?.click(); });
  const aClearImg = document.getElementById("a-clear-img");
  if (aClearImg) aClearImg.addEventListener("click", e=>{ e.stopPropagation(); state.animForm.source_image_b64=null; render(); });

  const aAudioInput = document.getElementById("a-source-audio");
  const aAudioDrop = document.getElementById("a-audio-dropzone");
  if (aAudioInput) aAudioInput.addEventListener("change", async e=>{
    if(e.target.files[0]) { state.animForm.audio_b64=await fileToBase64(e.target.files[0]); state.animAudioName=e.target.files[0].name; render(); }
  });
  if (aAudioDrop) aAudioDrop.addEventListener("click", e=>{ if(!e.target.classList.contains("dropzone-clear")) document.getElementById("a-source-audio")?.click(); });
  const aClearAudio = document.getElementById("a-clear-audio");
  if (aClearAudio) aClearAudio.addEventListener("click", e=>{ e.stopPropagation(); state.animForm.audio_b64=null; state.animAudioName=""; render(); });

  const af = document.getElementById("anim-form");
  if (af) af.addEventListener("submit", onAnimGenerate);

  // Job cancel buttons
  document.querySelectorAll("[data-cancel]").forEach(btn=>{
    btn.addEventListener("click", ()=>cancelJob(btn.dataset.cancel));
  });
}

function bindInput(id, setter) {
  const el = document.getElementById(id);
  if (!el) return;
  el.addEventListener("input", ()=>setter(el.value));
  el.addEventListener("change", ()=>setter(el.value));
}

/* ─── Init ───────────────────────────────────────── */

async function init() {
  render();
  await Promise.all([loadHealth(), loadModels(), loadJobs()]);
  render();
  // Periodic refresh
  setInterval(loadHealth, 15000);
  setInterval(async ()=>{ await loadJobs(); render(); }, 8000);
  // Re-connect any active jobs on page reload
  state.jobs.filter(j=>j.status==="processing"||j.status==="queued")
    .forEach(j=>startJobSSE(j.job_id));
}

init();
