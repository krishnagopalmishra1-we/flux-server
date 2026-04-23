/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   Neural Creation Studio â€” Multi-Modal AI Frontend
   Image Â· Video
   â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */

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

const SAMPLE_IMAGES = [
  {
    title: "Chrome Bloom",
    prompt: "iridescent chrome orchid, sunlit studio, macro lens, crisp petals, editorial product photo",
    src: "https://images.unsplash.com/photo-1541701494587-cb58502866ab?auto=format&fit=crop&w=640&q=80",
  },
  {
    title: "Solar Atelier",
    prompt: "bright futuristic fashion atelier, glass walls, golden hour, cinematic composition, ultra detailed",
    src: "https://images.unsplash.com/photo-1500530855697-b586d89ba3ee?auto=format&fit=crop&w=640&q=80",
  },
  {
    title: "Neon Vista",
    prompt: "wide angle neon valley at twilight, glowing river, dreamlike realism, atmospheric depth",
    src: "https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=640&q=80",
  },
  {
    title: "Aurora Frame",
    prompt: "cinematic portrait lit by aurora glass, soft rim light, luminous skin, high fashion editorial",
    src: "https://images.unsplash.com/photo-1519608487953-e999c86e7455?auto=format&fit=crop&w=640&q=80",
  },
];

const IMAGE_STYLES = [
  { id:"cinematic", label:"Cinematic", suffix:"cinematic lighting, dramatic composition, rich color grade, shallow depth of field" },
  { id:"editorial", label:"Editorial", suffix:"high fashion editorial, polished studio light, premium magazine finish, refined texture" },
  { id:"product", label:"Product", suffix:"luxury product photography, clean reflections, crisp material detail, bright commercial lighting" },
  { id:"anime", label:"Anime", suffix:"anime key visual, expressive character design, clean line art, luminous color palette" },
  { id:"realism", label:"Realism", suffix:"photorealistic, natural light, realistic skin and materials, high detail, professional camera" },
  { id:"concept", label:"Concept Art", suffix:"high-end concept art, cinematic worldbuilding, atmospheric scale, intricate design language" },
];

const VIDEO_STYLES = [
  { id:"cinematic", label:"Cinematic", suffix:"cinematic camera movement, soft depth of field, premium color grade, smooth motion" },
  { id:"commercial", label:"Commercial", suffix:"bright commercial look, clean product reveal, controlled studio motion, polished lighting" },
  { id:"dream", label:"Dreamlike", suffix:"ethereal movement, glowing atmosphere, graceful transitions, surreal cinematic mood" },
  { id:"anime", label:"Anime Motion", suffix:"anime style motion, dynamic framing, expressive lighting, clean animated composition" },
  { id:"documentary", label:"Documentary", suffix:"natural handheld camera feel, realistic movement, grounded lighting, observational detail" },
];

const TAB_PATHS = {
  image: "/image",
  video: "/video",
  queue: "/queue",
};

const state = {
  activeTab: "image",
  models: [], videoModels: [],
  health: null, categories: [],
  auth: {
    apiKey: localStorage.getItem("ncs_api_key") || "",
    required: false,
    configuredKeyCount: 0,
  },
  // Image
  history: [], loading: false, error: "", result: null, lightbox: null,
  loraUploadStatus: null,
  form: {
    prompt: "", negative_prompt: "", model_name: "flux-1-dev",
    width: 1024, height: 1024, num_inference_steps: 28,
    guidance_scale: 3.5, seed: "", lora_name: "None", lora_scale: 0.85,
    style_id: "cinematic",
  },
  loras: ["None"],
  // Video
  videoForm: {
    prompt: "", negative_prompt: "", model_name: "wan-t2v-1.3b",
    resolution: "480p", num_frames: 33, fps: 16,
    guidance_scale: 5.0, num_inference_steps: 30, seed: "",
    source_image_b64: null, lora_name: "None", lora_scale: 1.0,
    style_id: "cinematic",
  },
  videoSourceName: "",
  videoLoras: ["None"],
  videoLoraUploadStatus: null,
  // Jobs
  jobs: [],
  queueStats: { queued: 0, processing: 0, completed: 0, failed: 0, total: 0 },
  // Active SSE connections: jobId -> EventSource
  _sseConnections: {},
};

const root = document.getElementById("root");

/* â”€â”€â”€ Utilities â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */

function escapeHtml(v) {
  return String(v ?? "")
    .replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")
    .replace(/"/g,"&quot;").replace(/'/g,"&#39;");
}
function formatMs(ms)  { return ms  ? `${(ms/1000).toFixed(2)}s` : "-"; }
function formatDur(s)  { if (!s) return "-"; const m=Math.floor(s/60),r=Math.round(s%60); return m>0?`${m}m ${r}s`:`${r}s`; }
function activeModel() { return state.models.find(m=>m.name===state.form.model_name)||null; }
function imageSrc(r)   { return r?.image_base64 ? `data:image/png;base64,${r.image_base64}` : ""; }

function selectedImageStyle() { return IMAGE_STYLES.find(s=>s.id===state.form.style_id)||IMAGE_STYLES[0]; }
function selectedVideoStyle() { return VIDEO_STYLES.find(s=>s.id===state.videoForm.style_id)||VIDEO_STYLES[0]; }
function styledPrompt(prompt, style) {
  const base = (prompt || "").trim();
  if (!style?.suffix || !base) return base;
  return `${base}, ${style.suffix}`;
}
function tabFromPath(path = window.location.pathname) {
  if (path.startsWith("/video")) return "video";
  if (path.startsWith("/queue")) return "queue";
  return "image";
}

function statusBadge(status) {
  const map={
    queued:     {cls:"badge-queued",    icon:"", text:"Queued"},
    processing: {cls:"badge-processing",icon:"", text:"Processing"},
    completed:  {cls:"badge-done",      icon:"", text:"Done"},
    failed:     {cls:"badge-fail",      icon:"", text:"Failed"},
    cancelled:  {cls:"badge-cancel",    icon:"", text:"Cancelled"},
  };
  const s=map[status]||{cls:"",icon:"?",text:status};
  return `<span class="status-badge ${s.cls}">${s.icon ? `${s.icon} ` : ""}${s.text}</span>`;
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

/* â”€â”€â”€ API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */

function authHeaders(headers = {}) {
  const merged = {...headers};
  if (state.auth.apiKey) merged["X-API-Key"] = state.auth.apiKey;
  return merged;
}

async function fetchJson(url, opts = {}) {
  const r = await fetch(url, {...opts, headers: authHeaders(opts.headers || {})});
  const d = await r.json();
  return { response: r, data: d };
}

async function loadAuthStatus() {
  const before = JSON.stringify(state.auth);
  try {
    const { data } = await fetchJson("/api/auth/status");
    state.auth.required = Boolean(data.api_key_required);
    state.auth.configuredKeyCount = data.configured_key_count || 0;
  } catch {
    state.auth.required = false;
    state.auth.configuredKeyCount = 0;
  }
  if (JSON.stringify(state.auth) !== before) render();
}

async function loadHealth() {
  const before = state.health?.status || "";
  try {
    const { data } = await fetchJson("/health");
    state.health = data;
  } catch { state.health = null; }
  const after = state.health?.status || "";
  if (before !== after) render();
}

async function loadModels() {
  try {
    const { data } = await fetchJson("/models");
    state.models      = (data.models||[]).filter(m=>m.category==="image");
    state.videoModels = (data.models||[]).filter(m=>m.category==="video");
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
  const before = JSON.stringify({
    jobs: state.jobs.map(j=>[j.job_id,j.status,j.progress,j.processing_time_ms,j.error_message]),
    queue: state.queueStats,
  });
  try {
    const { data } = await fetchJson("/api/jobs?limit=30");
    state.jobs = data.jobs||[];
  } catch { state.jobs=[]; }
  try {
    const { data } = await fetchJson("/api/queue/status");
    state.queueStats = data;
  } catch {}
  const after = JSON.stringify({
    jobs: state.jobs.map(j=>[j.job_id,j.status,j.progress,j.processing_time_ms,j.error_message]),
    queue: state.queueStats,
  });
  return before !== after;
}

/* â”€â”€â”€ Image â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */

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
    const r = await fetch("/loras/upload", { method:"POST", headers:authHeaders(), body:form });
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
    const r = await fetch("/api/video/loras/upload", { method:"POST", headers:authHeaders(), body:form });
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
  const style = selectedImageStyle();
  const payload = {
    prompt: styledPrompt(state.form.prompt, style),
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
    const { response, data } = await fetchJson("/generate",{
      method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload),
    });
    if (!response.ok) {
      state.error = response.status === 401
        ? "Invalid or missing API key. Enter the server API key in the API Key field."
        : (data?.detail||"Generation failed.");
    }
    else {
      const r = {...data, model_name:state.form.model_name, prompt:state.form.prompt};
      state.result = r;
      state.history = [r,...state.history].slice(0,16);
    }
  } catch { state.error="Request failed. Check server/network."; }
  state.loading=false; render(); loadHealth();
}

/* â”€â”€â”€ Video â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */

async function onVideoGenerate(e) {
  e.preventDefault(); state.error="";
  const f = state.videoForm;
  const style = selectedVideoStyle();
  const payload = {
    prompt: styledPrompt(f.prompt, style),
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
    if (!response.ok) {
      state.error = response.status === 401
        ? "Invalid or missing API key. Enter the server API key in the API Key field."
        : (data?.detail||"Video generation failed.");
      render(); return;
    }
    startJobSSE(data.job_id);
  } catch { state.error="Request failed. Check server/network."; }
  render();
  await loadJobs(); render();
}

/* â”€â”€â”€ SSE Job Streaming â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */

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

/* â”€â”€â”€ Polling fallback â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */

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

function setTab(t, opts = {}) {
  const next = TAB_PATHS[t] ? t : "image";
  state.activeTab = next;
  state.error = "";
  const nextPath = TAB_PATHS[next];
  if (opts.push !== false && window.location.pathname !== nextPath) {
    history.pushState({tab: next}, "", nextPath);
  }
  render();
  if (opts.scroll !== false) {
    requestAnimationFrame(()=>document.querySelector(".page-content")?.scrollIntoView({behavior:"smooth", block:"start"}));
  }
}

function useSamplePrompt(prompt) {
  state.activeTab = "image";
  state.form.prompt = prompt;
  state.error = "";
  if (window.location.pathname !== TAB_PATHS.image) {
    history.pushState({tab: "image"}, "", TAB_PATHS.image);
  }
  render();
  const form = document.getElementById("generate-form");
  const promptEl = document.getElementById("prompt");
  form?.scrollIntoView({behavior:"smooth", block:"start"});
  if (form) {
    form.classList.add("focus-pulse");
    setTimeout(()=>form.classList.remove("focus-pulse"), 900);
  }
  promptEl?.focus({preventScroll:true});
}

function renderSampleShowcase() {
  return `
    <div class="hero-showcase">
      ${SAMPLE_IMAGES.map((sample, i)=>`
        <button type="button" class="sample-tile sample-${i+1}" data-sample-prompt="${escapeHtml(sample.prompt)}" title="${escapeHtml(sample.title)}">
          <img src="${sample.src}" alt="${escapeHtml(sample.title)}" loading="lazy" />
          <span>${escapeHtml(sample.title)}</span>
        </button>`).join("")}
    </div>`;
}

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   RENDER
   â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */

function render() {
  const cur = activeModel();
  const curImg = imageSrc(state.result);
  const tab = state.activeTab;
  const focused = document.activeElement;
  const focusSnapshot = focused && root.contains(focused) && focused.id ? {
    id: focused.id,
    start: focused.selectionStart,
    end: focused.selectionEnd,
    direction: focused.selectionDirection,
  } : null;

  root.innerHTML = `
    <div class="app-shell">
      <div class="ambient ambient-left"></div>
      <div class="ambient ambient-right"></div>
      <div class="app">

        <!-- HERO -->
        <section class="hero premium-card">
          <div class="hero-copy">
            <div class="brand-lockup">
              <span class="brand-mark">H</span>
              <div><div class="eyebrow">Hyperforge AI</div><strong>Creative Studio</strong></div>
            </div>
            <h1>Forge luminous images and motion.</h1>
            <p class="subtitle">A brighter workspace for image generation, LoRA experiments, motion tests, and polished creative previews.</p>
            <div class="hero-actions">
              <button type="button" class="hero-action primary" data-sample-prompt="${escapeHtml(SAMPLE_IMAGES[0].prompt)}">Try a Sample</button>
              <a class="hero-action" href="/queue" data-tab="queue">View Queue</a>
            </div>
          </div>
          <div class="hero-aside">
            ${renderSampleShowcase()}
            <div class="hero-status premium-card inner-card">
            <div class="status-top">
              <span class="dot ${state.health?.status==="healthy"?"ok":""}"></span>
              <span>${escapeHtml(state.health?.status==="healthy"?"System Ready":(state.health?.status||"Starting..."))}</span>
            </div>
            <div class="status-grid">
              <div><label>Studio</label><strong>${state.health?.status==="healthy"?"Ready":"Warming up"}</strong></div>
              <div><label>Queue</label><strong>${state.queueStats.queued + state.queueStats.processing} active</strong></div>
              <div><label>Styles</label><strong>${IMAGE_STYLES.length + VIDEO_STYLES.length} presets</strong></div>
            </div>
            ${state.auth.required ? `
              <div class="auth-field">
                <label for="api-key">API Key</label>
                <input id="api-key" type="password" autocomplete="off" placeholder="Required for image generation" value="${escapeHtml(state.auth.apiKey)}" />
              </div>` : ""}
            </div>
          </div>
        </section>

        <!-- TAB NAV -->
        <nav class="tab-nav">
          <a class="tab-btn ${tab==="image"?"active":""}" href="/image" data-tab="image">Image</a>
          <a class="tab-btn ${tab==="video"?"active":""}" href="/video" data-tab="video">Video</a>
          <a class="tab-btn ${tab==="queue"?"active":""}" href="/queue" data-tab="queue">Jobs
            <span class="job-count">${state.jobs.filter(j=>j.status==="queued"||j.status==="processing").length||""}</span>
          </a>
        </nav>

        <main class="page-content" data-page="${escapeHtml(tab)}">
          ${tab==="image"     ? renderImageTab(cur,curImg) : ""}
          ${tab==="video"     ? renderVideoTab()           : ""}
          ${tab==="queue"     ? renderQueueTab()           : ""}
        </main>

        ${state.error ? `<div class="global-error">${escapeHtml(state.error)}</div>` : ""}
      </div>
      ${state.lightbox ? `<div class="lightbox" id="lightbox"><img src="${state.lightbox}" alt="Preview" /></div>` : ""}
    </div>`;

  bindEvents();
  restoreFocus(focusSnapshot);
}

function restoreFocus(snapshot) {
  if (!snapshot) return;
  const el = document.getElementById(snapshot.id);
  if (!el) return;
  el.focus({preventScroll:true});
  if (typeof el.setSelectionRange === "function" && snapshot.start != null && snapshot.end != null) {
    try {
      el.setSelectionRange(snapshot.start, snapshot.end, snapshot.direction || "none");
    } catch {}
  }
}

/* â”€â”€â”€ IMAGE TAB â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */

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
          <textarea id="prompt" placeholder="Cinematic portrait, neon cityscape, hyperrealistic..." maxlength="2000" spellcheck="true" required>${escapeHtml(state.form.prompt)}</textarea>
          <div class="prompt-tools">
            <span>${escapeHtml(state.form.prompt.length)} / 2000</span>
            ${SAMPLE_IMAGES.slice(0,3).map(sample=>`<button type="button" data-sample-prompt="${escapeHtml(sample.prompt)}">${escapeHtml(sample.title)}</button>`).join("")}
          </div>
        </div>
        <div class="field"><label>Style</label>
          <div class="style-grid">${IMAGE_STYLES.map(style=>`
            <button type="button" class="style-card ${state.form.style_id===style.id?"active":""}" data-image-style="${escapeHtml(style.id)}">
              <strong>${escapeHtml(style.label)}</strong><span>${escapeHtml(style.suffix)}</span>
            </button>`).join("")}</div>
        </div>
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
                ${state.loraUploadStatus==="uploading"?"...":"Upload"}
              </label>
            </div>
            ${state.loraUploadStatus&&state.loraUploadStatus!=="uploading"?`<div class="lora-upload-msg ${state.loraUploadStatus.startsWith("ok")?"ok":"err"}">${escapeHtml(state.loraUploadStatus.startsWith("ok:")?"Uploaded: "+state.loraUploadStatus.slice(3):state.loraUploadStatus.slice(4))}</div>`:""}
          </div>
          <div class="field"><label>LoRA Scale</label>
            <input id="lora_scale" type="number" min="0" max="2" step="0.05" value="${escapeHtml(state.form.lora_scale)}" /></div>
        </div>

        <div class="field"><label>Resolution</label>
          <div class="preset-row">${RESOLUTION_PRESETS.map(p=>{
            const a=state.form.width===p.width&&state.form.height===p.height;
            return `<button type="button" class="preset-chip ${a?"active":""}" data-width="${p.width}" data-height="${p.height}">${escapeHtml(p.label)}<small>${p.width}x${p.height}</small></button>`;
          }).join("")}</div></div>

        <div class="field-group four-col">
          <div class="field"><label>W</label><input id="width" type="number" min="256" max="2048" step="8" value="${escapeHtml(state.form.width)}" /></div>
          <div class="field"><label>H</label><input id="height" type="number" min="256" max="2048" step="8" value="${escapeHtml(state.form.height)}" /></div>
          <div class="field"><label>Steps</label><input id="num_inference_steps" type="number" min="${currentModel?.min_steps||1}" max="${currentModel?.max_steps||50}" value="${escapeHtml(state.form.num_inference_steps)}" /></div>
          <div class="field"><label>CFG</label><input id="guidance_scale" type="number" min="0" max="20" step="0.5" value="${escapeHtml(state.form.guidance_scale)}" /></div>
        </div>

        <div class="field-group two-col align-end">
          <div class="field"><label>Seed</label><input id="seed" value="${escapeHtml(state.form.seed)}" placeholder="blank = random" /></div>
          <button class="primary-button" type="submit" ${state.loading?"disabled":""}>${state.loading?"Generating...":"Generate Image"}</button>
        </div>

        ${currentModel?`<div class="model-note"><strong>${escapeHtml(currentModel.name)}</strong><span>${escapeHtml(currentModel.description||"")}</span></div>`:""}
      </form>

      <section class="output-column">
        <div class="premium-card output-panel">
          <div class="section-heading compact-heading">
            <div><div class="eyebrow">Output</div><h2>Preview</h2></div>
            <div class="output-actions">
              ${currentImage&&!state.loading?`<button type="button" class="download-btn" id="download-btn">Download</button>`:""}
              <div class="metric-inline"><span>${escapeHtml(state.result?.model_name||state.health?.current_model||"No model")}</span></div>
            </div>
          </div>
          <div class="canvas" id="canvas">
            ${state.loading?`<div class="loading-spinner"><div class="spinner"></div><span>Generating image...</span></div>`
              :currentImage?`<img src="${currentImage}" alt="Generated" />`
              :`<span class="empty-state">Your image will appear here.</span>`}
          </div>
          <div class="result-grid">
            <div class="result-card"><label>Seed</label><strong>${escapeHtml(state.result?.seed_used??"-")}</strong></div>
            <div class="result-card"><label>Time</label><strong>${escapeHtml(formatMs(state.result?.inference_time_ms))}</strong></div>
            <div class="result-card"><label>Size</label><strong>${state.result?`${state.form.width}x${state.form.height}`:"-"}</strong></div>
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

/* â”€â”€â”€ VIDEO TAB â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */

function renderVideoTab() {
  const f = state.videoForm;
  const latestVideoJob = state.jobs.find(j=>j.job_type==="video"&&
    (j.status==="completed"||j.status==="processing"||j.status==="queued"));

  return `
    <div class="layout">
      <form class="premium-card control-panel" id="video-form">
        <div class="section-heading"><div><div class="eyebrow">Video Generation</div><h2>Text to Video</h2></div></div>

        <div class="field"><label>Prompt</label>
          <textarea id="v-prompt" placeholder="A camera slowly pans across a futuristic city at golden hour..." required>${escapeHtml(f.prompt)}</textarea></div>

        <div class="field"><label>Motion Style</label>
          <div class="style-grid video-style-grid">${VIDEO_STYLES.map(style=>`
            <button type="button" class="style-card ${f.style_id===style.id?"active":""}" data-video-style="${escapeHtml(style.id)}">
              <strong>${escapeHtml(style.label)}</strong><span>${escapeHtml(style.suffix)}</span>
            </button>`).join("")}</div>
        </div>

        <div class="field"><label>Negative Prompt</label>
          <textarea id="v-neg" class="compact" placeholder="blurry, static, low motion, worst quality">${escapeHtml(f.negative_prompt||"")}</textarea></div>

        <div class="field"><label>Model</label>
          <select id="v-model">${state.videoModels.map(m=>`
            <option value="${escapeHtml(m.name)}" ${f.model_name===m.name?"selected":""}>${escapeHtml(m.name)} - ${escapeHtml(m.description||"")}</option>`).join("")}
          </select></div>

        <div class="field-group two-col">
          <div class="field"><label>Resolution</label>
            <select id="v-resolution">
              <option value="480p" ${f.resolution==="480p"?"selected":""}>480p (848x480)</option>
              <option value="720p" ${f.resolution==="720p"?"selected":""}>720p (1280x720)</option>
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
              ${state.videoLoraUploadStatus==="uploading"?"...":"Upload"}
            </label>
          </div>
          ${state.videoLoraUploadStatus&&state.videoLoraUploadStatus!=="uploading"?`<div class="lora-upload-msg ${state.videoLoraUploadStatus.startsWith("ok")?"ok":"err"}">${escapeHtml(state.videoLoraUploadStatus.startsWith("ok:")?"Uploaded: "+state.videoLoraUploadStatus.slice(3):state.videoLoraUploadStatus.slice(4))}</div>`:""}
        </div>
        <div class="field"><label>LoRA Scale</label>
          <input id="v-lora-scale" type="number" min="0" max="2" step="0.05" value="${f.lora_scale}" /></div>

        <div class="field"><label>Image-to-Video source (optional)</label>
          <div class="dropzone" id="v-dropzone">
            <input type="file" id="v-source-img" accept="image/*" style="display:none" />
            ${f.source_image_b64
              ?`<div class="dropzone-preview"><img src="data:image/png;base64,${f.source_image_b64}" alt="Source" /><button type="button" class="dropzone-clear" id="v-clear-img">x</button></div>`
              :`<div class="dropzone-label" id="v-dropzone-label"><span>Drop or click to upload source image</span></div>`}
          </div></div>

        <button class="primary-button" type="submit">Generate Video</button>
      </form>

      <section class="output-column">
        <div class="premium-card output-panel">
          <div class="section-heading compact-heading">
            <div><div class="eyebrow">Output</div><h2>Video Preview</h2></div>
            ${latestVideoJob?.result?.video_url?`<a class="download-btn" href="${latestVideoJob.result.video_url}" download>Download</a>`:""}
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
        <div class="result-card"><label>Frames</label><strong>${job.result.num_frames||"-"}</strong></div>
        <div class="result-card"><label>Time</label><strong>${formatMs(job.result.inference_time_ms)}</strong></div>
      </div>`;
  }
  if (job.status==="processing"||job.status==="queued") {
    const pct = job.progress||0;
    const label = job.status==="queued"?"Waiting in queue...":`Generating video... ${pct.toFixed(0)}%`;
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

/* â”€â”€â”€ QUEUE TAB â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */

function renderQueueTab() {
  const s = state.queueStats;
  return `
    <div class="queue-layout">
      <div class="premium-card queue-stats-card">
        <div class="section-heading"><div><div class="eyebrow">Studio Queue</div><h2>Creation Activity</h2></div></div>
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
                ${job.status==="completed"&&job.result?.video_url?`<a class="job-result-link" href="${job.result.video_url}" target="_blank">View Video</a>`:""}
                ${job.status==="completed"&&job.result?.audio_url?`<a class="job-result-link" href="${job.result.audio_url}" target="_blank">Play Audio</a>`:""}
                ${job.status==="failed"&&job.error_message?`<div class="job-error">${escapeHtml(job.error_message)}</div>`:""}
              </div>`).join("")}
          </div>`}
      </div>
    </div>`;
}

/* â”€â”€â”€ Shared: recent jobs panel â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */

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
            ${job.status==="completed"&&job.result?.video_url?`<a class="job-result-link" href="${job.result.video_url}" target="_blank">Play</a>`:""}
            ${job.status==="completed"&&job.result?.audio_url?`<a class="job-result-link" href="${job.result.audio_url}" target="_blank">Play</a>`:""}
            ${job.status==="failed"?`<div class="job-error">${escapeHtml(job.error_message||"Failed")}</div>`:""}
          </div>`).join("")}
      </div>
    </div>`;
}

/* â”€â”€â”€ Event Binding â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */

function bindEvents() {
  bindInput("api-key", v=>{
    state.auth.apiKey = v.trim();
    localStorage.setItem("ncs_api_key", state.auth.apiKey);
  });

  // Tab navigation
  document.querySelectorAll("[data-tab]").forEach(btn=>{
    btn.addEventListener("click", e=>{
      e.preventDefault();
      setTab(btn.dataset.tab);
    });
  });

  document.querySelectorAll("[data-sample-prompt]").forEach(btn=>{
    btn.addEventListener("click", ()=>useSamplePrompt(btn.dataset.samplePrompt));
  });

  document.querySelectorAll("[data-image-style]").forEach(btn=>{
    btn.addEventListener("click", ()=>{
      state.form.style_id = btn.dataset.imageStyle;
      render();
      document.getElementById("generate-form")?.classList.add("focus-pulse");
      setTimeout(()=>document.getElementById("generate-form")?.classList.remove("focus-pulse"), 700);
    });
  });

  document.querySelectorAll("[data-video-style]").forEach(btn=>{
    btn.addEventListener("click", ()=>{
      state.videoForm.style_id = btn.dataset.videoStyle;
      render();
      document.getElementById("video-form")?.classList.add("focus-pulse");
      setTimeout(()=>document.getElementById("video-form")?.classList.remove("focus-pulse"), 700);
    });
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

/* â”€â”€â”€ Init â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */

async function init() {
  state.activeTab = tabFromPath();
  render();
  await Promise.all([loadAuthStatus(), loadHealth(), loadModels(), loadJobs()]);
  render();
  // Periodic refresh
  setInterval(loadHealth, 15000);
  setInterval(async ()=>{ if (await loadJobs()) render(); }, 8000);
  // Re-connect any active jobs on page reload
  state.jobs.filter(j=>j.status==="processing"||j.status==="queued")
    .forEach(j=>startJobSSE(j.job_id));
}

window.addEventListener("popstate", ()=>{
  state.activeTab = tabFromPath();
  state.error = "";
  render();
});

init();
