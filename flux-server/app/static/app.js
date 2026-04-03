/* ═══════════════════════════════════════════════════════════
   Neural Creation Studio — Multi-Modal AI Frontend
   Image · Video · Music · Animation
   ═══════════════════════════════════════════════════════════ */

const RESOLUTION_PRESETS = [
  { label: "Square", width: 1024, height: 1024 },
  { label: "Portrait", width: 896, height: 1152 },
  { label: "Landscape", width: 1152, height: 896 },
  { label: "Wide", width: 1344, height: 768 },
];

const VIDEO_FRAME_PRESETS = [
  { label: "1s", frames: 16 },
  { label: "2s", frames: 33 },
  { label: "3s", frames: 49 },
  { label: "5s", frames: 81 },
];

const GENRE_PILLS = [
  "Pop", "Rock", "Hip Hop", "Electronic", "Jazz", "R&B",
  "Classical", "Lo-fi", "Ambient", "Cinematic", "Metal", "Country",
];

const state = {
  activeTab: "image",
  models: [],
  health: null,
  loras: ["None"],
  categories: [],
  // Image
  history: [],
  loading: false,
  error: "",
  result: null,
  lightbox: null,
  loraUploadStatus: null,
  form: {
    prompt: "",
    negative_prompt: "",
    model_name: "flux-1-dev",
    width: 1024,
    height: 1024,
    num_inference_steps: 28,
    guidance_scale: 3.5,
    seed: "",
    lora_name: "None",
    lora_scale: 0.85,
  },
  // Video
  videoModels: [],
  videoForm: {
    prompt: "",
    negative_prompt: "",
    model_name: "ltx-video",
    resolution: "480p",
    num_frames: 33,
    fps: 16,
    guidance_scale: 5.0,
    num_inference_steps: 30,
    seed: "",
    source_image_b64: null,
    lora_name: "None",
    lora_scale: 1.0,
  },
  videoSourceName: "",
  // Music
  musicModels: [],
  musicForm: {
    prompt: "",
    model_name: "audioldm2",
    duration_seconds: 30,
    lyrics: "",
    genre: "",
    bpm: "",
    seed: "",
  },
  // Animation
  animModels: [],
  animForm: {
    model_name: "echomimic",
    expression_scale: 1.0,
    pose_style: 0,
    use_enhancer: false,
    source_image_b64: null,
    audio_b64: null,
  },
  animSourceName: "",
  animAudioName: "",
  // Jobs
  jobs: [],
  queueStats: { queued: 0, processing: 0, completed: 0, failed: 0, total: 0 },
};

const root = document.getElementById("root");

/* ─── Utilities ─────────────────────────────────────── */

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function formatSeconds(ms) {
  if (!ms) return "-";
  return `${(ms / 1000).toFixed(2)}s`;
}

function formatDuration(secs) {
  if (!secs) return "-";
  const m = Math.floor(secs / 60);
  const s = Math.round(secs % 60);
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

function activeModel() {
  return state.models.find((m) => m.name === state.form.model_name) || null;
}

function imageSrc(result) {
  if (!result || !result.image_base64) return "";
  return `data:image/png;base64,${result.image_base64}`;
}

function statusBadge(status) {
  const map = {
    queued: { icon: "◎", cls: "badge-queued", text: "In Queue" },
    processing: { icon: "⟳", cls: "badge-processing", text: "Processing" },
    completed: { icon: "✓", cls: "badge-done", text: "Complete" },
    failed: { icon: "✕", cls: "badge-fail", text: "Failed" },
    cancelled: { icon: "—", cls: "badge-cancel", text: "Cancelled" },
  };
  const s = map[status] || { icon: "?", cls: "", text: status };
  return `<span class="status-badge ${s.cls}">${s.icon} ${s.text}</span>`;
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result.split(",")[1]);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

/* ─── API ───────────────────────────────────────────── */

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  return { response, data };
}

async function loadHealth() {
  try {
    const { data } = await fetchJson("/health");
    state.health = data;
  } catch {
    state.health = null;
  }
  render();
}

async function loadModels() {
  try {
    const { data } = await fetchJson("/models");
    state.models = (data.models || []).filter((m) => m.category === "image");
    state.videoModels = (data.models || []).filter((m) => m.category === "video");
    state.musicModels = (data.models || []).filter((m) => m.category === "music");
    state.animModels = (data.models || []).filter((m) => m.category === "animation");
    state.categories = data.categories || [];

    const currentModel = data.current_model || state.models[0]?.name || "flux-1-dev";
    const info = state.models.find((m) => m.name === currentModel);
    state.form.model_name = currentModel;
    if (info) {
      state.form.num_inference_steps = info.default_steps ?? state.form.num_inference_steps;
      state.form.guidance_scale = info.default_guidance_scale ?? state.form.guidance_scale;
    }
    await loadLoras(currentModel);
  } catch {
    state.models = [];
    render();
  }
}

async function loadLoras(modelName) {
  try {
    const { data } = await fetchJson(`/loras?model_name=${encodeURIComponent(modelName)}`);
    state.loras = ["None", ...(data.loras || [])];
    state.form.lora_name = state.loras.includes(state.form.lora_name) ? state.form.lora_name : "None";
    state.form.lora_scale = data.recommended_scale ?? state.form.lora_scale;
  } catch {
    state.loras = ["None"];
  }
  render();
}

async function loadJobs() {
  try {
    const { data } = await fetchJson("/api/jobs?limit=20");
    state.jobs = data.jobs || [];
  } catch {
    state.jobs = [];
  }
  try {
    const { data } = await fetchJson("/api/queue/status");
    state.queueStats = data;
  } catch {}
}

/* ─── Image actions ─────────────────────────────────── */

function applyModel(name) {
  const info = state.models.find((m) => m.name === name);
  state.form.model_name = name;
  state.form.lora_name = "None";
  if (info) {
    state.form.num_inference_steps = info.default_steps ?? state.form.num_inference_steps;
    state.form.guidance_scale = info.default_guidance_scale ?? state.form.guidance_scale;
  }
  render();
  loadLoras(name);
}

function applyResolution(width, height) {
  state.form.width = width;
  state.form.height = height;
  render();
}

function downloadImage() {
  if (!state.result?.image_base64) return;
  const a = document.createElement("a");
  a.href = `data:image/png;base64,${state.result.image_base64}`;
  a.download = `image_${state.result.seed_used || Date.now()}.png`;
  a.click();
}

async function uploadLora(file) {
  state.loraUploadStatus = "uploading";
  render();
  const form = new FormData();
  form.append("file", file);
  try {
    const res = await fetch("/loras/upload", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) {
      state.loraUploadStatus = "error:" + (data?.detail || "Upload failed");
    } else {
      state.loraUploadStatus = "ok:" + data.filename;
      await loadLoras(state.form.model_name);
    }
  } catch {
    state.loraUploadStatus = "error:Network error";
  }
  render();
  setTimeout(() => { state.loraUploadStatus = null; render(); }, 4000);
}

async function onGenerate(event) {
  event.preventDefault();
  state.error = "";
  state.loading = true;
  render();

  const payload = {
    prompt: state.form.prompt,
    negative_prompt: state.form.negative_prompt || null,
    model_name: state.form.model_name,
    width: Number(state.form.width),
    height: Number(state.form.height),
    num_inference_steps: Number(state.form.num_inference_steps),
    guidance_scale: Number(state.form.guidance_scale),
    seed: state.form.seed === "" ? null : Number(state.form.seed),
    lora_name: !state.form.lora_name || state.form.lora_name === "None" ? null : state.form.lora_name,
    lora_scale: Number(state.form.lora_scale),
    use_refiner: false,
  };

  try {
    const { response, data } = await fetchJson("/generate-ui", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      state.error = data?.detail || "Generation failed.";
      state.loading = false;
      render();
      return;
    }
    const nextResult = { ...data, model_name: state.form.model_name, prompt: state.form.prompt };
    state.result = nextResult;
    state.history = [nextResult, ...state.history].slice(0, 12);
  } catch {
    state.error = "Request failed. Check server/network.";
  }
  state.loading = false;
  render();
  loadHealth();
}

/* ─── Video actions ─────────────────────────────────── */

async function onVideoGenerate(event) {
  event.preventDefault();
  state.error = "";
  const f = state.videoForm;
  const payload = {
    prompt: f.prompt,
    negative_prompt: f.negative_prompt || null,
    model_name: f.model_name,
    resolution: f.resolution,
    num_frames: Number(f.num_frames),
    fps: Number(f.fps),
    guidance_scale: Number(f.guidance_scale),
    num_inference_steps: Number(f.num_inference_steps),
    seed: f.seed === "" ? null : Number(f.seed),
    source_image_b64: f.source_image_b64 || null,
    lora_name: !f.lora_name || f.lora_name === "None" ? null : f.lora_name,
    lora_scale: Number(f.lora_scale),
  };
  try {
    const { response, data } = await fetchJson("/api/video/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      state.error = data?.detail || "Video generation failed.";
      render();
      return;
    }
    startJobPolling(data.job_id);
  } catch {
    state.error = "Request failed. Check server/network.";
  }
  render();
  await loadJobs();
  render();
}

/* ─── Music actions ─────────────────────────────────── */

async function onMusicGenerate(event) {
  event.preventDefault();
  state.error = "";
  const f = state.musicForm;
  const payload = {
    prompt: f.prompt,
    model_name: f.model_name,
    duration_seconds: Number(f.duration_seconds),
    lyrics: f.lyrics || null,
    genre: f.genre || null,
    bpm: f.bpm === "" ? null : Number(f.bpm),
    seed: f.seed === "" ? null : Number(f.seed),
  };
  try {
    const { response, data } = await fetchJson("/api/music/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      state.error = data?.detail || "Music generation failed.";
      render();
      return;
    }
    startJobPolling(data.job_id);
  } catch {
    state.error = "Request failed. Check server/network.";
  }
  render();
  await loadJobs();
  render();
}

/* ─── Animation actions ─────────────────────────────── */

async function onAnimGenerate(event) {
  event.preventDefault();
  state.error = "";
  const f = state.animForm;
  if (!f.source_image_b64 || !f.audio_b64) {
    state.error = "Please upload both a face image and an audio file.";
    render();
    return;
  }
  const payload = {
    model_name: f.model_name,
    source_image_b64: f.source_image_b64,
    audio_b64: f.audio_b64,
    expression_scale: Number(f.expression_scale),
    pose_style: Number(f.pose_style),
    use_enhancer: f.use_enhancer,
  };
  try {
    const { response, data } = await fetchJson("/api/animation/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      state.error = data?.detail || "Animation generation failed.";
      render();
      return;
    }
    startJobPolling(data.job_id);
  } catch {
    state.error = "Request failed. Check server/network.";
  }
  render();
  await loadJobs();
  render();
}

/* ─── Job polling ───────────────────────────────────── */

const _pollingTimers = {};

function startJobPolling(jobId) {
  if (_pollingTimers[jobId]) return;
  _pollingTimers[jobId] = setInterval(async () => {
    try {
      const { data } = await fetchJson(`/api/jobs/${jobId}`);
      // Update job in state
      const idx = state.jobs.findIndex((j) => j.job_id === jobId);
      if (idx >= 0) state.jobs[idx] = data;
      else state.jobs.unshift(data);

      if (data.status === "completed" || data.status === "failed" || data.status === "cancelled") {
        clearInterval(_pollingTimers[jobId]);
        delete _pollingTimers[jobId];
      }
      render();
    } catch {
      clearInterval(_pollingTimers[jobId]);
      delete _pollingTimers[jobId];
    }
  }, 2000);
}

async function cancelJob(jobId) {
  try {
    await fetch(`/api/jobs/${jobId}`, { method: "DELETE" });
    if (_pollingTimers[jobId]) {
      clearInterval(_pollingTimers[jobId]);
      delete _pollingTimers[jobId];
    }
    await loadJobs();
    render();
  } catch {}
}

/* ─── Tab switching ─────────────────────────────────── */

function setTab(tab) {
  state.activeTab = tab;
  state.error = "";
  render();
}

/* ─── RENDER ────────────────────────────────────────── */

function render() {
  const currentModel = activeModel();
  const currentImage = imageSrc(state.result);
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
              <span class="dot ${state.health?.status === "healthy" ? "ok" : ""}"></span>
              <span>${escapeHtml(state.health?.status === "healthy" ? "Ready" : (state.health?.status || "Starting"))}</span>
            </div>
            <div class="status-grid">
              <div><label>Model</label><strong>${escapeHtml(state.health?.current_model || "-")}</strong></div>
              <div><label>Queue</label><strong>${state.queueStats.queued} queued · ${state.queueStats.processing} active</strong></div>
              <div><label>Status</label><strong>Online</strong></div>
            </div>
          </div>
        </section>

        <!-- TAB NAV -->
        <nav class="tab-nav">
          <button class="tab-btn ${tab === "image" ? "active" : ""}" data-tab="image">
            <span class="tab-icon">◆</span> Image
          </button>
          <button class="tab-btn ${tab === "video" ? "active" : ""}" data-tab="video">
            <span class="tab-icon">▶</span> Video
          </button>
          <button class="tab-btn ${tab === "music" ? "active" : ""}" data-tab="music">
            <span class="tab-icon">♫</span> Music
          </button>
          <button class="tab-btn ${tab === "animation" ? "active" : ""}" data-tab="animation">
            <span class="tab-icon">◎</span> Animation
          </button>
          <button class="tab-btn ${tab === "queue" ? "active" : ""}" data-tab="queue">
            <span class="tab-icon">☰</span> Jobs <span class="job-count">${state.jobs.filter((j) => j.status === "queued" || j.status === "processing").length || ""}</span>
          </button>
        </nav>

        <!-- TAB CONTENT -->
        ${tab === "image" ? renderImageTab(currentModel, currentImage) : ""}
        ${tab === "video" ? renderVideoTab() : ""}
        ${tab === "music" ? renderMusicTab() : ""}
        ${tab === "animation" ? renderAnimationTab() : ""}
        ${tab === "queue" ? renderQueueTab() : ""}

        ${state.error ? `<div class="global-error">${escapeHtml(state.error)}</div>` : ""}
      </div>
      ${state.lightbox ? `<div class="lightbox" id="lightbox"><img src="${state.lightbox}" alt="Preview" /></div>` : ""}
    </div>
  `;
  bindEvents();
}

/* ─── IMAGE TAB ─────────────────────────────────────── */

function renderImageTab(currentModel, currentImage) {
  return `
    <!-- MODEL STRIP -->
    <section class="model-strip">
      ${state.models.map((model) => `
        <button type="button" class="model-card premium-card ${state.form.model_name === model.name ? "active" : ""}" data-model="${escapeHtml(model.name)}">
          <div class="model-name">${escapeHtml(model.name)}</div>
          <div class="model-summary">${escapeHtml(model.summary || model.description || "Ready")}</div>
        </button>
      `).join("")}
    </section>

    <div class="layout">
      <form class="premium-card control-panel" id="generate-form">
        <div class="section-heading"><div><div class="eyebrow">Image Generation</div><h2>Compose</h2></div></div>

        <div class="field"><label for="prompt">Prompt</label><textarea id="prompt" name="prompt" placeholder="Describe your vision — cinematic portrait, neon cityscape..." required>${escapeHtml(state.form.prompt)}</textarea></div>
        <div class="field"><label for="negative_prompt">Negative Prompt</label><textarea id="negative_prompt" class="compact" name="negative_prompt" placeholder="blurry, low detail, watermark">${escapeHtml(state.form.negative_prompt)}</textarea></div>

        <div class="field-group three-col">
          <div class="field"><label for="model_name">Model</label><select id="model_name" name="model_name">${state.models.map((m) => `<option value="${escapeHtml(m.name)}" ${state.form.model_name === m.name ? "selected" : ""}>${escapeHtml(m.name)}</option>`).join("")}</select></div>
          <div class="field"><label for="lora_name">LoRA</label><div class="lora-row"><select id="lora_name" name="lora_name">${state.loras.map((item) => `<option value="${escapeHtml(item)}" ${state.form.lora_name === item ? "selected" : ""}>${escapeHtml(item)}</option>`).join("")}</select><label class="upload-lora-btn ${state.loraUploadStatus === "uploading" ? "uploading" : ""}" title="Upload .safetensors LoRA"><input type="file" id="lora-file-input" accept=".safetensors" style="display:none" />${state.loraUploadStatus === "uploading" ? "..." : "&#x2B06;"}</label></div>${state.loraUploadStatus && state.loraUploadStatus !== "uploading" ? `<div class="lora-upload-msg ${state.loraUploadStatus.startsWith("ok") ? "ok" : "err"}">${escapeHtml(state.loraUploadStatus.startsWith("ok:") ? "✓ Uploaded: " + state.loraUploadStatus.slice(3) : state.loraUploadStatus.slice(6))}</div>` : ""}</div>
          <div class="field"><label for="lora_scale">LoRA Scale</label><input id="lora_scale" name="lora_scale" type="number" min="0" max="2" step="0.05" value="${escapeHtml(state.form.lora_scale)}" /></div>
        </div>

        <div class="field"><label>Resolution</label><div class="preset-row">${RESOLUTION_PRESETS.map((p) => {
          const active = state.form.width === p.width && state.form.height === p.height;
          return `<button type="button" class="preset-chip ${active ? "active" : ""}" data-width="${p.width}" data-height="${p.height}">${escapeHtml(p.label)}<small>${p.width}×${p.height}</small></button>`;
        }).join("")}</div></div>

        <div class="field-group four-col">
          <div class="field"><label for="width">Width</label><input id="width" name="width" type="number" min="256" max="2048" step="8" value="${escapeHtml(state.form.width)}" /></div>
          <div class="field"><label for="height">Height</label><input id="height" name="height" type="number" min="256" max="2048" step="8" value="${escapeHtml(state.form.height)}" /></div>
          <div class="field"><label for="num_inference_steps">Steps</label><input id="num_inference_steps" name="num_inference_steps" type="number" min="${currentModel?.min_steps || 1}" max="${currentModel?.max_steps || 50}" value="${escapeHtml(state.form.num_inference_steps)}" /></div>
          <div class="field"><label for="guidance_scale">Guidance</label><input id="guidance_scale" name="guidance_scale" type="number" min="0" max="20" step="0.5" value="${escapeHtml(state.form.guidance_scale)}" /></div>
        </div>

        <div class="field-group two-col align-end">
          <div class="field"><label for="seed">Seed</label><input id="seed" name="seed" value="${escapeHtml(state.form.seed)}" placeholder="blank = random" /></div>
          <button class="primary-button" type="submit" ${state.loading ? "disabled" : ""}>${state.loading ? "Generating..." : "Generate Image"}</button>
        </div>

        ${currentModel ? `<div class="model-note"><strong>${escapeHtml(currentModel.name)}</strong><span>${escapeHtml(currentModel.description || "")}</span></div>` : ""}
      </form>

      <section class="output-column">
        <div class="premium-card output-panel">
          <div class="section-heading compact-heading"><div><div class="eyebrow">Output</div><h2>Preview</h2></div><div class="output-actions">${currentImage && !state.loading ? `<button type="button" class="download-btn" id="download-btn">&#x2B07; Download</button>` : ""}<div class="metric-inline"><span>${escapeHtml(state.result?.model_name || state.health?.current_model || "No model")}</span></div></div></div>
          <div class="canvas" id="canvas">${state.loading ? `<div class="loading-spinner"><div class="spinner"></div><span>Generating image...</span></div>` : currentImage ? `<img src="${currentImage}" alt="Generated" />` : `<span class="empty-state">Your generated image will appear here.</span>`}</div>
          <div class="result-grid">
            <div class="result-card"><label>Seed</label><strong>${escapeHtml(state.result?.seed_used ?? "-")}</strong></div>
            <div class="result-card"><label>Time</label><strong>${escapeHtml(formatSeconds(state.result?.inference_time_ms))}</strong></div>
            <div class="result-card wide"><label>Job</label><strong>${escapeHtml(state.result?.job_id || "-")}</strong></div>
          </div>
        </div>

        <div class="premium-card history-panel">
          <div class="section-heading compact-heading"><div><div class="eyebrow">History</div><h2>All generations (${state.history.length})</h2></div></div>
          <div class="history-grid">${state.history.length === 0 ? `<div class="history-empty">No generations yet.</div>` : state.history.map((item, i) => `<button type="button" class="history-item" data-history-index="${i}"><img src="data:image/png;base64,${item.image_base64}" alt="${escapeHtml(item.prompt || "")}" /><div class="history-meta"><strong>${escapeHtml(item.model_name)}</strong><span>${escapeHtml(formatSeconds(item.inference_time_ms))}</span></div></button>`).join("")}</div>
        </div>
      </section>
    </div>
  `;
}

/* ─── VIDEO TAB ─────────────────────────────────────── */

function renderVideoTab() {
  const f = state.videoForm;
  const latestVideoJob = state.jobs.find((j) => j.job_type === "video" && (j.status === "completed" || j.status === "processing"));

  return `
    <div class="layout">
      <form class="premium-card control-panel" id="video-form">
        <div class="section-heading"><div><div class="eyebrow">Video Generation</div><h2>Text to Video</h2></div></div>

        <div class="field"><label for="v-prompt">Prompt</label><textarea id="v-prompt" placeholder="A camera slowly pans across a futuristic city at sunset..." required>${escapeHtml(f.prompt)}</textarea></div>

        <div class="field"><label for="v-model">Model</label><select id="v-model">${state.videoModels.map((m) => `<option value="${escapeHtml(m.name)}" ${f.model_name === m.name ? "selected" : ""}>${escapeHtml(m.name)} — ${escapeHtml(m.description || "")}</option>`).join("")}</select></div>

        <div class="field-group two-col">
          <div class="field"><label for="v-resolution">Resolution</label><select id="v-resolution"><option value="480p" ${f.resolution === "480p" ? "selected" : ""}>480p (480×848)</option><option value="720p" ${f.resolution === "720p" ? "selected" : ""}>720p (720×1280)</option></select></div>
          <div class="field"><label for="v-fps">FPS</label><select id="v-fps"><option value="16" ${f.fps == 16 ? "selected" : ""}>16 fps</option><option value="24" ${f.fps == 24 ? "selected" : ""}>24 fps</option></select></div>
        </div>

        <div class="field"><label>Duration</label><div class="preset-row">${VIDEO_FRAME_PRESETS.map((p) => `<button type="button" class="preset-chip ${Number(f.num_frames) === p.frames ? "active" : ""}" data-vframes="${p.frames}">${p.label}<small>${p.frames} frames</small></button>`).join("")}</div></div>

        <div class="field-group two-col">
          <div class="field"><label for="v-steps">Steps</label><input id="v-steps" type="number" min="10" max="50" value="${f.num_inference_steps}" /></div>
          <div class="field"><label for="v-guidance">Guidance</label><input id="v-guidance" type="number" min="0" max="20" step="0.5" value="${f.guidance_scale}" /></div>
        </div>

        <div class="field"><label for="v-seed">Seed</label><input id="v-seed" value="${escapeHtml(f.seed)}" placeholder="blank = random" /></div>

        <div class="field-group two-col">
          <div class="field"><label for="v-lora">LoRA</label><select id="v-lora">${state.loras.map((item) => `<option value="${escapeHtml(item)}" ${f.lora_name === item ? "selected" : ""}>${escapeHtml(item)}</option>`).join("")}</select></div>
          <div class="field"><label for="v-lora-scale">LoRA Scale</label><input id="v-lora-scale" type="number" min="0" max="2" step="0.05" value="${f.lora_scale}" /></div>
        </div>

        <div class="field"><label>Image-to-Video (optional)</label>
          <div class="dropzone" id="v-dropzone">
            <input type="file" id="v-source-img" accept="image/*" style="display:none" />
            ${f.source_image_b64 ? `<div class="dropzone-preview"><img src="data:image/png;base64,${f.source_image_b64}" alt="Source" /><button type="button" class="dropzone-clear" id="v-clear-img">✕</button></div>` : `<div class="dropzone-label" id="v-dropzone-label"><span class="dropzone-icon">🖼️</span><span>Drop or click to upload source image</span></div>`}
          </div>
        </div>

        <button class="primary-button" type="submit">Generate Video</button>
      </form>

      <section class="output-column">
        <div class="premium-card output-panel">
          <div class="section-heading compact-heading"><div><div class="eyebrow">Output</div><h2>Video Preview</h2></div></div>
          ${latestVideoJob?.status === "completed" && latestVideoJob.result?.video_url ? `
            <div class="video-container"><video controls autoplay loop src="${latestVideoJob.result.video_url}"></video></div>
            <div class="result-grid">
              <div class="result-card"><label>Duration</label><strong>${formatDuration(latestVideoJob.result.duration_seconds)}</strong></div>
              <div class="result-card"><label>Frames</label><strong>${latestVideoJob.result.num_frames || "-"}</strong></div>
              <div class="result-card"><label>Time</label><strong>${formatSeconds(latestVideoJob.result.inference_time_ms)}</strong></div>
            </div>
          ` : latestVideoJob?.status === "processing" ? `
            <div class="canvas"><div class="loading-spinner"><div class="spinner"></div><span>Generating video... This may take several minutes.</span></div></div>
          ` : `
            <div class="canvas"><span class="empty-state">Submit a prompt to generate a video.</span></div>
          `}
        </div>
        ${renderRecentJobs("video")}
      </section>
    </div>
  `;
}

/* ─── MUSIC TAB ─────────────────────────────────────── */

function renderMusicTab() {
  const f = state.musicForm;
  const latestMusicJob = state.jobs.find((j) => j.job_type === "music" && (j.status === "completed" || j.status === "processing"));

  return `
    <div class="layout">
      <form class="premium-card control-panel" id="music-form">
        <div class="section-heading"><div><div class="eyebrow">Music Generation</div><h2>Create Music</h2></div></div>

        <div class="field"><label for="m-prompt">Prompt</label><textarea id="m-prompt" placeholder="Upbeat electronic track with synth pads and deep bass..." required>${escapeHtml(f.prompt)}</textarea></div>

        <div class="field"><label for="m-model">Model</label><select id="m-model">${state.musicModels.map((m) => `<option value="${escapeHtml(m.name)}" ${f.model_name === m.name ? "selected" : ""}>${escapeHtml(m.name)} — ${escapeHtml(m.description || "")}</option>`).join("")}</select></div>

        <div class="field"><label>Genre</label><div class="style-row">${GENRE_PILLS.map((g) => `<button type="button" class="style-chip ${f.genre === g ? "active" : ""}" data-genre="${escapeHtml(g)}">${escapeHtml(g)}</button>`).join("")}</div></div>

        <div class="field-group three-col">
          <div class="field"><label for="m-duration">Duration (s)</label><input id="m-duration" type="number" min="5" max="300" value="${f.duration_seconds}" /></div>
          <div class="field"><label for="m-bpm">BPM</label><input id="m-bpm" type="number" min="40" max="240" value="${escapeHtml(f.bpm)}" placeholder="auto" /></div>
          <div class="field"><label for="m-seed">Seed</label><input id="m-seed" value="${escapeHtml(f.seed)}" placeholder="random" /></div>
        </div>

        ${f.model_name === "ace-step" ? `
          <div class="field"><label for="m-lyrics">Lyrics (ACE-Step only)</label><textarea id="m-lyrics" class="compact" placeholder="[Verse]&#10;Walking through the city lights...&#10;[Chorus]&#10;We're alive tonight...">${escapeHtml(f.lyrics)}</textarea></div>
        ` : ""}

        <button class="primary-button" type="submit">Generate Music</button>
      </form>

      <section class="output-column">
        <div class="premium-card output-panel">
          <div class="section-heading compact-heading"><div><div class="eyebrow">Output</div><h2>Audio Preview</h2></div></div>
          ${latestMusicJob?.status === "completed" && latestMusicJob.result?.audio_url ? `
            <div class="audio-container">
              <audio controls src="${latestMusicJob.result.audio_url}" style="width:100%"></audio>
              <div class="result-grid" style="margin-top:14px">
                <div class="result-card"><label>Duration</label><strong>${formatDuration(latestMusicJob.result.duration_seconds)}</strong></div>
                <div class="result-card"><label>Sample Rate</label><strong>${latestMusicJob.result.sample_rate || "-"} Hz</strong></div>
                <div class="result-card"><label>Time</label><strong>${formatSeconds(latestMusicJob.result.inference_time_ms)}</strong></div>
              </div>
            </div>
          ` : latestMusicJob?.status === "processing" ? `
            <div class="canvas"><div class="loading-spinner"><div class="spinner"></div><span>Generating music...</span></div></div>
          ` : `
            <div class="canvas"><span class="empty-state">Submit a prompt to generate music or a song.</span></div>
          `}
        </div>
        ${renderRecentJobs("music")}
      </section>
    </div>
  `;
}

/* ─── ANIMATION TAB ─────────────────────────────────── */

function renderAnimationTab() {
  const f = state.animForm;
  const latestAnimJob = state.jobs.find((j) => j.job_type === "animation" && (j.status === "completed" || j.status === "processing"));

  return `
    <div class="layout">
      <form class="premium-card control-panel" id="anim-form">
        <div class="section-heading"><div><div class="eyebrow">Animation</div><h2>Audio-Driven Talking Head</h2></div></div>

        <div class="field"><label for="a-model">Model</label><select id="a-model">${state.animModels.map((m) => `<option value="${escapeHtml(m.name)}" ${f.model_name === m.name ? "selected" : ""}>${escapeHtml(m.name)} — ${escapeHtml(m.description || "")}</option>`).join("")}</select></div>

        <div class="field"><label>Face Image</label>
          <div class="dropzone" id="a-img-dropzone">
            <input type="file" id="a-source-img" accept="image/*" style="display:none" />
            ${f.source_image_b64 ? `<div class="dropzone-preview"><img src="data:image/png;base64,${f.source_image_b64}" alt="Face" /><button type="button" class="dropzone-clear" id="a-clear-img">✕</button></div>` : `<div class="dropzone-label" id="a-img-label"><span class="dropzone-icon">👤</span><span>Upload a clear face photo</span></div>`}
          </div>
        </div>

        <div class="field"><label>Audio File</label>
          <div class="dropzone" id="a-audio-dropzone">
            <input type="file" id="a-source-audio" accept="audio/*,.wav,.mp3,.ogg" style="display:none" />
            ${f.audio_b64 ? `<div class="dropzone-file"><span>🎵 ${escapeHtml(state.animAudioName || "Audio loaded")}</span><button type="button" class="dropzone-clear" id="a-clear-audio">✕</button></div>` : `<div class="dropzone-label" id="a-audio-label"><span class="dropzone-icon">🎙️</span><span>Upload speech or song audio (WAV/MP3)</span></div>`}
          </div>
        </div>

        <div class="field-group two-col">
          <div class="field"><label for="a-expr">Expression Scale</label><input id="a-expr" type="number" min="0.1" max="3.0" step="0.1" value="${f.expression_scale}" /></div>
          <div class="field"><label for="a-pose">Pose Style</label><input id="a-pose" type="number" min="0" max="46" value="${f.pose_style}" /></div>
        </div>

        <button class="primary-button" type="submit" ${!f.source_image_b64 || !f.audio_b64 ? "disabled" : ""}>Generate Animation</button>
      </form>

      <section class="output-column">
        <div class="premium-card output-panel">
          <div class="section-heading compact-heading"><div><div class="eyebrow">Output</div><h2>Animation Preview</h2></div></div>
          ${latestAnimJob?.status === "completed" && latestAnimJob.result?.video_url ? `
            <div class="video-container"><video controls autoplay loop src="${latestAnimJob.result.video_url}"></video></div>
            <div class="result-grid">
              <div class="result-card"><label>Duration</label><strong>${formatDuration(latestAnimJob.result.duration_seconds)}</strong></div>
              <div class="result-card"><label>Time</label><strong>${formatSeconds(latestAnimJob.result.inference_time_ms)}</strong></div>
            </div>
          ` : latestAnimJob?.status === "processing" ? `
            <div class="canvas"><div class="loading-spinner"><div class="spinner"></div><span>Generating animation...</span></div></div>
          ` : `
            <div class="canvas"><span class="empty-state">Upload a face image and audio to generate a talking head animation.</span></div>
          `}
        </div>
        ${renderRecentJobs("animation")}
      </section>
    </div>
  `;
}

/* ─── QUEUE TAB ─────────────────────────────────────── */

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
        ${state.jobs.length === 0 ? `<div class="empty-state" style="padding:24px">No jobs submitted yet. Generate something to see activity here.</div>` : `
          <div class="job-list">
            ${state.jobs.map((job) => `
              <div class="job-item ${job.status}">
                <div class="job-item-header">
                  <span class="job-type-badge">${escapeHtml(job.job_type)}</span>
                  ${statusBadge(job.status)}
                  <span class="job-model">${escapeHtml(job.model_name)}</span>
                  <span class="job-time">${formatSeconds(job.processing_time_ms)}</span>
                  ${job.status === "queued" ? `<button class="job-cancel-btn" data-cancel-job="${job.job_id}">Cancel</button>` : ""}
                </div>
                ${job.status === "processing" ? `<div class="job-progress"><div class="job-progress-bar" style="width:${job.progress || 5}%"></div></div>` : ""}
                ${job.status === "completed" && job.result?.video_url ? `<a href="${job.result.video_url}" target="_blank" class="job-result-link">▶ View video</a>` : ""}
                ${job.status === "completed" && job.result?.audio_url ? `<a href="${job.result.audio_url}" target="_blank" class="job-result-link">♫ Play audio</a>` : ""}
                ${job.status === "failed" ? `<div class="job-error">${escapeHtml(job.error_message)}</div>` : ""}
              </div>
            `).join("")}
          </div>
        `}
      </div>
    </div>
  `;
}

/* ─── Recent Jobs Sidebar ───────────────────────────── */

function renderRecentJobs(type) {
  const filtered = state.jobs.filter((j) => j.job_type === type).slice(0, 5);
  if (filtered.length === 0) return "";
  return `
    <div class="premium-card history-panel">
      <div class="section-heading compact-heading"><div><div class="eyebrow">Recent</div><h2>${escapeHtml(type)} jobs (${filtered.length})</h2></div></div>
      <div class="job-list compact">
        ${filtered.map((job) => `
          <div class="job-item ${job.status}">
            <div class="job-item-header">
              ${statusBadge(job.status)}
              <span class="job-model">${escapeHtml(job.model_name)}</span>
              <span class="job-time">${formatSeconds(job.processing_time_ms)}</span>
            </div>
            ${job.status === "processing" ? `<div class="job-progress"><div class="job-progress-bar" style="width:${job.progress || 5}%"></div></div>` : ""}
            ${job.status === "completed" && job.result?.video_url ? `<a href="${job.result.video_url}" target="_blank" class="job-result-link">▶ View</a>` : ""}
            ${job.status === "completed" && job.result?.audio_url ? `<a href="${job.result.audio_url}" target="_blank" class="job-result-link">♫ Play</a>` : ""}
          </div>
        `).join("")}
      </div>
    </div>
  `;
}

/* ─── EVENT BINDING ─────────────────────────────────── */

function bindEvents() {
  // Tab nav
  root.querySelectorAll("[data-tab]").forEach((btn) => {
    btn.addEventListener("click", () => setTab(btn.dataset.tab));
  });

  // Lightbox
  const lightbox = document.getElementById("lightbox");
  if (lightbox) lightbox.addEventListener("click", () => { state.lightbox = null; render(); });

  // Cancel job buttons
  root.querySelectorAll("[data-cancel-job]").forEach((btn) => {
    btn.addEventListener("click", () => cancelJob(btn.dataset.cancelJob));
  });

  // ─── Image tab events ───
  const genForm = document.getElementById("generate-form");
  if (genForm) {
    genForm.addEventListener("submit", onGenerate);

    ["prompt", "negative_prompt", "width", "height", "num_inference_steps", "guidance_scale", "seed", "lora_scale"].forEach((name) => {
      const el = document.getElementById(name);
      if (el) el.addEventListener("input", (e) => { state.form[name] = e.target.value; });
    });

    const modelSel = document.getElementById("model_name");
    if (modelSel) modelSel.addEventListener("change", (e) => applyModel(e.target.value));

    const loraSel = document.getElementById("lora_name");
    if (loraSel) loraSel.addEventListener("change", (e) => { state.form.lora_name = e.target.value; });

    root.querySelectorAll("[data-model]").forEach((btn) => {
      btn.addEventListener("click", () => applyModel(btn.dataset.model));
    });

    const dlBtn = document.getElementById("download-btn");
    if (dlBtn) dlBtn.addEventListener("click", downloadImage);

    const loraInput = document.getElementById("lora-file-input");
    if (loraInput) loraInput.addEventListener("change", (e) => { if (e.target.files[0]) uploadLora(e.target.files[0]); });

    root.querySelectorAll("[data-width]").forEach((btn) => {
      btn.addEventListener("click", () => applyResolution(Number(btn.dataset.width), Number(btn.dataset.height)));
    });

    const canvas = document.getElementById("canvas");
    if (canvas && state.result) {
      canvas.addEventListener("click", () => { state.lightbox = imageSrc(state.result); render(); });
    }

    root.querySelectorAll("[data-history-index]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const item = state.history[Number(btn.dataset.historyIndex)];
        if (item) { state.lightbox = `data:image/png;base64,${item.image_base64}`; render(); }
      });
    });
  }

  // ─── Video tab events ───
  const videoForm = document.getElementById("video-form");
  if (videoForm) {
    videoForm.addEventListener("submit", onVideoGenerate);
    const vp = document.getElementById("v-prompt");
    if (vp) vp.addEventListener("input", (e) => { state.videoForm.prompt = e.target.value; });
    const vm = document.getElementById("v-model");
    if (vm) vm.addEventListener("change", (e) => { state.videoForm.model_name = e.target.value; render(); });
    const vr = document.getElementById("v-resolution");
    if (vr) vr.addEventListener("change", (e) => { state.videoForm.resolution = e.target.value; });
    const vfps = document.getElementById("v-fps");
    if (vfps) vfps.addEventListener("change", (e) => { state.videoForm.fps = Number(e.target.value); });
    const vs = document.getElementById("v-steps");
    if (vs) vs.addEventListener("input", (e) => { state.videoForm.num_inference_steps = e.target.value; });
    const vg = document.getElementById("v-guidance");
    if (vg) vg.addEventListener("input", (e) => { state.videoForm.guidance_scale = e.target.value; });
    const vseed = document.getElementById("v-seed");
    if (vseed) vseed.addEventListener("input", (e) => { state.videoForm.seed = e.target.value; });
    const vlora = document.getElementById("v-lora");
    if (vlora) vlora.addEventListener("change", (e) => { state.videoForm.lora_name = e.target.value; });
    const vloraScale = document.getElementById("v-lora-scale");
    if (vloraScale) vloraScale.addEventListener("input", (e) => { state.videoForm.lora_scale = e.target.value; });

    root.querySelectorAll("[data-vframes]").forEach((btn) => {
      btn.addEventListener("click", () => { state.videoForm.num_frames = Number(btn.dataset.vframes); render(); });
    });

    // Video image upload
    const vDropzone = document.getElementById("v-dropzone-label") || document.getElementById("v-dropzone");
    const vInput = document.getElementById("v-source-img");
    if (vDropzone && vInput) {
      vDropzone.addEventListener("click", () => vInput.click());
      vInput.addEventListener("change", async (e) => {
        const file = e.target.files[0];
        if (file) {
          state.videoForm.source_image_b64 = await fileToBase64(file);
          state.videoSourceName = file.name;
          render();
        }
      });
    }
    const vClear = document.getElementById("v-clear-img");
    if (vClear) vClear.addEventListener("click", () => { state.videoForm.source_image_b64 = null; state.videoSourceName = ""; render(); });
  }

  // ─── Music tab events ───
  const musicForm = document.getElementById("music-form");
  if (musicForm) {
    musicForm.addEventListener("submit", onMusicGenerate);
    const mp = document.getElementById("m-prompt");
    if (mp) mp.addEventListener("input", (e) => { state.musicForm.prompt = e.target.value; });
    const mm = document.getElementById("m-model");
    if (mm) mm.addEventListener("change", (e) => { state.musicForm.model_name = e.target.value; render(); });
    const md = document.getElementById("m-duration");
    if (md) md.addEventListener("input", (e) => { state.musicForm.duration_seconds = e.target.value; });
    const mb = document.getElementById("m-bpm");
    if (mb) mb.addEventListener("input", (e) => { state.musicForm.bpm = e.target.value; });
    const ms = document.getElementById("m-seed");
    if (ms) ms.addEventListener("input", (e) => { state.musicForm.seed = e.target.value; });
    const ml = document.getElementById("m-lyrics");
    if (ml) ml.addEventListener("input", (e) => { state.musicForm.lyrics = e.target.value; });

    root.querySelectorAll("[data-genre]").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.musicForm.genre = state.musicForm.genre === btn.dataset.genre ? "" : btn.dataset.genre;
        render();
      });
    });
  }

  // ─── Animation tab events ───
  const animForm = document.getElementById("anim-form");
  if (animForm) {
    animForm.addEventListener("submit", onAnimGenerate);
    const am = document.getElementById("a-model");
    if (am) am.addEventListener("change", (e) => { state.animForm.model_name = e.target.value; });
    const ae = document.getElementById("a-expr");
    if (ae) ae.addEventListener("input", (e) => { state.animForm.expression_scale = e.target.value; });
    const ap = document.getElementById("a-pose");
    if (ap) ap.addEventListener("input", (e) => { state.animForm.pose_style = e.target.value; });

    // Face image upload
    const aImgDz = document.getElementById("a-img-label") || document.getElementById("a-img-dropzone");
    const aImgIn = document.getElementById("a-source-img");
    if (aImgDz && aImgIn) {
      aImgDz.addEventListener("click", () => aImgIn.click());
      aImgIn.addEventListener("change", async (e) => {
        const file = e.target.files[0];
        if (file) {
          state.animForm.source_image_b64 = await fileToBase64(file);
          state.animSourceName = file.name;
          render();
        }
      });
    }
    const aClearImg = document.getElementById("a-clear-img");
    if (aClearImg) aClearImg.addEventListener("click", () => { state.animForm.source_image_b64 = null; render(); });

    // Audio upload
    const aAudioDz = document.getElementById("a-audio-label") || document.getElementById("a-audio-dropzone");
    const aAudioIn = document.getElementById("a-source-audio");
    if (aAudioDz && aAudioIn) {
      aAudioDz.addEventListener("click", () => aAudioIn.click());
      aAudioIn.addEventListener("change", async (e) => {
        const file = e.target.files[0];
        if (file) {
          state.animForm.audio_b64 = await fileToBase64(file);
          state.animAudioName = file.name;
          render();
        }
      });
    }
    const aClearAudio = document.getElementById("a-clear-audio");
    if (aClearAudio) aClearAudio.addEventListener("click", () => { state.animForm.audio_b64 = null; state.animAudioName = ""; render(); });
  }
}

/* ─── INIT ──────────────────────────────────────────── */

async function init() {
  render();
  await Promise.all([loadHealth(), loadModels(), loadJobs()]);
  setInterval(loadHealth, 15000);
  setInterval(loadJobs, 5000);
}

init();
