const RESOLUTION_PRESETS = [
  { label: "Square", width: 1024, height: 1024 },
  { label: "Portrait", width: 896, height: 1152 },
  { label: "Landscape", width: 1152, height: 896 },
  { label: "Wide", width: 1344, height: 768 },
];

const state = {
  models: [],
  health: null,
  loras: ["None"],
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
};

const root = document.getElementById("root");

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function formatSeconds(ms) {
  if (!ms) return "-";
  return `${(ms / 1000).toFixed(2)}s`;
}

function activeModel() {
  return state.models.find((model) => model.name === state.form.model_name) || null;
}

function imageSrc(result) {
  if (!result || !result.image_base64) return "";
  return `data:image/png;base64,${result.image_base64}`;
}

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
    state.models = data.models || [];
    const currentModel = data.current_model || state.models[0]?.name || "flux-1-dev";
    const info = state.models.find((item) => item.name === currentModel);
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

function applyModel(name) {
  const info = state.models.find((model) => model.name === name);
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

    const nextResult = {
      ...data,
      model_name: state.form.model_name,
      prompt: state.form.prompt,
    };
    state.result = nextResult;
    state.history = [nextResult, ...state.history];
  } catch {
    state.error = "Request failed. Check server/network.";
  }

  state.loading = false;
  render();
  loadHealth();
}

function render() {
  const currentModel = activeModel();
  const currentImage = imageSrc(state.result);

  root.innerHTML = `
    <div class="app-shell">
      <div class="ambient ambient-left"></div>
      <div class="ambient ambient-right"></div>
      <div class="app">
        <section class="hero premium-card">
          <div class="hero-copy">
            <div class="eyebrow">AI Image Studio</div>
            <h1>Create high-quality images with simple controls.</h1>
            <p class="subtitle">Choose a model, write your prompt, and generate images. Only the information needed for normal use is shown here.</p>
          </div>
          <div class="hero-status premium-card inner-card">
            <div class="status-top">
              <span class="dot ${state.health?.status === "healthy" ? "ok" : ""}"></span>
              <span>${escapeHtml(state.health?.status === "healthy" ? "Ready" : (state.health?.status || "Starting"))}</span>
            </div>
            <div class="status-grid">
              <div><label>Current model</label><strong>${escapeHtml(state.health?.current_model || "-")}</strong></div>
              <div><label>Models</label><strong>${escapeHtml(state.models.length || "-")}</strong></div>
              <div><label>Status</label><strong>Online</strong></div>
            </div>
          </div>
        </section>

        <section class="model-strip">
          ${state.models.map((model) => `
            <button type="button" class="model-card premium-card ${state.form.model_name === model.name ? "active" : ""}" data-model="${escapeHtml(model.name)}">
              <div class="model-name">${escapeHtml(model.name)}</div>
              <div class="model-summary">${escapeHtml(model.summary || model.description || "Model ready")}</div>
            </button>
          `).join("")}
        </section>

        <div class="layout">
          <form class="premium-card control-panel" id="generate-form">
            <div class="section-heading"><div><div class="eyebrow">Controls</div><h2>Compose</h2></div></div>
            <div class="field"><label for="prompt">Prompt</label><textarea id="prompt" name="prompt" placeholder="Editorial portrait with sculpted light, premium materials, cinematic framing..." required>${escapeHtml(state.form.prompt)}</textarea></div>
            <div class="field"><label for="negative_prompt">Negative Prompt</label><textarea id="negative_prompt" class="compact" name="negative_prompt" placeholder="blurry, low detail, watermark">${escapeHtml(state.form.negative_prompt)}</textarea></div>
            <div class="field-group three-col">
              <div class="field"><label for="model_name">Model</label><select id="model_name" name="model_name">${state.models.map((m) => `<option value="${escapeHtml(m.name)}" ${state.form.model_name === m.name ? "selected" : ""}>${escapeHtml(m.name)}</option>`).join("")}</select></div>
              <div class="field"><label for="lora_name">LoRA</label><div class="lora-row"><select id="lora_name" name="lora_name">${state.loras.map((item) => `<option value="${escapeHtml(item)}" ${state.form.lora_name === item ? "selected" : ""}>${escapeHtml(item)}</option>`).join("")}</select><label class="upload-lora-btn ${state.loraUploadStatus === 'uploading' ? 'uploading' : ''}" title="Upload .safetensors LoRA"><input type="file" id="lora-file-input" accept=".safetensors" style="display:none" />${state.loraUploadStatus === 'uploading' ? '...' : '&#x2B06;'}</label></div>${state.loraUploadStatus && state.loraUploadStatus !== 'uploading' ? `<div class="lora-upload-msg ${state.loraUploadStatus.startsWith('ok') ? 'ok' : 'err'}">${escapeHtml(state.loraUploadStatus.startsWith('ok:') ? '✓ Uploaded: ' + state.loraUploadStatus.slice(3) : state.loraUploadStatus.slice(6))}</div>` : ''}</div>
              <div class="field"><label for="lora_scale">LoRA Scale</label><input id="lora_scale" name="lora_scale" type="number" min="0" max="2" step="0.05" value="${escapeHtml(state.form.lora_scale)}" /></div>
            </div>
            <div class="field"><label>Resolution Presets</label><div class="preset-row">${RESOLUTION_PRESETS.map((preset) => {
              const active = state.form.width === preset.width && state.form.height === preset.height;
              return `<button type="button" class="preset-chip ${active ? "active" : ""}" data-width="${preset.width}" data-height="${preset.height}">${escapeHtml(preset.label)}<small>${preset.width}×${preset.height}</small></button>`;
            }).join("")}</div></div>
            <div class="field-group four-col">
              <div class="field"><label for="width">Width</label><input id="width" name="width" type="number" min="256" max="2048" step="8" value="${escapeHtml(state.form.width)}" /></div>
              <div class="field"><label for="height">Height</label><input id="height" name="height" type="number" min="256" max="2048" step="8" value="${escapeHtml(state.form.height)}" /></div>
              <div class="field"><label for="num_inference_steps">Steps</label><input id="num_inference_steps" name="num_inference_steps" type="number" min="${escapeHtml(currentModel?.min_steps || 1)}" max="${escapeHtml(currentModel?.max_steps || 50)}" value="${escapeHtml(state.form.num_inference_steps)}" /></div>
              <div class="field"><label for="guidance_scale">Guidance</label><input id="guidance_scale" name="guidance_scale" type="number" min="0" max="20" step="0.5" value="${escapeHtml(state.form.guidance_scale)}" /></div>
            </div>
            <div class="field-group two-col align-end">
              <div class="field"><label for="seed">Seed</label><input id="seed" name="seed" value="${escapeHtml(state.form.seed)}" placeholder="blank = random" /></div>
              <button class="primary-button" type="submit" ${state.loading ? "disabled" : ""}>${state.loading ? "Generating..." : "Generate Image"}</button>
            </div>
            ${currentModel ? `<div class="model-note"><strong>${escapeHtml(currentModel.name)}</strong><span>${escapeHtml(currentModel.description || currentModel.summary || "Model selected.")}</span></div>` : ""}
            ${state.error ? `<div class="error-banner">${escapeHtml(state.error)}</div>` : ""}
          </form>

          <section class="output-column">
            <div class="premium-card output-panel">
              <div class="section-heading compact-heading"><div><div class="eyebrow">Output</div><h2>Preview</h2></div><div class="output-actions">${currentImage && !state.loading ? `<button type="button" class="download-btn" id="download-btn">&#8203;&#x2B07; Download</button>` : ""}<div class="metric-inline"><span>${escapeHtml(state.result?.model_name || state.health?.current_model || "No model")}</span></div></div></div>
              <div class="canvas" id="canvas">${state.loading ? `<div class="loading-spinner"><div class="spinner"></div><span>Generating image...</span></div>` : currentImage ? `<img src="${currentImage}" alt="Generated" />` : `<span class="empty-state">Your generated image will appear here.</span>`}</div>
              <div class="result-grid">
                <div class="result-card"><label>Seed</label><strong>${escapeHtml(state.result?.seed_used ?? "-")}</strong></div>
                <div class="result-card"><label>Time</label><strong>${escapeHtml(formatSeconds(state.result?.inference_time_ms))}</strong></div>
                <div class="result-card wide"><label>Job</label><strong>${escapeHtml(state.result?.job_id || "-")}</strong></div>
              </div>
            </div>
            <div class="premium-card history-panel">
              <div class="section-heading compact-heading"><div><div class="eyebrow">History</div><h2>All generations (${state.history.length})</h2></div></div>
              <div class="history-grid">${state.history.length === 0 ? `<div class="history-empty">No generations yet.</div>` : state.history.map((item, index) => `<button type="button" class="history-item" data-history-index="${index}"><img src="data:image/png;base64,${item.image_base64}" alt="${escapeHtml(item.prompt || item.job_id)}" /><div class="history-meta"><strong>${escapeHtml(item.model_name)}</strong><span>${escapeHtml(formatSeconds(item.inference_time_ms))}</span></div></button>`).join("")}</div>
            </div>
          </section>
        </div>
      </div>
      ${state.lightbox ? `<div class="lightbox" id="lightbox"><img src="${state.lightbox}" alt="Preview" /></div>` : ""}
    </div>
  `;

  bindEvents();
}

function bindEvents() {
  const form = document.getElementById("generate-form");
  if (form) form.addEventListener("submit", onGenerate);

  ["prompt", "negative_prompt", "width", "height", "num_inference_steps", "guidance_scale", "seed", "lora_scale"].forEach((name) => {
    const input = document.getElementById(name);
    if (!input) return;
    input.addEventListener("input", (event) => {
      state.form[name] = event.target.value;
    });
  });

  const modelSelect = document.getElementById("model_name");
  if (modelSelect) modelSelect.addEventListener("change", (event) => applyModel(event.target.value));

  const loraSelect = document.getElementById("lora_name");
  if (loraSelect) loraSelect.addEventListener("change", (event) => { state.form.lora_name = event.target.value; });

  root.querySelectorAll("[data-model]").forEach((button) => {
    button.addEventListener("click", () => applyModel(button.dataset.model));
  });

  const dlBtn = document.getElementById("download-btn");
  if (dlBtn) dlBtn.addEventListener("click", downloadImage);

  const loraFileInput = document.getElementById("lora-file-input");
  if (loraFileInput) {
    loraFileInput.addEventListener("change", (event) => {
      const file = event.target.files[0];
      if (file) uploadLora(file);
    });
  }

  root.querySelectorAll("[data-width]").forEach((button) => {
    button.addEventListener("click", () => applyResolution(Number(button.dataset.width), Number(button.dataset.height)));
  });

  const canvas = document.getElementById("canvas");
  if (canvas && state.result) {
    canvas.addEventListener("click", () => {
      state.lightbox = imageSrc(state.result);
      render();
    });
  }

  root.querySelectorAll("[data-history-index]").forEach((button) => {
    button.addEventListener("click", () => {
      const item = state.history[Number(button.dataset.historyIndex)];
      if (!item) return;
      state.lightbox = `data:image/png;base64,${item.image_base64}`;
      render();
    });
  });

  const lightbox = document.getElementById("lightbox");
  if (lightbox) {
    lightbox.addEventListener("click", () => {
      state.lightbox = null;
      render();
    });
  }
}

async function init() {
  render();
  await Promise.all([loadHealth(), loadModels()]);
  setInterval(loadHealth, 15000);
}

init();
