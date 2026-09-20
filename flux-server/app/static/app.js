/* ══════════════════════════════════════════════════════════════════════════
   Hyperforge AI Creative Studio — Image Generation Frontend
   ══════════════════════════════════════════════════════════════════════════ */

const RESOLUTION_PRESETS = [
  { label: "Square",    width: 1024, height: 1024 },
  { label: "Portrait",  width: 896,  height: 1152 },
  { label: "Landscape", width: 1152, height: 896  },
  { label: "Wide",      width: 1344, height: 768  },
];

const QUALITY_PRESETS = [
  { id: "draft",    label: "Draft",    steps: 15, guidance: 3.0, icon: "⚡" },
  { id: "balanced", label: "Balanced", steps: 25, guidance: 3.5, icon: "⚖" },
  { id: "hq",       label: "HQ",       steps: 35, guidance: 5.0, icon: "✦" },
  { id: "ultra",    label: "Ultra",    steps: 50, guidance: 7.0, icon: "◆" },
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
  {
    title: "Cyber City",
    prompt: "futuristic cyberpunk megacity, rain-slicked streets, neon billboards, cinematic wide shot, blade runner aesthetic",
    src: "https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?auto=format&fit=crop&w=640&q=80",
  },
  {
    title: "Forest Mist",
    prompt: "ancient forest at dawn, shafts of golden light, morning mist, moody atmospheric, photo real",
    src: "https://images.unsplash.com/photo-1448375240586-882707db888b?auto=format&fit=crop&w=640&q=80",
  },
  {
    title: "Galaxy Scape",
    prompt: "milky way over volcanic landscape, long exposure night sky, star trails, vivid colours, epic scale",
    src: "https://images.unsplash.com/photo-1462331940025-496dfbfc7564?auto=format&fit=crop&w=640&q=80",
  },
  {
    title: "Ocean Glass",
    prompt: "crystal clear tropical ocean, underwater perspective, sunlight caustics, coral reef, vibrant teal",
    src: "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?auto=format&fit=crop&w=640&q=80",
  },
  {
    title: "Marble Studio",
    prompt: "luxury marble texture studio, elegant architecture, soft diffused light, minimalist luxury",
    src: "https://images.unsplash.com/photo-1497366216548-37526070297c?auto=format&fit=crop&w=640&q=80",
  },
  {
    title: "Mountain Peak",
    prompt: "dramatic mountain peak above clouds, golden sunrise, epic alpine panorama, ultra sharp detail",
    src: "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?auto=format&fit=crop&w=640&q=80",
  },
  {
    title: "Abstract Flow",
    prompt: "abstract fluid art, iridescent paint swirls, macro photograph, vibrant metallic colours, high detail",
    src: "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?auto=format&fit=crop&w=640&q=80",
  },
  {
    title: "Retro Neon",
    prompt: "retro synthwave landscape, purple neon grid, palm trees silhouette, cinematic 80s aesthetic",
    src: "https://images.unsplash.com/photo-1518655048521-f130df041f66?auto=format&fit=crop&w=640&q=80",
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

const TAB_PATHS = {
  image:   "/image",
  library: "/library",
};

const state = {
  activeTab: "image",
  models: [],
  health: null,
  auth: {
    apiKey: localStorage.getItem("ncs_api_key") || "",
    required: false,
    configuredKeyCount: 0,
  },
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
    style_id: "cinematic",
  },
  loras: ["None"],
  library: [],
  libraryTotal: 0,
  libraryView: "grid",
  libraryLightbox: null,
};

const root = document.getElementById("root");

/* ─── Utilities ─────────────────────────────────────────────────────────── */

function escapeHtml(v) {
  return String(v ?? "")
    .replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")
    .replace(/"/g,"&quot;").replace(/'/g,"&#39;");
}
function formatMs(ms)  { return ms ? `${(ms/1000).toFixed(2)}s` : "-"; }
function activeModel() { return state.models.find(m=>m.name===state.form.model_name)||null; }
function imageSrc(r)   { return r?.image_base64 ? `data:image/png;base64,${r.image_base64}` : ""; }

function selectedImageStyle() { return IMAGE_STYLES.find(s=>s.id===state.form.style_id)||IMAGE_STYLES[0]; }
function styledPrompt(prompt, style) {
  const base = (prompt || "").trim();
  if (!style?.suffix || !base) return base;
  return `${base}, ${style.suffix}`;
}
function tabFromPath(path = window.location.pathname) {
  if (path.startsWith("/library")) return "library";
  return "image";
}

function fileToBase64(file) {
  return new Promise((res,rej)=>{
    const r=new FileReader();
    r.onload=()=>res(r.result.split(",")[1]);
    r.onerror=rej;
    r.readAsDataURL(file);
  });
}

/* ─── API ───────────────────────────────────────────────────────────────── */

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
    state.models = (data.models||[]);
    const cur = data.current_model || state.models[0]?.name || "flux-1-dev";
    const info = state.models.find(m=>m.name===cur);
    state.form.model_name = cur;
    if (info) {
      state.form.num_inference_steps = info.default_steps ?? state.form.num_inference_steps;
      state.form.guidance_scale = info.default_guidance_scale ?? state.form.guidance_scale;
    }
    await loadLoras(cur);
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

async function loadLibrary() {
  try {
    const { data } = await fetchJson(`/api/library?limit=200`);
    state.library = data.items || [];
    state.libraryTotal = data.total || 0;
  } catch { state.library = []; }
  render();
}

async function deleteLibraryItem(id) {
  try {
    await fetchJson(`/api/library/${encodeURIComponent(id)}`, { method: 'DELETE' });
    await loadLibrary();
  } catch {}
}

function applyQualityPreset(id) {
  const p = QUALITY_PRESETS.find(q => q.id === id);
  if (!p) return;
  state.form.num_inference_steps = p.steps;
  state.form.guidance_scale = p.guidance;
  render();
}

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

function setTab(t, opts = {}) {
  const next = TAB_PATHS[t] ? t : "image";
  state.activeTab = next;
  state.error = "";
  if (next === "library") loadLibrary();
  const nextPath = TAB_PATHS[next];
  if (opts.push !== false && window.location.pathname !== nextPath) {
    history.pushState({tab: next}, "", nextPath);
  }
  render();
  requestAnimationFrame(()=>{
    const page = document.querySelector(".page-content");
    if (page) { page.classList.remove("page-in"); void page.offsetWidth; page.classList.add("page-in"); }
  });
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

/* ══════════════════════════════════════════════════════════════════════════
   RENDER
   ══════════════════════════════════════════════════════════════════════════ */

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
            <h1>Forge luminous images.</h1>
            <p class="subtitle">A brighter workspace for image generation, LoRA experiments, and polished creative previews.</p>
            <div class="hero-actions">
              <button type="button" class="hero-action primary" data-sample-prompt="${escapeHtml(SAMPLE_IMAGES[0].prompt)}">Try a Sample</button>
              <a class="hero-action" href="/library" data-tab="library">View Library</a>
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
                <div><label>Styles</label><strong>${IMAGE_STYLES.length} presets</strong></div>
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
          <a class="tab-btn ${tab==="library"?"active":""}" href="/library" data-tab="library">Library
            <span class="job-count">${state.libraryTotal||""}</span>
          </a>
        </nav>

        <main class="page-content" data-page="${escapeHtml(tab)}">
          ${tab==="image"     ? renderImageTab(cur,curImg) : ""}
          ${tab==="library"   ? renderLibraryTab()         : ""}
        </main>

        ${state.error ? `<div class="global-error">${escapeHtml(state.error)}</div>` : ""}
      </div>
      ${state.lightbox ? `<div class="lightbox" id="lightbox"><img src="${state.lightbox}" alt="Preview" /></div>` : ""}
      ${state.libraryLightbox ? `<div class="lib-lightbox" id="lib-lightbox"><button class="lib-lb-close">&#10005;</button><img src="${state.libraryLightbox.url}" alt="Preview" /></div>` : ""}
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

/* ─── IMAGE TAB ─────────────────────────────────────────────────────────── */

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
        <div class="field"><label>Quality Preset</label>
          <div class="preset-row quality-row">
            ${QUALITY_PRESETS.map(p=>{
              const steps = Number(state.form.num_inference_steps);
              const active = steps===p.steps && Number(state.form.guidance_scale)===p.guidance;
              return `<button type="button" class="preset-chip quality-chip ${active?"active":""}" data-quality="${p.id}">
                <span class="quality-icon">${p.icon}</span><span>${p.label}</span><small>${p.steps}st · cfg${p.guidance}</small>
              </button>`;
            }).join("")}
          </div>
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

/* ─── LIBRARY TAB ───────────────────────────────────────────────────────── */

function renderLibraryTab() {
  const items = state.library;
  const view = state.libraryView;

  const viewBtns = `
    <button type="button" class="lib-view-btn ${view==='grid'?'active':''}" data-lib-view="grid" title="Grid view">&#9632;&#9632;</button>
    <button type="button" class="lib-view-btn ${view==='list'?'active':''}" data-lib-view="list" title="List view">&#9776;</button>
  `;

  const empty = `<div class="lib-empty"><span>✦</span><p>Nothing saved yet.</p><p class="lib-empty-sub">Generate an image and it will appear here automatically.</p></div>`;

  const gridItems = items.map(item => {
    const thumb = `<img src="${item.url}" alt="${escapeHtml(item.prompt||'')}" loading="lazy" />`;
    const date = new Date(item.created_at * 1000).toLocaleDateString(undefined, {month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'});
    return `
      <div class="lib-item" data-lib-lightbox="${escapeHtml(item.url)}">
        <div class="lib-thumb">${thumb}
          <div class="lib-item-overlay">
            <a class="lib-btn" href="${item.url}" download title="Download">&#8681;</a>
            <button type="button" class="lib-btn lib-delete-btn" data-lib-delete="${escapeHtml(item.id)}" title="Delete">&#10005;</button>
          </div>
        </div>
        <div class="lib-meta">
          <div class="lib-prompt">${escapeHtml((item.prompt||'').slice(0,80))}${(item.prompt||'').length>80?'&hellip;':''}</div>
          <div class="lib-detail">
            <span>${escapeHtml(item.model_name||'')}</span>
            <span>${date}</span>
            ${item.width ? `<span>${item.width}&times;${item.height}</span>` : ''}
          </div>
        </div>
      </div>`;
  }).join('');

  const listItems = items.map(item => {
    const date = new Date(item.created_at * 1000).toLocaleDateString(undefined, {month:'short',day:'numeric',year:'numeric'});
    return `
      <div class="lib-list-item">
        <div class="lib-list-prompt">${escapeHtml((item.prompt||'').slice(0,120))}${(item.prompt||'').length>120?'&hellip;':''}</div>
        <span class="lib-list-model">${escapeHtml(item.model_name||'')}</span>
        <span class="lib-list-date">${date}</span>
        <div class="lib-list-actions">
          <a class="lib-btn" href="${item.url}" download title="Download">&#8681;</a>
          <button type="button" class="lib-btn lib-delete-btn" data-lib-delete="${escapeHtml(item.id)}" title="Delete">&#10005;</button>
        </div>
      </div>`;
  }).join('');

  return `
    <div class="lib-shell">
      <div class="lib-toolbar">
        <div class="lib-toolbar-left">
          <h2>Library Inventory</h2>
        </div>
        <div class="lib-toolbar-right">
          <span class="lib-count">${state.libraryTotal} item${state.libraryTotal!==1?'s':''}</span>
          <div class="lib-views">${viewBtns}</div>
          <button type="button" class="lib-refresh-btn" id="lib-refresh-btn" title="Refresh">&#8635;</button>
        </div>
      </div>
      ${items.length===0 ? empty : view==='grid'
        ? `<div class="lib-grid">${gridItems}</div>`
        : `<div class="lib-list">${listItems}</div>`
      }
    </div>`;
}

/* ─── Event Binding ─────────────────────────────────────────────────────── */

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

  // Model cards
  document.querySelectorAll("[data-model]").forEach(card=>{
    card.addEventListener("click", ()=>applyModel(card.dataset.model));
  });

  // Quality presets
  document.querySelectorAll("[data-quality]").forEach(btn=>{
    btn.addEventListener("click", ()=>applyQualityPreset(btn.dataset.quality));
  });

  // Library controls
  document.querySelectorAll("[data-lib-view]").forEach(btn=>{
    btn.addEventListener("click", ()=>{ state.libraryView=btn.dataset.libView; render(); });
  });
  document.querySelectorAll("[data-lib-delete]").forEach(btn=>{
    btn.addEventListener("click", e=>{ e.stopPropagation(); if(confirm('Delete this item?')) deleteLibraryItem(btn.dataset.libDelete); });
  });
  document.querySelectorAll("[data-lib-lightbox]").forEach(el=>{
    el.addEventListener("click", e=>{
      if(e.target.closest('.lib-btn')) return;
      const url=el.dataset.libLightbox;
      state.libraryLightbox={url}; render();
    });
  });
  const libRefresh = document.getElementById("lib-refresh-btn");
  if (libRefresh) libRefresh.addEventListener("click", ()=>loadLibrary());
  // Library lightbox close
  const libLb = document.getElementById("lib-lightbox");
  if (libLb) libLb.addEventListener("click", e=>{
    if(e.target===libLb||e.target.classList.contains('lib-lb-close')) { state.libraryLightbox=null; render(); }
  });

  // Resolution presets
  document.querySelectorAll("[data-width]").forEach(btn=>{
    btn.addEventListener("click", ()=>applyResolution(Number(btn.dataset.width),Number(btn.dataset.height)));
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
}

function bindInput(id, setter) {
  const el = document.getElementById(id);
  if (!el) return;
  el.addEventListener("input", ()=>setter(el.value));
  el.addEventListener("change", ()=>setter(el.value));
}

/* ─── Init ──────────────────────────────────────────────────────────────── */

async function init() {
  state.activeTab = tabFromPath();
  render();
  await Promise.all([loadAuthStatus(), loadHealth(), loadModels(), loadLibrary()]);
  render();
  // Periodic refresh
  setInterval(loadHealth, 15000);
}

window.addEventListener("popstate", ()=>{
  const next = tabFromPath();
  state.activeTab = next;
  state.error = "";
  if (next === "library") loadLibrary();
  render();
});

init();
