const { useEffect, useMemo, useState } = React;

const RESOLUTION_PRESETS = [
  { label: "Square", width: 1024, height: 1024 },
  { label: "Portrait", width: 896, height: 1152 },
  { label: "Landscape", width: 1152, height: 896 },
  { label: "Wide", width: 1344, height: 768 },
];

function formatSeconds(ms) {
  if (!ms) return "-";
  return `${(ms / 1000).toFixed(2)}s`;
}

function App() {
  const [models, setModels] = useState([]);
  const [health, setHealth] = useState(null);
  const [loras, setLoras] = useState([]);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [lightbox, setLightbox] = useState(null);

  const [form, setForm] = useState({
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
  });

  const activeModel = useMemo(
    () => models.find((model) => model.name === form.model_name) || null,
    [models, form.model_name]
  );

  const imageSrc = useMemo(() => {
    if (!result?.image_base64) return "";
    return `data:image/png;base64,${result.image_base64}`;
  }, [result]);

  useEffect(() => {
    loadBootstrap();
    const timer = setInterval(loadHealth, 15000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!form.model_name) return;
    loadLoras(form.model_name);
  }, [form.model_name]);

  async function loadBootstrap() {
    await Promise.all([loadHealth(), loadModels()]);
  }

  async function loadHealth() {
    try {
      const r = await fetch("/health");
      const data = await r.json();
      setHealth(data);
    } catch {
      setHealth(null);
    }
  }

  async function loadModels() {
    try {
      const r = await fetch("/models");
      const data = await r.json();
      const nextModels = data.models || [];
      setModels(nextModels);
      const currentModel = data.current_model || nextModels[0]?.name || "flux-1-dev";
      setForm((prev) => {
        const info = nextModels.find((item) => item.name === currentModel);
        if (!info) return { ...prev, model_name: currentModel };
        return {
          ...prev,
          model_name: currentModel,
          num_inference_steps: info.default_steps ?? prev.num_inference_steps,
          guidance_scale: info.default_guidance_scale ?? prev.guidance_scale,
        };
      });
    } catch {
      setModels([]);
    }
  }

  async function loadLoras(modelName) {
    try {
      const r = await fetch(`/loras?model_name=${encodeURIComponent(modelName)}`);
      const data = await r.json();
      const nextLoras = ["None", ...(data.loras || [])];
      setLoras(nextLoras);
      setForm((prev) => ({
        ...prev,
        lora_name: nextLoras.includes(prev.lora_name) ? prev.lora_name : "None",
        lora_scale: data.recommended_scale ?? prev.lora_scale,
      }));
    } catch {
      setLoras(["None"]);
    }
  }

  function updateField(name, value) {
    setForm((prev) => ({ ...prev, [name]: value }));
  }

  function applyModel(name) {
    const info = models.find((model) => model.name === name);
    setForm((prev) => ({
      ...prev,
      model_name: name,
      num_inference_steps: info?.default_steps ?? prev.num_inference_steps,
      guidance_scale: info?.default_guidance_scale ?? prev.guidance_scale,
      lora_name: "None",
    }));
  }

  function applyResolution(width, height) {
    setForm((prev) => ({ ...prev, width, height }));
  }

  async function onGenerate(e) {
    e.preventDefault();
    setError("");
    setLoading(true);

    const payload = {
      prompt: form.prompt,
      negative_prompt: form.negative_prompt || null,
      model_name: form.model_name,
      width: Number(form.width),
      height: Number(form.height),
      num_inference_steps: Number(form.num_inference_steps),
      guidance_scale: Number(form.guidance_scale),
      seed: form.seed === "" ? null : Number(form.seed),
      lora_name: !form.lora_name || form.lora_name === "None" ? null : form.lora_name,
      lora_scale: Number(form.lora_scale),
      use_refiner: false,
    };

    try {
      const r = await fetch("/generate-ui", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await r.json();

      if (!r.ok) {
        setError(data?.detail || "Generation failed.");
        return;
      }

      const nextResult = {
        ...data,
        model_name: form.model_name,
        prompt: form.prompt,
      };
      setResult(nextResult);
      setHistory((prev) => [nextResult, ...prev].slice(0, 8));
    } catch {
      setError("Request failed. Check server/network.");
    } finally {
      setLoading(false);
      loadHealth();
    }
  }

  return (
    <div className="app-shell">
      <div className="ambient ambient-left"></div>
      <div className="ambient ambient-right"></div>

      <div className="app">
        <section className="hero premium-card">
          <div className="hero-copy">
            <div className="eyebrow">AI Image Studio</div>
            <h1>Create high-quality images with simple controls.</h1>
            <p className="subtitle">
              Choose a model, write your prompt, and generate images. Only the information needed for normal use is shown here.
            </div>
          </div>
          <div className="hero-status premium-card inner-card">
            <div className="status-top">
              <span className={`dot ${health?.status === "healthy" ? "ok" : "idle"}`}></span>
              <span>{health?.status === "healthy" ? "Ready" : (health?.status || "Starting")}</span>
            </div>
            <div className="status-grid">
              <div>
                <label>Current model</label>
                <strong>{health?.current_model || "-"}</strong>
              </div>
              <div>
                <label>Models</label>
                <strong>{models.length || "-"}</strong>
              </div>
              <div>
                <label>Status</label>
                <strong>Online</strong>
              </div>
            </div>
          </div>
        </section>

        <section className="model-strip">
          {models.map((model) => (
            <button
              key={model.name}
              type="button"
              className={`model-card premium-card ${form.model_name === model.name ? "active" : ""}`}
              onClick={() => applyModel(model.name)}
            >
              <div className="model-name">{model.name}</div>
              <div className="model-summary">{model.summary || model.description || "Model ready"}</div>
            </button>
          ))}
        </section>

        <div className="layout">
          <form className="premium-card control-panel" onSubmit={onGenerate}>
            <div className="section-heading">
              <div>
                <div className="eyebrow">Controls</div>
                <h2>Compose</h2>
              </div>
            </div>

            <div className="field">
              <label>Prompt</label>
              <textarea
                value={form.prompt}
                onChange={(e) => updateField("prompt", e.target.value)}
                placeholder="Editorial portrait with sculpted light, premium materials, cinematic framing..."
                required
              />
            </div>

            <div className="field">
              <label>Negative Prompt</label>
              <textarea
                className="compact"
                value={form.negative_prompt}
                onChange={(e) => updateField("negative_prompt", e.target.value)}
                placeholder="blurry, low detail, watermark"
              />
            </div>

            <div className="field-group three-col">
              <div className="field">
                <label>Model</label>
                <select value={form.model_name} onChange={(e) => applyModel(e.target.value)}>
                  {models.map((m) => (
                    <option key={m.name} value={m.name}>{m.name}</option>
                  ))}
                </select>
              </div>
              <div className="field">
                <label>LoRA</label>
                <select value={form.lora_name} onChange={(e) => updateField("lora_name", e.target.value)}>
                  {loras.map((item) => (
                    <option key={item} value={item}>{item}</option>
                  ))}
                </select>
              </div>
              <div className="field">
                <label>LoRA Scale</label>
                <input type="number" min="0" max="2" step="0.05" value={form.lora_scale} onChange={(e) => updateField("lora_scale", e.target.value)} />
              </div>
            </div>

            <div className="field">
              <label>Resolution Presets</label>
              <div className="preset-row">
                {RESOLUTION_PRESETS.map((preset) => {
                  const active = form.width === preset.width && form.height === preset.height;
                  return (
                    <button
                      type="button"
                      key={preset.label}
                      className={`preset-chip ${active ? "active" : ""}`}
                      onClick={() => applyResolution(preset.width, preset.height)}
                    >
                      {preset.label}
                      <small>{preset.width}×{preset.height}</small>
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="field-group four-col">
              <div className="field">
                <label>Width</label>
                <input type="number" min="256" max="2048" step="8" value={form.width} onChange={(e) => updateField("width", e.target.value)} />
              </div>
              <div className="field">
                <label>Height</label>
                <input type="number" min="256" max="2048" step="8" value={form.height} onChange={(e) => updateField("height", e.target.value)} />
              </div>
              <div className="field">
                <label>Steps</label>
                <input
                  type="number"
                  min={activeModel?.min_steps || 1}
                  max={activeModel?.max_steps || 50}
                  value={form.num_inference_steps}
                  onChange={(e) => updateField("num_inference_steps", e.target.value)}
                />
              </div>
              <div className="field">
                <label>Guidance</label>
                <input type="number" min="0" max="20" step="0.5" value={form.guidance_scale} onChange={(e) => updateField("guidance_scale", e.target.value)} />
              </div>
            </div>

            <div className="field-group two-col align-end">
              <div className="field">
                <label>Seed</label>
                <input value={form.seed} onChange={(e) => updateField("seed", e.target.value)} placeholder="blank = random" />
              </div>
              <button className="primary-button" type="submit" disabled={loading}>
                {loading ? "Generating..." : "Generate Image"}
              </button>
            </div>

            {activeModel && (
              <div className="model-note">
                <strong>{activeModel.name}</strong>
                <span>{activeModel.description || activeModel.summary || "Model selected."}</span>
              </div>
            )}

            {error && <div className="error-banner">{error}</div>}
          </form>

          <section className="output-column">
            <div className="premium-card output-panel">
              <div className="section-heading compact-heading">
                <div>
                  <div className="eyebrow">Output</div>
                  <h2>Preview</h2>
                </div>
                <div className="metric-inline">
                  <span>{result?.model_name || health?.current_model || "No model"}</span>
                </div>
              </div>

              <div className="canvas" onClick={() => imageSrc && setLightbox(imageSrc)}>
                {imageSrc ? <img src={imageSrc} alt="Generated" /> : <span className="empty-state">Your generated image will appear here.</span>}
              </div>

              <div className="result-grid">
                <div className="result-card">
                  <label>Seed</label>
                  <strong>{result?.seed_used ?? "-"}</strong>
                </div>
                <div className="result-card">
                  <label>Time</label>
                  <strong>{formatSeconds(result?.inference_time_ms)}</strong>
                </div>
                <div className="result-card wide">
                  <label>Job</label>
                  <strong>{result?.job_id || "-"}</strong>
                </div>
              </div>
            </div>

            <div className="premium-card history-panel">
              <div className="section-heading compact-heading">
                <div>
                  <div className="eyebrow">History</div>
                  <h2>Recent generations</h2>
                </div>
              </div>
              <div className="history-grid">
                {history.length === 0 && <div className="history-empty">No generations yet.</div>}
                {history.map((item) => {
                  const src = `data:image/png;base64,${item.image_base64}`;
                  return (
                    <button key={item.job_id} type="button" className="history-item" onClick={() => setLightbox(src)}>
                      <img src={src} alt={item.prompt || item.job_id} />
                      <div className="history-meta">
                        <strong>{item.model_name}</strong>
                        <span>{formatSeconds(item.inference_time_ms)}</span>
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>
          </section>
        </div>
      </div>

      {lightbox && (
        <div className="lightbox" onClick={() => setLightbox(null)}>
          <img src={lightbox} alt="Preview" />
        </div>
      )}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
