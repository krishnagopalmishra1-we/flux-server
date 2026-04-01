import gradio as gr
import base64
import io
import logging
import random
import shutil
from pathlib import Path
from PIL import Image
from app.pipeline_new import inference_pipeline as flux_pipeline
from app.config import get_settings

logger = logging.getLogger(__name__)

_gallery_images: list = []
MAX_GALLERY = 20

STYLE_PRESETS = {
    "None": "",
    "Cinematic": ", cinematic lighting, dramatic atmosphere, film grain, shallow depth of field, 8k, photorealistic",
    "Digital Art": ", digital art, highly detailed, vibrant colors, sharp focus, artstation trending, concept art",
    "Photorealistic": ", ultra realistic photograph, DSLR, 85mm lens, natural lighting, sharp focus, 8k resolution, detailed skin texture",
    "Anime": ", anime style, cel shading, vibrant colors, clean lines, studio ghibli inspired, high quality illustration",
    "Oil Painting": ", oil painting, brush strokes visible, classical art style, rich colors, gallery quality, masterpiece",
    "Watercolor": ", watercolor painting, soft edges, pastel colors, artistic, gentle washes, paper texture visible",
    "3D Render": ", 3D render, octane render, unreal engine 5, volumetric lighting, highly detailed, photorealistic materials",
    "Fantasy": ", fantasy art, magical atmosphere, ethereal lighting, enchanted, mystical, detailed environment, epic composition",
}

RESOLUTION_PRESETS = {
    "1024 × 1024 (Square)": (1024, 1024),
    "1152 × 896 (Landscape)": (1152, 896),
    "896 × 1152 (Portrait)": (896, 1152),
    "1344 × 768 (Wide)": (1344, 768),
    "768 × 1344 (Tall)": (768, 1344),
    "512 × 512 (Small Square)": (512, 512),
}

QUALITY_PROFILES = {
    "Balanced": {"step_factor": 1.0, "guidance_shift": 0.0, "resolution": "1024 × 1024 (Square)"},
    "Speed": {"step_factor": 0.7, "guidance_shift": -0.5, "resolution": "896 × 1152 (Portrait)"},
    "Detail": {"step_factor": 1.25, "guidance_shift": 0.8, "resolution": "1152 × 896 (Landscape)"},
    "Cinematic": {"step_factor": 1.1, "guidance_shift": 0.4, "resolution": "1344 × 768 (Wide)"},
}

PROMPT_LAB_SUBJECTS = [
    "none", "futuristic portrait", "architectural masterpiece",
    "mythic character", "cinematic cityscape", "fashion editorial",
]
PROMPT_LAB_ENVIRONMENTS = [
    "none", "in a rain-soaked neon alley", "inside a brutalist gallery",
    "on a floating island above clouds", "in a desert with geometric monoliths",
    "inside a hyper-minimal studio",
]
PROMPT_LAB_MOODS = [
    "none", "calm, contemplative", "epic, high tension",
    "dreamlike, surreal", "luxury, editorial polish", "raw, documentary realism",
]
PROMPT_LAB_FRAMING = [
    "none", "wide-angle cinematic composition", "35mm film look, shallow depth",
    "symmetrical composition, clean negative space", "low-angle heroic shot", "macro detail shot",
]


def _format_bytes(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024


def get_lora_server_status(current_model: str) -> str:
    try:
        lora_dir = Path("loras")
        lora_dir.mkdir(exist_ok=True)
        files = sorted(lora_dir.glob("*.safetensors"), key=lambda p: p.stat().st_mtime, reverse=True)
        compatible = flux_pipeline.get_compatible_loras(current_model)
        total_size = sum(p.stat().st_size for p in files)
        if not files:
            return (
                "No LoRA files on server yet.\n"
                "Note: Upload Status updates only after file transfer fully completes. "
                "For very large LoRAs (10GB+), prefer direct SCP upload to the VM."
            )
        lines = [
            f"Server LoRAs: {len(files)} total | {len(compatible)} compatible with {current_model} | Total size: {_format_bytes(total_size)}",
            "Latest files:",
        ]
        for p in files[:8]:
            lines.append(f"- {p.name} ({_format_bytes(p.stat().st_size)})")
        lines.append("Note: Browser upload status appears here only after upload completes.")
        return "\n".join(lines)
    except Exception as e:
        return f"Failed to read LoRA server status: {e}"


def get_gpu_status() -> str:
    try:
        info = flux_pipeline.gpu_info()
        current = flux_pipeline.model_manager.current_model
        loaded = flux_pipeline.is_loaded
        pct = (info["used_gb"] / info["total_gb"] * 100) if info["total_gb"] > 0 else 0
        bar_filled = int(pct / 5)
        bar_empty = 20 - bar_filled
        bar = "█" * bar_filled + "░" * bar_empty
        icon = "◉" if loaded else "◎"
        return (
            f"**{icon} {current}** &nbsp;·&nbsp; "
            f"GPU: {info['name']} &nbsp;·&nbsp; "
            f"VRAM [{bar}] {info['used_gb']:.1f}/{info['total_gb']:.1f} GB ({pct:.0f}%)"
        )
    except Exception:
        return "⚠ GPU status unavailable"


def improve_prompt(prompt: str, style: str) -> str:
    if not prompt.strip():
        return prompt
    suffix = STYLE_PRESETS.get(style, "")
    return prompt.strip() + suffix


def generate_image(
    prompt, negative_prompt, model_name, style, resolution,
    steps, guidance, seed, lora_name, lora_scale, auto_improve,
):
    if not prompt.strip():
        raise gr.Error("Please enter a prompt.")
    final_prompt = improve_prompt(prompt, style) if auto_improve else prompt
    width, height = RESOLUTION_PRESETS.get(resolution, (1024, 1024))
    use_seed = seed if seed > 0 else None
    model_family = flux_pipeline.get_model_family(model_name)
    effective_negative = (negative_prompt or "").strip()
    auto_negative_applied = False
    if not effective_negative and model_family != "flux":
        effective_negative = flux_pipeline.get_auto_negative_prompt(model_name, style)
        auto_negative_applied = True
    try:
        model_info = flux_pipeline.get_model_info(model_name)
        steps = max(model_info["min_steps"], min(steps, model_info["max_steps"]))
        img_b64, seed_used, elapsed_ms = flux_pipeline.generate(
            prompt=final_prompt,
            negative_prompt=effective_negative if effective_negative else None,
            model_name=model_name,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            seed=use_seed,
            lora_name=lora_name,
            lora_scale=lora_scale,
            use_refiner=False,
            style=style,
        )
    except Exception as e:
        logger.exception("Generation failed")
        raise gr.Error(f"Generation failed: {e}")
    img_bytes = base64.b64decode(img_b64)
    image = Image.open(io.BytesIO(img_bytes))
    info_text = (
        f"Model: {model_name}  ·  Seed: {seed_used}  ·  "
        f"Time: {elapsed_ms / 1000:.1f}s  ·  Size: {width}×{height}  ·  "
        f"Steps: {steps}  ·  Guidance: {guidance}"
    )
    if lora_name and lora_name != "None":
        info_text += f"  ·  LoRA: {lora_name} ({lora_scale})"
    if auto_negative_applied:
        info_text += "  ·  Negative: auto"
    _gallery_images.insert(0, image)
    while len(_gallery_images) > MAX_GALLERY:
        _gallery_images.pop()
    gpu_status = get_gpu_status()
    return image, final_prompt, info_text, seed_used, list(_gallery_images), gpu_status


def upload_lora_files(files, current_model: str):
    if not files:
        return gr.update(), "No files selected."
    lora_dir = Path("loras")
    lora_dir.mkdir(exist_ok=True)
    uploaded, skipped = [], []
    for file_path in files:
        fname = Path(file_path).name
        if not fname.endswith(".safetensors"):
            skipped.append(fname)
            continue
        dest = lora_dir / fname
        shutil.copy2(file_path, str(dest))
        uploaded.append(fname)
    new_loras = ["None"] + flux_pipeline.get_compatible_loras(current_model)
    last = uploaded[-1] if uploaded else "None"
    if last not in new_loras:
        last = "None"
    parts = []
    if uploaded:
        parts.append(f"✓ Uploaded {len(uploaded)}: {', '.join(uploaded)}")
    if skipped:
        parts.append(f"⚠ Skipped (not .safetensors): {', '.join(skipped)}")
    status = "  ·  ".join(parts) if parts else "No valid files."
    status = status + "\n" + get_lora_server_status(current_model)
    return gr.update(choices=new_loras, value=last), status


def apply_quality_profile(profile: str, model_name: str):
    info = flux_pipeline.get_model_info(model_name)
    profile_cfg = QUALITY_PROFILES.get(profile, QUALITY_PROFILES["Balanced"])
    base_steps = info.get("default_steps", 20)
    min_steps = info.get("min_steps", 1)
    max_steps = info.get("max_steps", 50)
    suggested_steps = int(round(base_steps * profile_cfg["step_factor"]))
    suggested_steps = max(min_steps, min(max_steps, suggested_steps))
    guidance = info.get("default_guidance_scale", 3.5) + profile_cfg["guidance_shift"]
    guidance = max(0.0, min(20.0, guidance))
    return (
        gr.update(value=suggested_steps),
        gr.update(value=guidance),
        gr.update(value=profile_cfg["resolution"]),
    )


def compose_prompt_lab(base_prompt, subject, environment, mood, framing):
    parts = []
    if base_prompt and base_prompt.strip():
        parts.append(base_prompt.strip())
    for value in (subject, environment, mood, framing):
        if value and value != "none":
            parts.append(value)
    if not parts:
        return base_prompt
    return ", ".join(parts)


def randomize_creative_setup(model_name: str):
    info = flux_pipeline.get_model_info(model_name)
    style_name = random.choice([k for k in STYLE_PRESETS.keys() if k != "None"])
    resolution = random.choice(list(RESOLUTION_PRESETS.keys()))
    steps = random.randint(info["min_steps"], info["max_steps"])
    guidance = round(max(0.0, min(20.0, info.get("default_guidance_scale", 3.5) + random.uniform(-1.0, 1.0))), 1)
    return (
        gr.update(value=style_name),
        gr.update(value=resolution),
        gr.update(value=steps),
        gr.update(value=guidance),
        gr.update(value=-1),
        gr.update(value=True),
    )


# ═══════════════════════════════════════════════════════
#  FUTURISTIC CSS — CYBERPUNK NEURAL INTERFACE THEME
# ═══════════════════════════════════════════════════════
FUTURISTIC_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Manrope:wght@400;500;600;700&display=swap');

:root {
    --bg: #0c1018;
    --bg-soft: #121826;
    --card: #161e31;
    --card-2: #1a2338;
    --text: #e9eefc;
    --text-soft: #a7b2cf;
    --line: #2a3551;
    --brand: #4dd2ff;
    --brand-2: #7d8bff;
}

body {
    background:
        radial-gradient(1200px 500px at 15% -10%, rgba(77, 210, 255, 0.10), transparent 55%),
        radial-gradient(900px 420px at 90% 0%, rgba(125, 139, 255, 0.12), transparent 60%),
        var(--bg) !important;
}

.gradio-container {
    max-width: 1520px !important;
    padding: 0 !important;
    color: var(--text) !important;
    font-family: 'Manrope', sans-serif !important;
    background: transparent !important;
}

.hero-wrap {
    padding: 32px 36px 24px;
    border-bottom: 1px solid var(--line);
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.02), transparent);
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: clamp(28px, 3.4vw, 48px) !important;
    line-height: 1.1 !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    color: var(--text) !important;
    margin: 0 !important;
}

.hero-sub {
    margin-top: 8px !important;
    color: var(--text-soft) !important;
    font-size: 14px !important;
    letter-spacing: 0.02em !important;
}

.gpu-bar {
    margin: 0 !important;
    padding: 10px 36px !important;
    border-bottom: 1px solid var(--line) !important;
    background: rgba(255, 255, 255, 0.02) !important;
}

.tab-nav {
    padding: 0 24px !important;
    border-bottom: 1px solid var(--line) !important;
}

.tab-nav button {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
}

.main-row {
    padding: 20px 24px;
    gap: 18px !important;
}

.panel-left, .panel-right {
    background: linear-gradient(180deg, var(--card), var(--card-2)) !important;
    border: 1px solid var(--line) !important;
    border-radius: 14px !important;
    padding: 20px !important;
}

.section-label {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--brand) !important;
    border-bottom: 1px solid var(--line) !important;
    padding-bottom: 10px !important;
    margin-bottom: 12px !important;
}

.generate-btn button, .generate-btn {
    min-height: 52px !important;
    border-radius: 12px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em !important;
}

.gr-image,
.gr-gallery,
.info-box textarea {
    border-radius: 12px !important;
}

footer, .footer { display: none !important; }

@media (max-width: 900px) {
    .hero-wrap { padding: 22px 16px 16px; }
    .main-row { padding: 12px !important; }
    .panel-left, .panel-right { padding: 14px !important; }
}
"""


FUTURISTIC_THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.sky,
    secondary_hue=gr.themes.colors.indigo,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Manrope"), "ui-sans-serif", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#010409",
    body_background_fill_dark="#010409",
    body_text_color="#c8e0ff",
    body_text_color_dark="#c8e0ff",
    background_fill_primary="#010409",
    background_fill_primary_dark="#010409",
    background_fill_secondary="#040d1a",
    background_fill_secondary_dark="#040d1a",
    block_background_fill="#081226",
    block_background_fill_dark="#081226",
    block_border_color="rgba(0, 229, 255, 0.12)",
    block_border_color_dark="rgba(0, 229, 255, 0.12)",
    block_label_background_fill="#081226",
    block_label_background_fill_dark="#081226",
    block_title_text_color="#c8e0ff",
    block_title_text_color_dark="#c8e0ff",
    input_background_fill="#040d1a",
    input_background_fill_dark="#040d1a",
    input_border_color="rgba(0, 229, 255, 0.12)",
    input_border_color_dark="rgba(0, 229, 255, 0.12)",
    input_border_color_focus="#00e5ff",
    input_border_color_focus_dark="#00e5ff",
    input_placeholder_color="#3d6a8a",
    input_placeholder_color_dark="#3d6a8a",
    button_primary_background_fill="#003d57",
    button_primary_background_fill_dark="#003d57",
    button_primary_background_fill_hover="#005f8a",
    button_primary_background_fill_hover_dark="#005f8a",
    button_primary_border_color="#00e5ff",
    button_primary_border_color_dark="#00e5ff",
    button_primary_text_color="#00e5ff",
    button_primary_text_color_dark="#00e5ff",
    button_secondary_background_fill="#040d1a",
    button_secondary_background_fill_dark="#040d1a",
    button_secondary_background_fill_hover="rgba(0, 229, 255, 0.08)",
    button_secondary_background_fill_hover_dark="rgba(0, 229, 255, 0.08)",
    button_secondary_border_color="rgba(0, 229, 255, 0.12)",
    button_secondary_border_color_dark="rgba(0, 229, 255, 0.12)",
    button_secondary_text_color="#7aabdd",
    button_secondary_text_color_dark="#7aabdd",
    checkbox_background_color="#040d1a",
    checkbox_background_color_dark="#040d1a",
    checkbox_background_color_selected="#00e5ff",
    checkbox_background_color_selected_dark="#00e5ff",
    checkbox_border_color="rgba(0, 229, 255, 0.16)",
    checkbox_border_color_dark="rgba(0, 229, 255, 0.16)",
    slider_color="#00e5ff",
    slider_color_dark="#00e5ff",
)


def _injected_theme_markup() -> str:
    return f"""
    <style>
    {FUTURISTIC_CSS}
    html, body {{
        background: #010409 !important;
        color: #c8e0ff !important;
    }}
    </style>
    <script>
    (() => {{
        const root = document.documentElement;
        const body = document.body;
        if (root) {{
            root.style.background = '#010409';
            root.style.colorScheme = 'dark';
        }}
        if (body) {{
            body.style.background = '#010409';
            body.style.color = '#c8e0ff';
        }}
    }})();
    </script>
    """


def build_ui() -> gr.Blocks:
    available_models = flux_pipeline.list_available_models()
    default_model_info = flux_pipeline.get_model_info("flux-1-dev")

    with gr.Blocks(
        title="FLUX Neural Studio",
        css=FUTURISTIC_CSS,
        theme=FUTURISTIC_THEME,
    ) as demo:

        # ── HERO HEADER ──
        gr.HTML("""
        <div class="hero-wrap">
            <div class="hero-title">Neural Image Studio</div>
            <div class="hero-sub">Fast, reliable multi-model generation with a cleaner studio-grade interface.</div>
        </div>
        """)

        # ── GPU STATUS ──
        gpu_status_bar = gr.Markdown(value=get_gpu_status(), elem_classes=["gpu-bar"])

        # ── TABS ──
        with gr.Tabs(elem_classes=["tab-nav"]):

            # ══ GENERATE TAB ══
            with gr.Tab("⬡  Generate"):
                with gr.Row(elem_classes=["main-row"]):

                    # ── LEFT: CONTROLS ──
                    with gr.Column(scale=2, elem_classes=["panel-left", "panel-corner"]):

                        gr.Markdown("<div class='section-label'>Model Configuration</div>")
                        model_dropdown = gr.Dropdown(
                            label="Active Model",
                            choices=list(available_models.keys()),
                            value="flux-1-dev",
                            info="Switching unloads previous model to free VRAM.",
                        )
                        model_hint = gr.Markdown(
                            "Select a model to see its profile.",
                            elem_classes=["info-text"],
                        )

                        gr.Markdown("<div class='section-label' style='margin-top:16px'>Prompt Engineering</div>")
                        prompt = gr.Textbox(
                            label="Primary Prompt",
                            placeholder="Describe your vision — a cyberpunk cityscape at dawn, neon reflections on wet asphalt...",
                            lines=3, max_lines=6,
                        )

                        with gr.Accordion("⬡  Prompt Composer Lab", open=False):
                            with gr.Row():
                                lab_subject = gr.Dropdown(
                                    label="Subject",
                                    choices=PROMPT_LAB_SUBJECTS, value="none",
                                )
                                lab_environment = gr.Dropdown(
                                    label="Environment",
                                    choices=PROMPT_LAB_ENVIRONMENTS, value="none",
                                )
                            with gr.Row():
                                lab_mood = gr.Dropdown(
                                    label="Mood",
                                    choices=PROMPT_LAB_MOODS, value="none",
                                )
                                lab_framing = gr.Dropdown(
                                    label="Framing",
                                    choices=PROMPT_LAB_FRAMING, value="none",
                                )
                            compose_prompt_btn = gr.Button("⬡  Compose Prompt")

                        with gr.Accordion("⬡  Negative Prompt", open=False):
                            negative_prompt = gr.Textbox(
                                label="Negative Prompt (disabled for FLUX models)",
                                placeholder="blurry, low quality, distorted, watermark, text...",
                                lines=2, max_lines=4,
                            )

                        gr.Markdown("<div class='section-label' style='margin-top:16px'>Style & Quality</div>")
                        with gr.Row():
                            auto_improve = gr.Checkbox(label="Auto Enhance Prompt", value=True)
                            style = gr.Dropdown(
                                label="Style Preset",
                                choices=list(STYLE_PRESETS.keys()),
                                value="None",
                            )
                        profile_dropdown = gr.Dropdown(
                            label="Quality Profile",
                            choices=list(QUALITY_PROFILES.keys()),
                            value="Balanced",
                            info="One-click tuning: Speed · Balanced · Detail · Cinematic",
                        )
                        improved_prompt = gr.Textbox(
                            label="Final Enhanced Prompt",
                            interactive=False, lines=2,
                        )

                        gr.Markdown("<div class='section-label' style='margin-top:16px'>Rendering Parameters</div>")
                        resolution = gr.Dropdown(
                            label="Resolution",
                            choices=list(RESOLUTION_PRESETS.keys()),
                            value="1024 × 1024 (Square)",
                        )
                        with gr.Row():
                            steps = gr.Slider(
                                label="Diffusion Steps",
                                minimum=default_model_info["min_steps"],
                                maximum=default_model_info["max_steps"],
                                value=default_model_info["default_steps"],
                                step=1,
                            )
                            guidance = gr.Slider(
                                label="Guidance Scale",
                                minimum=0.0, maximum=20.0,
                                value=default_model_info.get("default_guidance_scale", 3.5),
                                step=0.5,
                            )
                        with gr.Row():
                            seed = gr.Number(label="Seed  (−1 = random)", value=-1, precision=0)
                            randomize_btn = gr.Button("⬡  Randomize Setup")

                        with gr.Accordion("⬡  LoRA Adapter Module", open=True):
                            loras = ["None"] + flux_pipeline.get_compatible_loras("flux-1-dev")
                            lora_dropdown = gr.Dropdown(
                                label="LoRA Adapter",
                                choices=loras, value="None",
                                info="All compatible LoRAs shown for selected model.",
                            )
                            lora_scale = gr.Slider(
                                label="LoRA Influence Strength",
                                minimum=0.0, maximum=2.0,
                                value=flux_pipeline.get_recommended_lora_scale("flux-1-dev"),
                                step=0.05,
                            )
                            lora_upload = gr.File(
                                label="Upload LoRA Files  (.safetensors)",
                                file_types=[".safetensors"],
                                file_count="multiple",
                                type="filepath",
                            )
                            gr.Markdown(
                                "For large files (10 GB+), browser uploads may be slow. "
                                "Status updates after transfer completes."
                            )
                            lora_upload_status = gr.Textbox(
                                label="Server LoRA Status",
                                interactive=False, lines=8,
                                value=get_lora_server_status("flux-1-dev"),
                                elem_classes=["info-box"],
                            )
                            with gr.Row():
                                refresh_btn = gr.Button("⬡  Refresh LoRA List")
                                check_upload_btn = gr.Button("⬡  Check Upload Status")

                        # ── GENERATE CTA ──
                        gr.HTML("<div style='height:12px'></div>")
                        generate_btn = gr.Button(
                            "⬡  GENERATE IMAGE",
                            variant="primary",
                            elem_classes=["generate-btn"],
                        )

                    # ── RIGHT: OUTPUT ──
                    with gr.Column(scale=3, elem_classes=["panel-right", "panel-corner"]):
                        gr.HTML("""
                        <div class='section-label' style='display:flex; justify-content:space-between; align-items:center;'>
                            <span>Output Canvas</span>
                            <span style="font-family:'Share Tech Mono',monospace; font-size:9px; color:var(--t4); letter-spacing:0.1em;">CLICK IMAGE TO EXPAND</span>
                        </div>
                        """)
                        output_image = gr.Image(
                            label="Generated Image",
                            type="pil",
                            height=580,
                            elem_classes=["output-image"],
                        )
                        with gr.Row():
                            info_box = gr.Textbox(
                                label="Generation Metadata",
                                interactive=False,
                                elem_classes=["info-box"],
                            )
                            used_seed = gr.Number(
                                label="Seed Used",
                                interactive=False,
                                precision=0,
                            )

            # ══ GALLERY TAB ══
            with gr.Tab("⬡  Gallery"):
                gr.HTML("""
                <div style="padding:24px 0 16px;">
                    <div style="display:flex; align-items:center; gap:12px; margin-bottom:8px;">
                        <div style="
                            width:6px; height:6px; border-radius:50%;
                            background:#00e5ff;
                            box-shadow: 0 0 8px #00e5ff;
                            animation: dotBlink 2s ease-in-out infinite;
                        "></div>
                        <div style="
                            font-family:'Orbitron',monospace; font-size:11px;
                            font-weight:700; letter-spacing:0.2em; text-transform:uppercase;
                            color:#00e5ff;
                        ">Session Archive</div>
                        <div style="
                            flex:1; height:1px; margin-left:8px;
                            background: linear-gradient(90deg, rgba(0,229,255,0.3), transparent);
                        "></div>
                    </div>
                    <div style="
                        font-family:'Space Grotesk',sans-serif; font-size:13px;
                        color:#3d6a8a; letter-spacing:0.02em;
                    ">All images generated in this session — click to preview full size</div>
                </div>
                """)
                gallery = gr.Gallery(
                    label="",
                    columns=4, rows=3,
                    height=520,
                    object_fit="contain",
                    preview=True,
                    elem_classes=["gallery-panel"],
                )

            # ══ MODELS TAB ══
            with gr.Tab("⬡  Models & API"):
                gr.HTML("<div style='height:12px'></div>")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Available Models")
                        for name, desc in available_models.items():
                            info = flux_pipeline.get_model_info(name)
                            badge = "`◉ LOADED`" if info.get("loaded") else "`◎ STANDBY`"
                            gr.Markdown(
                                f"**{name}** &nbsp; {badge}  \n"
                                f"{desc}  \n"
                                f"VRAM: ~{info.get('vram_needed_gb','?')} GB &nbsp;·&nbsp; "
                                f"Steps: {info.get('min_steps')}–{info.get('max_steps')} &nbsp;·&nbsp; "
                                f"Default: {info.get('default_steps')}"
                            )
                    with gr.Column():
                        gr.Markdown("### API Endpoints")
                        gr.Markdown(
                            "- `GET  /health` — Server health & GPU status  \n"
                            "- `GET  /models` — List available models  \n"
                            "- `POST /generate` — Generate image (header: `X-API-Key`)  \n"
                            "- `GET  /docs` — Interactive API documentation"
                        )

        # ══ EVENT WIRING ══
        def on_model_change(selected_model, current_lora):
            info = flux_pipeline.get_model_info(selected_model)
            is_flux = selected_model.startswith("flux-")
            neg_label = "Negative Prompt" + (" (disabled for FLUX)" if is_flux else "")
            if selected_model in {"realvisxl-v5", "juggernaut-xl"}:
                guidance_min, guidance_max = 1.0, 12.0
            elif selected_model == "sd3.5-large":
                guidance_min, guidance_max = 1.0, 10.0
            else:
                guidance_min, guidance_max = 0.0, 10.0
            hint = (
                f"**{selected_model}** — {info.get('description', '')}  \n"
                f"Steps: {info['min_steps']}–{info['max_steps']} &nbsp;·&nbsp; "
                f"Default: {info['default_steps']} &nbsp;·&nbsp; "
                f"Guidance: {info.get('default_guidance_scale', 3.5)} &nbsp;·&nbsp; "
                f"VRAM: ~{info['vram_needed_gb']} GB"
            )
            lora_choices = ["None"] + flux_pipeline.get_compatible_loras(selected_model)
            lora_value = current_lora if current_lora in lora_choices else "None"
            return (
                gr.update(value=hint),
                gr.update(minimum=info["min_steps"], maximum=info["max_steps"], value=info["default_steps"]),
                gr.update(label=neg_label),
                gr.update(minimum=guidance_min, maximum=guidance_max, value=info.get("default_guidance_scale", 3.5)),
                gr.update(choices=lora_choices, value=lora_value),
                gr.update(value=flux_pipeline.get_recommended_lora_scale(selected_model)),
            )

        def on_lora_change(selected_lora, current_model):
            if not selected_lora or selected_lora == "None":
                return (
                    gr.update(value=current_model), gr.update(),
                    gr.update(value=flux_pipeline.get_recommended_lora_scale(current_model)), "",
                )
            target_model = flux_pipeline.pick_model_for_lora(selected_lora, current_model)
            compatible = ["None"] + flux_pipeline.get_compatible_loras(target_model)
            chosen_lora = selected_lora if selected_lora in compatible else "None"
            recommended = flux_pipeline.get_recommended_lora_scale(target_model)
            if chosen_lora == "None":
                status = (
                    f"'{selected_lora}' is not compatible with {target_model}. "
                    "Choose a LoRA that appears in the dropdown for this model."
                )
                return (
                    gr.update(value=target_model),
                    gr.update(choices=compatible, value="None"),
                    gr.update(value=recommended), status,
                )
            if target_model != current_model:
                status = f"Auto-switched model: {current_model} → {target_model} to match LoRA '{selected_lora}'."
            else:
                status = f"LoRA '{selected_lora}' matched with {target_model}."
            return (
                gr.update(value=target_model),
                gr.update(choices=compatible, value=chosen_lora),
                gr.update(value=recommended), status,
            )

        compose_prompt_btn.click(
            fn=compose_prompt_lab,
            inputs=[prompt, lab_subject, lab_environment, lab_mood, lab_framing],
            outputs=[prompt],
        )
        profile_dropdown.change(
            fn=apply_quality_profile,
            inputs=[profile_dropdown, model_dropdown],
            outputs=[steps, guidance, resolution],
        )
        randomize_btn.click(
            fn=randomize_creative_setup,
            inputs=[model_dropdown],
            outputs=[style, resolution, steps, guidance, seed, auto_improve],
        )
        model_dropdown.change(
            fn=on_model_change,
            inputs=[model_dropdown, lora_dropdown],
            outputs=[model_hint, steps, negative_prompt, guidance, lora_dropdown, lora_scale],
        )
        lora_dropdown.change(
            fn=on_lora_change,
            inputs=[lora_dropdown, model_dropdown],
            outputs=[model_dropdown, lora_dropdown, lora_scale, lora_upload_status],
        )
        generate_btn.click(
            fn=generate_image,
            inputs=[
                prompt, negative_prompt, model_dropdown, style, resolution,
                steps, guidance, seed, lora_dropdown, lora_scale, auto_improve,
            ],
            outputs=[output_image, improved_prompt, info_box, used_seed, gallery, gpu_status_bar],
        )
        prompt.submit(
            fn=generate_image,
            inputs=[
                prompt, negative_prompt, model_dropdown, style, resolution,
                steps, guidance, seed, lora_dropdown, lora_scale, auto_improve,
            ],
            outputs=[output_image, improved_prompt, info_box, used_seed, gallery, gpu_status_bar],
        )
        lora_upload.change(
            fn=upload_lora_files,
            inputs=[lora_upload, model_dropdown],
            outputs=[lora_dropdown, lora_upload_status],
        )
        refresh_btn.click(
            fn=lambda selected_model: gr.update(
                choices=["None"] + flux_pipeline.get_compatible_loras(selected_model),
                value="None",
            ),
            inputs=[model_dropdown],
            outputs=[lora_dropdown],
        )
        check_upload_btn.click(
            fn=get_lora_server_status,
            inputs=[model_dropdown],
            outputs=[lora_upload_status],
        )

    return demo