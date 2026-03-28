import gradio as gr
import base64
import io
import logging
import shutil
from pathlib import Path
from PIL import Image
from app.pipeline_new import inference_pipeline as flux_pipeline
from app.config import get_settings

logger = logging.getLogger(__name__)

# In-memory gallery for the current session
_gallery_images: list = []
MAX_GALLERY = 20

# Prompt improvement templates
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


def _format_bytes(num_bytes: int) -> str:
    """Human-readable byte size."""
    size = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024


def get_lora_server_status(current_model: str) -> str:
    """Summarize LoRA files currently available on the server."""
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
    """Return formatted GPU status string."""
    try:
        info = flux_pipeline.gpu_info()
        current = flux_pipeline.model_manager.current_model
        loaded = flux_pipeline.is_loaded
        pct = (info["used_gb"] / info["total_gb"] * 100) if info["total_gb"] > 0 else 0
        icon = "🟢" if loaded else "🟡"
        return (
            f"**{icon} {current}** &nbsp;|&nbsp; "
            f"GPU: {info['name']} &nbsp;|&nbsp; "
            f"VRAM: {info['used_gb']:.1f} / {info['total_gb']:.1f} GB ({pct:.0f}%)"
        )
    except Exception:
        return "⚠️ GPU status unavailable"


def improve_prompt(prompt: str, style: str) -> str:
    """Enhance the user's prompt with the selected style preset."""
    if not prompt.strip():
        return prompt
    suffix = STYLE_PRESETS.get(style, "")
    return prompt.strip() + suffix


def generate_image(
    prompt: str,
    negative_prompt: str,
    model_name: str,
    style: str,
    resolution: str,
    steps: int,
    guidance: float,
    seed: int,
    lora_name: str,
    lora_scale: float,
    auto_improve: bool,
):
    """Generate an image via the pipeline and return it to Gradio."""
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
        f"Model: {model_name}  |  Seed: {seed_used}  |  "
        f"Time: {elapsed_ms / 1000:.1f}s  |  Size: {width}×{height}  |  "
        f"Steps: {steps}  |  Guidance: {guidance}"
    )
    if lora_name and lora_name != "None":
        info_text += f"  |  LoRA: {lora_name} ({lora_scale})"
    if auto_negative_applied:
        info_text += "  |  Negative: auto"

    _gallery_images.insert(0, image)
    while len(_gallery_images) > MAX_GALLERY:
        _gallery_images.pop()

    gpu_status = get_gpu_status()
    return image, final_prompt, info_text, seed_used, list(_gallery_images), gpu_status


def upload_lora_files(files, current_model: str):
    """Handle one or more LoRA file uploads from the UI."""
    if not files:
        return gr.update(), "No files selected."
    lora_dir = Path("loras")
    lora_dir.mkdir(exist_ok=True)
    uploaded = []
    skipped = []
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
        parts.append(f"✅ Uploaded {len(uploaded)}: {', '.join(uploaded)}")
    if skipped:
        parts.append(f"⚠️ Skipped (not .safetensors): {', '.join(skipped)}")
    status = "  |  ".join(parts) if parts else "No valid files."
    status = status + "\n" + get_lora_server_status(current_model)
    return gr.update(choices=new_loras, value=last), status


def build_ui() -> gr.Blocks:
    """Build the Gradio Blocks UI with tabs, gallery, and LoRA management."""
    available_models = flux_pipeline.list_available_models()
    default_model_info = flux_pipeline.get_model_info("flux-1-dev")

    custom_css = """
        .generate-btn {
            min-height: 56px !important;
            font-size: 18px !important;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            border: 0 !important;
            color: white !important;
            border-radius: 12px !important;
            font-weight: bold !important;
        }
        .generate-btn:hover {
            box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4) !important;
        }
        .gradio-container { max-width: 1400px !important; }
        .panel {
            border: 1px solid #e2e8f0 !important;
            border-radius: 16px !important;
            padding: 12px !important;
        }
        .gpu-bar {
            background: linear-gradient(135deg, #f0f4ff, #e8f4fd) !important;
            border-radius: 12px !important;
            padding: 10px 16px !important;
            border: 1px solid #c7d2fe !important;
        }
        footer { display: none !important; }
    """

    with gr.Blocks(
        title="AI Image Studio",
    ) as demo:
        gr.Markdown(
            "# 🎨 AI Image Studio\n"
            "Generate images with **FLUX.1**, **SD3**, and **SDXL** on A100 GPU. "
            "Switch models, apply LoRA adapters, and use style presets."
        )

        gpu_status_bar = gr.Markdown(value=get_gpu_status(), elem_classes=["gpu-bar"])

        with gr.Tabs():
            # ── Generate Tab ──
            with gr.Tab("🖼️ Generate"):
                with gr.Row():
                    with gr.Column(scale=2, elem_classes=["panel"]):
                        model_dropdown = gr.Dropdown(
                            label="🤖 Model",
                            choices=list(available_models.keys()),
                            value="flux-1-dev",
                            info="Switching models unloads the previous one to free VRAM.",
                        )
                        model_hint = gr.Markdown("Select a model to see its profile.")

                        prompt = gr.Textbox(
                            label="✏️ Prompt",
                            placeholder="A majestic phoenix rising from golden flames, ultra detailed...",
                            lines=3, max_lines=6,
                        )
                        negative_prompt = gr.Textbox(
                            label="🚫 Negative Prompt (not used for FLUX models)",
                            placeholder="blurry, low quality, distorted, watermark, text...",
                            lines=2, max_lines=4,
                        )

                        with gr.Row():
                            auto_improve = gr.Checkbox(label="Auto Improve", value=True)
                            style = gr.Dropdown(
                                label="Style Preset",
                                choices=list(STYLE_PRESETS.keys()),
                                value="None",
                            )

                        improved_prompt = gr.Textbox(label="Final Prompt", interactive=False, lines=2)

                        resolution = gr.Dropdown(
                            label="📐 Resolution",
                            choices=list(RESOLUTION_PRESETS.keys()),
                            value="1024 × 1024 (Square)",
                        )
                        with gr.Row():
                            steps = gr.Slider(
                                label="Steps",
                                minimum=default_model_info["min_steps"],
                                maximum=default_model_info["max_steps"],
                                value=default_model_info["default_steps"],
                                step=1,
                            )
                            guidance = gr.Slider(
                                label="Guidance Scale",
                                minimum=0.0,
                                maximum=20.0,
                                value=default_model_info.get("default_guidance_scale", 3.5),
                                step=0.5,
                            )

                        seed = gr.Number(label="🎲 Seed (-1 = random)", value=-1, precision=0)

                        with gr.Accordion("🎭 LoRA Adapter", open=True):
                            loras = ["None"] + flux_pipeline.get_compatible_loras("flux-1-dev")
                            lora_dropdown = gr.Dropdown(
                                label="Select LoRA", choices=loras, value="None",
                                info="All available LoRAs are shown.",
                            )
                            lora_scale = gr.Slider(
                                label="LoRA Strength",
                                minimum=0.0,
                                maximum=2.0,
                                value=flux_pipeline.get_recommended_lora_scale("flux-1-dev"),
                                step=0.05,
                            )
                            lora_upload = gr.File(
                                label="Upload LoRA(s) — select multiple .safetensors files",
                                file_types=[".safetensors"],
                                file_count="multiple",
                                type="filepath",
                            )
                            gr.Markdown(
                                "Upload note: for very large files (10GB+), browser uploads can take a long time. "
                                "Status updates after transfer completion."
                            )
                            lora_upload_status = gr.Textbox(
                                label="Upload Status", interactive=False, lines=8,
                                value=get_lora_server_status("flux-1-dev"),
                            )
                            refresh_btn = gr.Button("🔄 Refresh LoRA List", size="sm")
                            check_upload_btn = gr.Button("Check Server Upload Status", size="sm")

                        generate_btn = gr.Button(
                            "🚀 Generate Image",
                            variant="primary",
                            elem_classes=["generate-btn"],
                        )

                    with gr.Column(scale=3, elem_classes=["panel"]):
                        output_image = gr.Image(
                            label="Generated Image", type="pil",
                            height=600,
                        )
                        info_box = gr.Textbox(label="Generation Info", interactive=False)
                        used_seed = gr.Number(label="Seed Used", interactive=False, precision=0)

            # ── Gallery Tab ──
            with gr.Tab("📸 Gallery"):
                gr.Markdown("### Recent Generations\nImages generated during this session.")
                gallery = gr.Gallery(
                    label="History", columns=4, rows=3,
                    height=500, object_fit="contain",
                )

            # ── Models Tab ──
            with gr.Tab("📊 Models"):
                gr.Markdown("### Available Models")
                for name, desc in available_models.items():
                    info = flux_pipeline.get_model_info(name)
                    badge = "🟢 Loaded" if info.get("loaded") else "⚪ Available"
                    gr.Markdown(
                        f"**{name}** {badge}  \n"
                        f"{desc}  \n"
                        f"VRAM: ~{info.get('vram_needed_gb', '?')} GB | "
                        f"Steps: {info.get('min_steps')}–{info.get('max_steps')} | "
                        f"Default: {info.get('default_steps')}"
                    )
                gr.Markdown(
                    "---\n### API Endpoints\n"
                    "- `GET /health` — Server health and GPU status\n"
                    "- `GET /models` — List available models\n"
                    "- `POST /generate` — Generate image (header: `X-API-Key`)\n"
                    "- `GET /docs` — Interactive API docs"
                )

        # ── Event Handlers ──
        def on_model_change(selected_model, current_lora):
            info = flux_pipeline.get_model_info(selected_model)
            is_flux = selected_model.startswith("flux-")
            neg_label = "🚫 Negative Prompt" + (" (not used for FLUX)" if is_flux else "")
            if selected_model in {"realvisxl-v5", "juggernaut-xl"}:
                guidance_min, guidance_max = 1.0, 12.0
            elif selected_model == "sd3-medium":
                guidance_min, guidance_max = 1.0, 10.0
            else:
                guidance_min, guidance_max = 0.0, 10.0
            hint = (
                f"**{selected_model}** — {info.get('description', '')}  \n"
                f"Steps: {info['min_steps']}–{info['max_steps']} | "
                f"Default: {info['default_steps']} | Guidance default: {info.get('default_guidance_scale', 3.5)} | "
                f"VRAM: ~{info['vram_needed_gb']} GB"
            )
            lora_choices = ["None"] + flux_pipeline.get_compatible_loras(selected_model)
            lora_value = current_lora if current_lora in lora_choices else "None"
            return (
                gr.update(value=hint),
                gr.update(
                    minimum=info["min_steps"],
                    maximum=info["max_steps"],
                    value=info["default_steps"],
                ),
                gr.update(label=neg_label),
                gr.update(
                    minimum=guidance_min,
                    maximum=guidance_max,
                    value=info.get("default_guidance_scale", 3.5),
                ),
                gr.update(choices=lora_choices, value=lora_value),
                gr.update(value=flux_pipeline.get_recommended_lora_scale(selected_model)),
            )

        def on_lora_change(selected_lora, current_model):
            if not selected_lora or selected_lora == "None":
                return (
                    gr.update(value=current_model),
                    gr.update(),
                    gr.update(value=flux_pipeline.get_recommended_lora_scale(current_model)),
                    "",
                )

            target_model = flux_pipeline.pick_model_for_lora(selected_lora, current_model)
            compatible = ["None"] + flux_pipeline.get_compatible_loras(target_model)
            chosen_lora = selected_lora if selected_lora in compatible else "None"
            recommended = flux_pipeline.get_recommended_lora_scale(target_model)

            if chosen_lora == "None":
                status = (
                        f"'{selected_lora}' is not a compatible LoRA for {target_model}. "
                        "Please choose a LoRA that appears in the dropdown for this model."
                    )
                return (
                    gr.update(value=target_model),
                    gr.update(choices=compatible, value="None"),
                    gr.update(value=recommended),
                    status,
                )

            if target_model != current_model:
                status = (
                    f"Auto-switched model: {current_model} → {target_model} "
                    f"to match LoRA '{selected_lora}'."
                )
            else:
                status = f"LoRA '{selected_lora}' matched with {target_model}."

            return (
                gr.update(value=target_model),
                gr.update(choices=compatible, value=chosen_lora),
                gr.update(value=recommended),
                status,
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
