import gradio as gr
import base64
import io
import logging
from PIL import Image
from app.pipeline import flux_pipeline
from app.config import get_settings

logger = logging.getLogger(__name__)

# Prompt improvement templates — adds cinematic/artistic detail
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


def improve_prompt(prompt: str, style: str) -> str:
    """Enhance the user's prompt with the selected style preset."""
    if not prompt.strip():
        return prompt
    suffix = STYLE_PRESETS.get(style, "")
    return prompt.strip() + suffix


def generate_image(
    prompt: str,
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

    if not flux_pipeline.is_loaded:
        raise gr.Error("Model is still loading. Please wait and try again.")

    # Apply prompt improvement if enabled
    final_prompt = improve_prompt(prompt, style) if auto_improve else prompt

    # Resolve resolution
    width, height = RESOLUTION_PRESETS.get(resolution, (1024, 1024))

    # Seed: -1 or 0 means random
    use_seed = seed if seed > 0 else None

    try:
        img_b64, seed_used, elapsed_ms = flux_pipeline.generate(
            prompt=final_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            seed=use_seed,
            lora_name=lora_name,
            lora_scale=lora_scale,
        )
    except Exception as e:
        logger.exception("Generation failed")
        raise gr.Error(f"Generation failed: {e}")

    # Decode base64 to PIL Image for Gradio
    img_bytes = base64.b64decode(img_b64)
    image = Image.open(io.BytesIO(img_bytes))

    info_text = (
        f"Seed: {seed_used}  |  "
        f"Time: {elapsed_ms / 1000:.1f}s  |  "
        f"Size: {width}×{height}  |  "
        f"Steps: {steps}  |  "
        f"Guidance: {guidance}  |  "
        f"LoRA: {lora_name} ({lora_scale})"
    )

    return image, final_prompt, info_text, seed_used


def build_ui() -> gr.Blocks:
    """Build the Gradio Blocks UI."""
    settings = get_settings()
    model_name = settings.model_id.split("/")[-1]

    with gr.Blocks(
        title="FLUX Image Generator",
        theme=gr.themes.Soft(),
        css="""
            .generate-btn { min-height: 56px !important; font-size: 18px !important; }
            footer { display: none !important; }
        """,
    ) as demo:
        gr.Markdown(
            f"# FLUX Image Generator\n"
            f"**Model**: `{settings.model_id}` — NF4 quantized on NVIDIA L4 GPU"
        )

        with gr.Row():
            # Left column — controls
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the image you want to generate...",
                    lines=4,
                    max_lines=8,
                )

                with gr.Row():
                    auto_improve = gr.Checkbox(
                        label="Auto Improve Prompt",
                        value=True,
                    )
                    style = gr.Dropdown(
                        label="Style Preset",
                        choices=list(STYLE_PRESETS.keys()),
                        value="None",
                    )

                improved_prompt = gr.Textbox(
                    label="Final Prompt (after improvement)",
                    interactive=False,
                    lines=3,
                )

                with gr.Accordion("LoRA Settings", open=True):
                    loras = ["None"] + flux_pipeline.get_available_loras()
                    lora_dropdown = gr.Dropdown(
                        label="Select LoRA",
                        choices=loras,
                        value="None",
                        info="Place .safetensors files in the 'loras/' directory.",
                    )
                    lora_scale = gr.Slider(
                        label="LoRA Strength",
                        minimum=0.0,
                        maximum=2.0,
                        value=1.0,
                        step=0.05,
                    )
                    refresh_loras = gr.Button("Refresh LoRA List", size="sm")

                    def update_loras():
                        new_loras = ["None"] + flux_pipeline.get_available_loras()
                        return gr.update(choices=new_loras)

                    refresh_loras.click(fn=update_loras, outputs=lora_dropdown)

                resolution = gr.Dropdown(
                    label="Resolution",
                    choices=list(RESOLUTION_PRESETS.keys()),
                    value="1024 × 1024 (Square)",
                )

                steps = gr.Slider(
                    label="Inference Steps",
                    minimum=1,
                    maximum=50,
                    value=20,
                    step=1,
                    info="More steps = better quality, slower. 20 is a good default.",
                )

                guidance = gr.Slider(
                    label="Guidance Scale",
                    minimum=0.0,
                    maximum=20.0,
                    value=3.5,
                    step=0.5,
                    info="How closely to follow the prompt. 3.5 is recommended for FLUX.",
                )

                seed = gr.Number(
                    label="Seed",
                    value=-1,
                    precision=0,
                    info="Set to -1 for random. Use a specific number to reproduce results.",
                )

                generate_btn = gr.Button(
                    "Generate Image",
                    variant="primary",
                    elem_classes=["generate-btn"],
                )

            # Right column — output
            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="Generated Image",
                    type="pil",
                    height=600,
                )
                info_box = gr.Textbox(
                    label="Generation Info",
                    interactive=False,
                )
                used_seed = gr.Number(
                    label="Seed Used",
                    interactive=False,
                    precision=0,
                )

        # Wire up the generate button
        generate_btn.click(
            fn=generate_image,
            inputs=[
                prompt, style, resolution, steps, guidance, seed, 
                lora_dropdown, lora_scale, auto_improve
            ],
            outputs=[output_image, improved_prompt, info_box, used_seed],
        )

        # Also trigger on Enter in prompt box
        prompt.submit(
            fn=generate_image,
            inputs=[
                prompt, style, resolution, steps, guidance, seed, 
                lora_dropdown, lora_scale, auto_improve
            ],
            outputs=[output_image, improved_prompt, info_box, used_seed],
        )

    return demo
