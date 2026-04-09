# Diffusers + BitsAndBytes Quantization Error

## Issue
When heavily quantizing a pipeline component (like a 14B transformer) using `BitsAndBytesConfig` (e.g. `load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`), the model weights are natively managed by the `bitsandbytes` hooks on the GPU.

If you attempt to call `.to("cuda")` or `.to(device)` on the entire Diffusion pipeline, it propagates to the quantized component and throws the following fatal exception:

> [!WARNING]
> `NotImplementedError: You are trying to call .to on a model that has been loaded with 'bitsandbytes' quantization. Please upgrade the installation...`

## Solution
Instead of universally moving the entire pipeline:

```python
# ❌ INCORRECT (Triggers Error)
self._pipe = WanPipeline.from_pretrained(model_id, transformer=nf4_transformer)
self._pipe.to(self.device)
```

You must surgically iterate through the pipeline's subcomponents and move **only** the non-quantized pieces (like the `text_encoder`, `vae`, and `image_encoder`) to the GPU:

```python
# ✅ CORRECT (Safe for NF4 / 4-bit)
self._pipe = WanPipeline.from_pretrained(model_id, transformer=nf4_transformer)

for name, component in self._pipe.components.items():
    if name != "transformer" and hasattr(component, "to"):
        component.to(self.device)
```

This prevents touching the internal memory hooks established by `bitsandbytes` while ensuring the rest of the generation pipeline correctly routes tensors through CUDA.
