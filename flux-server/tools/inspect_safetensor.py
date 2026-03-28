from pathlib import Path
import os
from safetensors.torch import load_file

p = Path("/opt/flux-server/loras/flux_dev.safetensors")
print(f"exists={p.exists()}")
if not p.exists():
    raise SystemExit(0)

print(f"size_gb={round(os.path.getsize(p) / 1024 / 1024 / 1024, 2)}")
tensors = load_file(str(p), device="cpu")
keys = list(tensors.keys())
print(f"tensor_count={len(keys)}")
print("first_keys=")
for k in keys[:20]:
    print(k)

joined = " ".join(keys[:300]).lower()
if any(x in joined for x in ["lora_unet", "lora_te", "lora_up", "lora_down"]):
    print("detected=likely_lora")
elif any(x in joined for x in ["double_blocks", "single_blocks", "time_text_embed", "x_embedder"]):
    print("detected=likely_flux_full_checkpoint")
else:
    print("detected=unknown_full_or_custom")
