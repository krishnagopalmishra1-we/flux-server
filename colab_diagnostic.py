import subprocess
import sys

print("=" * 60)
print("COLAB ENVIRONMENT DIAGNOSTIC REPORT")
print("=" * 60)

# 1. Python version
print(f"\n[1] Python version: {sys.version}")
print(f"    Python executable: {sys.executable}")

# 2. pip version
subprocess.run([sys.executable, "-m", "pip", "--version"])

# 3. CUDA / GPU info
print("\n[3] GPU Info:")
subprocess.run("nvidia-smi", shell=True)

# 4. Pre-installed packages (FULL LIST)
print("\n[4] ALL pre-installed packages:")
subprocess.run([sys.executable, "-m", "pip", "list", "--format=columns"])

# 5. Test if transformers BitsAndBytesConfig imports without bitsandbytes
print("\n[5] Testing imports WITHOUT installing anything:")
tests = [
    ("import torch; print(f'  torch={torch.__version__}, CUDA={torch.version.cuda}')", "torch"),
    ("from transformers import BitsAndBytesConfig; print('  BitsAndBytesConfig: OK')", "transformers.BitsAndBytesConfig"),
    ("import diffusers; print(f'  diffusers={diffusers.__version__}')", "diffusers"),
    ("import transformers; print(f'  transformers={transformers.__version__}')", "transformers"),
    ("import bitsandbytes; print(f'  bitsandbytes={bitsandbytes.__version__}')", "bitsandbytes"),
    ("import accelerate; print(f'  accelerate={accelerate.__version__}')", "accelerate"),
    ("import safetensors; print(f'  safetensors={safetensors.__version__}')", "safetensors"),
    ("import peft; print(f'  peft={peft.__version__}')", "peft"),
    ("import sentencepiece; print(f'  sentencepiece={sentencepiece.__version__}')", "sentencepiece"),
    ("import fastapi; print(f'  fastapi={fastapi.__version__}')", "fastapi"),
    ("import uvicorn; print(f'  uvicorn={uvicorn.__version__}')", "uvicorn"),
    ("import PIL; print(f'  Pillow={PIL.__version__}')", "Pillow"),
    ("import numpy; print(f'  numpy={numpy.__version__}')", "numpy"),
    ("import scipy; print(f'  scipy={scipy.__version__}')", "scipy"),
    ("import einops; print(f'  einops={einops.__version__}')", "einops"),
    ("import pydantic_settings; print(f'  pydantic_settings={pydantic_settings.__version__}')", "pydantic-settings"),
    ("import huggingface_hub; print(f'  huggingface_hub={huggingface_hub.__version__}')", "huggingface-hub"),
    ("import protobuf; print(f'  protobuf: OK')", "protobuf"),
    ("import aiofiles; print(f'  aiofiles={aiofiles.__version__}')", "aiofiles"),
]

for code, label in tests:
    try:
        exec(code)
    except Exception as e:
        print(f"  {label}: NOT AVAILABLE ({type(e).__name__}: {e})")

# 6. Check available bitsandbytes wheels for this Python version
print(f"\n[6] Checking PyPI for bitsandbytes wheels for Python {sys.version_info.major}.{sys.version_info.minor}:")
subprocess.run([sys.executable, "-m", "pip", "install", "--dry-run", "--no-deps", "bitsandbytes>=0.43.3"], capture_output=False)

# 7. Check available diffusers wheels
print(f"\n[7] Checking PyPI for diffusers==0.35.1 wheel:")
subprocess.run([sys.executable, "-m", "pip", "install", "--dry-run", "--no-deps", "diffusers==0.35.1"], capture_output=False)

# 8. Check available transformers wheels
print(f"\n[8] Checking PyPI for transformers==4.44.2 wheel:")
subprocess.run([sys.executable, "-m", "pip", "install", "--dry-run", "--no-deps", "transformers==4.44.2"], capture_output=False)

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
