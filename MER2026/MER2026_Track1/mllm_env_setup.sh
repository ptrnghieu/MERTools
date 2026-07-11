#!/bin/bash
# Environment setup for the Cognitive-MLLM pipeline on an A100 (Ampere sm_80,
# CUDA 12.x). Pins versions known-good on A100 -- NO Blackwell/sm_120 surprises.
# flash-attn is optional (falls back to torch sdpa); vLLM is installed in a
# SEPARATE venv because it pins torch versions that clash with the trainer.
#
#   bash mllm_env_setup.sh            # creates ./mllm_env (training/inference)
#   source mllm_env/bin/activate
#   # later, for fast 20k inference:  bash mllm_env_setup.sh vllm
#
# Keep the big HF cache on the large /workspace disk.
set -euo pipefail
cd "$(dirname "$0")"

export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
mkdir -p "$HF_HOME"
echo "HF_HOME=$HF_HOME  (add to ~/.bashrc so downloads land on the big disk)"

# ---------------------------------------------------------------- vLLM branch
if [ "${1:-}" = "vllm" ]; then
  echo "===== installing vLLM in a SEPARATE venv (./vllm_env) ====="
  python3 -m venv vllm_env
  # shellcheck disable=SC1091
  source vllm_env/bin/activate
  pip install -U pip wheel
  pip install "vllm>=0.7.2"          # brings its own torch; supports Qwen2.5-VL
  pip install "qwen-vl-utils[decord]"
  python -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"
  echo "vLLM env ready -> source vllm_env/bin/activate"
  exit 0
fi

# ------------------------------------------------------------ training branch
echo "===== creating ./mllm_env (torch cu124 + transformers + peft/trl + bnb) ====="
python3 -m venv mllm_env
# shellcheck disable=SC1091
source mllm_env/bin/activate
pip install -U pip wheel setuptools

# torch 2.5.1 + cu124: rock-solid on A100 (sm_80), ships flash-sdpa kernels
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# HF training stack (transformers >=4.49 has Qwen2.5-VL)
pip install \
  "transformers==4.51.3" \
  "accelerate==1.4.0" \
  "peft==0.14.0" \
  "trl==0.15.2" \
  "datasets==3.3.2" \
  "bitsandbytes==0.45.3" \
  "qwen-vl-utils[decord]==0.0.10" \
  sentencepiece pillow numpy pandas scikit-learn tqdm

# flash-attn: OPTIONAL. Big speedup but needs nvcc + a few min to build.
# If it fails, the trainer still runs with attn_implementation="sdpa".
echo "===== attempting flash-attn (optional; sdpa fallback if this fails) ====="
pip install flash-attn==2.7.4.post1 --no-build-isolation || \
  echo "!! flash-attn install failed -- use attn_implementation='sdpa' (fine on A100)"

echo "===== GPU / stack sanity check ====="
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda avail", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA not visible!"
print("gpu:", torch.cuda.get_device_name(0), "| capability", torch.cuda.get_device_capability(0))
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
print("bf16 matmul ok:", float((x @ x).float().sum()))
import bitsandbytes as bnb; print("bitsandbytes", bnb.__version__, "ok")
try:
    import flash_attn; print("flash_attn", flash_attn.__version__, "ok")
except Exception as e:
    print("flash_attn NOT available -> attn_implementation='sdpa'")
import transformers; print("transformers", transformers.__version__)
try:
    from transformers import Qwen2_5_VLForConditionalGeneration  # noqa
    print("Qwen2.5-VL class available")
except Exception as e:
    print("!! Qwen2.5-VL class missing -- bump transformers:", repr(e)[:100])
PY

echo "===== DONE. Activate with: source mllm_env/bin/activate ====="
echo "Next: set HF token if the model is gated ->  export HF_TOKEN=..."
echo "Model to pull:  Qwen/Qwen2.5-VL-7B-Instruct  (~16GB, into $HF_HOME)"
