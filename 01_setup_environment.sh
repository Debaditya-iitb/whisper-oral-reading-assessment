#!/bin/bash
# STEP 1 — build the Python environment on a fresh server.
#
#   bash 01_setup_environment.sh
#   ENV_KIND=venv bash 01_setup_environment.sh        # skip conda even if it exists
#   CUDA_TAG=cu124 bash 01_setup_environment.sh       # different CUDA wheel
#
# Creates a `whisper` environment, installs a CUDA build of torch that supports
# H100 (sm_90), installs the rest, then PROVES the GPU works before exiting.
# Safe to re-run: an existing environment is reused, not rebuilt.
#
# Why not `conda create --clone w2v` like the old cluster: that clone only
# exists on the original machine. This builds from scratch.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config_paths.sh"
cd "$WHISPER_HOME"

PY_VER="${PY_VER:-3.10}"
CUDA_TAG="${CUDA_TAG:-cu121}"     # cu121 covers H100 (sm_90). cu124 also fine.
TORCH_VER="${TORCH_VER:-2.4.1}"
ENV_KIND="${ENV_KIND:-auto}"      # auto | conda | venv
VENV_DIR="${VENV_DIR:-$WHISPER_HOME/.venv}"

echo "=============================================================="
echo " STEP 1 — environment"
echo "   python $PY_VER, torch $TORCH_VER+$CUDA_TAG, env name '$CONDA_ENV'"
echo "=============================================================="

if [ "$ENV_KIND" = "auto" ]; then
    if command -v conda >/dev/null 2>&1 || [ -f "$CONDA_SH" ]; then
        ENV_KIND=conda
    else
        ENV_KIND=venv
    fi
fi
echo "[INFO] using $ENV_KIND"

if [ "$ENV_KIND" = "conda" ]; then
    if [ -f "$CONDA_SH" ]; then
        # shellcheck disable=SC1090
        source "$CONDA_SH"
    else
        eval "$(conda shell.bash hook)"
    fi
    if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
        echo "[SKIP] conda env '$CONDA_ENV' already exists"
    else
        conda create -y -n "$CONDA_ENV" "python=$PY_VER"
    fi
    conda activate "$CONDA_ENV"
else
    if [ -d "$VENV_DIR" ]; then
        echo "[SKIP] venv already exists at $VENV_DIR"
    else
        python3 -m venv "$VENV_DIR"
    fi
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    echo "[NOTE] venv mode: put this in your shell profile so later steps find it"
    echo "         export CONDA_SH=/dev/null"
    echo "         source $VENV_DIR/bin/activate"
fi

python -m pip install --upgrade pip wheel

# torch FIRST and from the CUDA index. Installing it after (or letting another
# package pull it in) is how you end up with a CPU-only wheel on a GPU box.
if python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "[SKIP] torch already installed with CUDA: $(python -c 'import torch;print(torch.__version__)')"
else
    echo "[INFO] installing torch $TORCH_VER+$CUDA_TAG"
    python -m pip install "torch==$TORCH_VER" \
        --index-url "https://download.pytorch.org/whl/$CUDA_TAG"
fi

python -m pip install -r requirements.txt

echo
echo "=============================================================="
echo " verification"
echo "=============================================================="
python - <<'PY'
import sys, torch, transformers, datasets
print(f"python       {sys.version.split()[0]}")
print(f"torch        {torch.__version__}   cuda build {torch.version.cuda}")
print(f"transformers {transformers.__version__}")
print(f"datasets     {datasets.__version__}")

if not torch.cuda.is_available():
    print("\nFATAL: torch cannot see a GPU.")
    print("  - `nvidia-smi` working?")
    print("  - CPU-only wheel? reinstall: pip install torch --index-url "
          "https://download.pytorch.org/whl/cu121")
    sys.exit(1)

n = torch.cuda.device_count()
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f"gpu {i}        {p.name}  {p.total_memory/2**30:.0f} GiB  sm_{p.major}{p.minor}")

major = torch.cuda.get_device_properties(0).major
print(f"bf16         {'yes' if torch.cuda.is_bf16_supported() else 'NO — set USE_BF16=False, USE_FP16=True'}")
if major >= 9:
    print("hopper       yes — sdpa/FlashAttention-2 kernels and bf16 are the fast path")
elif major < 8:
    print("WARNING: pre-Ampere GPU. bf16 and TF32 unavailable; edit "
          "finetune_whisper_seq2seq.py to USE_BF16=False / USE_FP16=True")

x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize()
print(f"matmul check {(x @ x).float().sum().item():.3e}  (bf16 on device OK)")

import jiwer, soundfile  # noqa: F401
print("jiwer, soundfile  OK")
PY

echo
echo "[DONE] step 1. Next:  bash 02_download_pretrained_models.sh"
