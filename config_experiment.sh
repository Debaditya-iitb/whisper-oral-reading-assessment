#!/bin/bash
# Shared configuration for the staged Whisper experiments.
# Sourced by every run_stage_*.sh — edit here, not in the individual scripts.

# All machine-specific paths live in config_paths.sh — never hardcode one here.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config_paths.sh"

HERE=$WHISPER_HOME
MAN=$MANIFESTS

# whisper-small to get the loop working end to end; switch to whisper-medium
# for the real numbers. Changing this REQUIRES rebuilding the arrow datasets
# (mel-bin count and tokenizer are baked in) — slurm_prepare_data.sh does that.
MODEL_TAG=${MODEL_TAG:-whisper-small}
export MODEL_TAG
PRETRAINED=$PRETRAINED_ROOT/$MODEL_TAG

HF=$HF_ROOT/$MODEL_TAG                           # arrow datasets, per model size
MODELS=$MODELS_ROOT/$MODEL_TAG                   # fine-tuned checkpoints
DEC=$DECODE_ROOT/$MODEL_TAG                      # hypotheses
RES=$RESULTS_ROOT/$MODEL_TAG                     # per-grade reports

# Per-device batch x grad-accum, sized to the GPU. Effective batch stays 32 in
# every case, so results are comparable across machines — only the step count
# and wall time change.
# Which physical GPU to use. GPU_ID is the friendly name for it; an already-set
# CUDA_VISIBLE_DEVICES wins so existing habits keep working. Empty means "GPU 0".
#   GPU_ID=3 bash 05_finetune_all_stages.sh
#   CUDA_VISIBLE_DEVICES=3 bash 05_finetune_all_stages.sh      # equivalent
GPU_ID="${GPU_ID:-${CUDA_VISIBLE_DEVICES%%,*}}"

# Probe the GPU we will actually use, not physical GPU 0 — they differ on a
# heterogeneous box, and nvidia-smi ignores CUDA_VISIBLE_DEVICES.
_SMI_SEL=()
[ -n "${GPU_ID:-}" ] && _SMI_SEL=(-i "$GPU_ID")
GPU_NAME="$(nvidia-smi "${_SMI_SEL[@]}" --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
GPU_MEM_MB="$(nvidia-smi "${_SMI_SEL[@]}" --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)"; GPU_MEM_MB=${GPU_MEM_MB:-0}

if echo "$GPU_NAME" | grep -qiE "H100|H200|B200"; then
    case "$MODEL_TAG" in
        whisper-small)  BS=32; ACC=1 ;;
        whisper-medium) BS=16; ACC=2 ;;
        *)              BS=8;  ACC=4 ;;   # large-v3
    esac
elif [ "${GPU_MEM_MB:-0}" -ge 70000 ]; then      # A100-80G / L40S-48G class
    case "$MODEL_TAG" in
        whisper-small)  BS=16; ACC=2 ;;
        whisper-medium) BS=8;  ACC=4 ;;
        *)              BS=4;  ACC=8 ;;
    esac
else                                              # <=48 GB, or no nvidia-smi
    case "$MODEL_TAG" in
        whisper-small)  BS=8;  ACC=4 ;;
        whisper-medium) BS=4;  ACC=8 ;;
        *)              BS=2;  ACC=16 ;;
    esac
fi
# Override from the environment if you are tuning: BS=24 ACC=2 bash 05_finetune_all_stages.sh
BS="${BS_OVERRIDE:-$BS}"
ACC="${ACC_OVERRIDE:-$ACC}"

# Stage names -> output checkpoint directories.
#   A  adult only            pretrained -> IITM
#   B  sequential            A          -> WPP        <- the thing you asked for
#   W  children only         pretrained -> WPP        <- the control for B
#   C  pooled, single run    pretrained -> WPP+IITM
STAGE_A=$MODELS/stageA_iitm
STAGE_B=$MODELS/stageB_iitm_then_wpp
STAGE_W=$MODELS/stageW_wpp_only
STAGE_C=$MODELS/stageC_combined

