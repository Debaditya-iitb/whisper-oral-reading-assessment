#!/bin/bash
# STEP 4 — fine-tune. Runs directly on the GPU box (no SLURM needed).
#
#   bash 05_finetune_all_stages.sh                    # all four stages, in order
#   bash 05_finetune_all_stages.sh W                  # one stage
#   bash 05_finetune_all_stages.sh A B                # a subset, in the order given
#   MODEL_TAG=whisper-medium bash 05_finetune_all_stages.sh
#   USE_SLURM=1 bash 05_finetune_all_stages.sh        # submit as dependent SLURM jobs
#
# Stages:
#   W  pretrained -> WPP            children only          CONTROL
#   A  pretrained -> IITM           adult long-form read
#   B  stage A    -> WPP            sequential: adult then children
#   C  pretrained -> WPP + IITM     pooled, one run
#
# B depends on A. W and C depend on nothing. Default order W A B C puts the
# cheapest, most informative run first so a mistake surfaces in ~30 min rather
# than after the adult stage.
#
# Each stage writes run_config.json and dev_metrics.json into its output dir,
# and skips itself if that output already has a trained model — so re-running
# after an interruption resumes at the stage that failed.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config_experiment.sh"
cd "$WHISPER_HOME"

STAGES=("$@")
[ ${#STAGES[@]} -eq 0 ] && STAGES=(W A B C)

if [ "${USE_SLURM:-0}" = "1" ]; then
    echo "[INFO] submitting to SLURM instead of running here"
    exec bash slurm_submit_all_stages.sh
fi

activate_env

# --- GPU selection -------------------------------------------------------
# HF Trainer moves everything to CUDA by itself, so there is nothing to set for
# the single-GPU case. Multiple visible GPUs is the trap: without torchrun the
# Trainer falls back to DataParallel AND multiplies the batch by the device
# count, changing the recipe. Default to one GPU so the run is reproducible;
# opt in explicitly with NUM_GPUS.
N_VISIBLE=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo 0)
NUM_GPUS="${NUM_GPUS:-1}"

if [ "$N_VISIBLE" -eq 0 ]; then
    echo "[FATAL] nvidia-smi sees no GPU. Fix that before training —"
    echo "        the trainer would otherwise run on CPU, ~100x slower."
    exit 1
fi

LAUNCH=(python)
if [ "$NUM_GPUS" -le 1 ]; then
    export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
    # Containers (and previous torchrun runs) often leave LOCAL_RANK set.
    # accelerate treats that as "this is DDP" and then dies on the missing
    # WORLD_SIZE, so clear the whole set for the single-process path.
    unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE GROUP_RANK \
          MASTER_ADDR MASTER_PORT TORCHELASTIC_RUN_ID
    echo "[INFO] using physical GPU $CUDA_VISIBLE_DEVICES of $N_VISIBLE" \
         "(torch will call it cuda:0 — that is expected)"
    [ "$N_VISIBLE" -gt 1 ] && echo \
        "       GPU_ID=<n> picks a different one; NUM_GPUS=$N_VISIBLE runs real DDP."
else
    # Real DDP. Effective batch = per_device * NUM_GPUS * accum, so divide the
    # per-device batch to keep the effective batch at 32 and the recipe intact.
    if [ $((BS % NUM_GPUS)) -ne 0 ]; then
        echo "[FATAL] per-device batch $BS is not divisible by NUM_GPUS=$NUM_GPUS."
        echo "        Set BS_OVERRIDE to a multiple, e.g. BS_OVERRIDE=$((NUM_GPUS*4))"
        exit 1
    fi
    BS=$((BS / NUM_GPUS))
    LAUNCH=(torchrun --standalone --nproc_per_node="$NUM_GPUS")
    echo "[INFO] DDP across $NUM_GPUS GPUs; per-device batch reduced to $BS so the"
    echo "       effective batch stays $((BS * NUM_GPUS * ACC))."
fi

echo "=============================================================="
echo " STEP 4 — fine-tuning   model=$MODEL_TAG"
echo " GPUs: $NUM_GPUS of $N_VISIBLE visible   ${GPU_NAME:-unknown} ${GPU_MEM_MB} MiB"
echo " batch ${BS} x ${ACC} x ${NUM_GPUS} gpu = effective $((BS*ACC*NUM_GPUS))"
echo "=============================================================="

stage_spec () {   # sets BASE TRAIN DEV OUT LR EP WARM NOTE
    case "$1" in
      W) BASE=$PRETRAINED;  TRAIN=$HF/wpp_train;      DEV=$HF/wpp_dev;  OUT=$STAGE_W
         LR=1e-5;   EP=10; WARM=0.05
         NOTE="stage W: pretrained -> WPP only (control for B)" ;;
      A) BASE=$PRETRAINED;  TRAIN=$HF/iitm_train;     DEV=$HF/iitm_dev; OUT=$STAGE_A
         LR=1e-5;   EP=6;  WARM=0.05
         NOTE="stage A: pretrained -> IITM (adult long-form read)" ;;
      B) BASE=$STAGE_A;     TRAIN=$HF/wpp_train;      DEV=$HF/wpp_dev;  OUT=$STAGE_B
         # Lower LR and shorter warmup: this starts from an already-adapted
         # checkpoint, and a full-strength second pass washes stage A out.
         LR=7.5e-6; EP=10; WARM=0.02
         NOTE="stage B: stageA -> WPP (sequential)" ;;
      C) BASE=$PRETRAINED;  TRAIN=$HF/combined_train; DEV=$HF/wpp_dev;  OUT=$STAGE_C
         LR=1e-5;   EP=8;  WARM=0.05
         NOTE="stage C: pretrained -> WPP+IITM pooled" ;;
      *) echo "[FATAL] unknown stage '$1' (use W A B C)"; exit 1 ;;
    esac
}

for st in "${STAGES[@]}"; do
    stage_spec "$st"
    echo
    echo "--------------------------------------------------------------"
    echo " $NOTE"
    echo "--------------------------------------------------------------"

    if [ -f "$OUT/model.safetensors" ] || [ -f "$OUT/pytorch_model.bin" ]; then
        echo "[SKIP] $OUT already trained (delete it to redo)"
        continue
    fi
    for d in "$BASE" "$TRAIN" "$DEV"; do
        [ -e "$d" ] || { echo "[FATAL] missing input: $d"; exit 1; }
    done

    mkdir -p "$(dirname "$OUT")"
    start=$(date +%s)
    "${LAUNCH[@]}" finetune_whisper_seq2seq.py \
        --base_model    "$BASE" \
        --train_dataset "$TRAIN" \
        --dev_dataset   "$DEV" \
        --output_dir    "$OUT" \
        --learning_rate "$LR" \
        --epochs        "$EP" \
        --warmup_ratio  "$WARM" \
        --batch_size    "$BS" \
        --grad_accum    "$ACC" \
        --stage_note    "$NOTE" \
        2>&1 | tee "$OUT.log"
    echo "[TIME] stage $st took $(( ($(date +%s) - start) / 60 )) min"
    cat "$OUT/dev_metrics.json" 2>/dev/null || true
done

echo
echo "[DONE] step 4. Next:  bash 06_evaluate_on_mps_testset.sh"
