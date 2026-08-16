#!/bin/bash
#SBATCH --job-name=wh_ft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=47:59:59
#SBATCH --output=job.%j.out
#SBATCH --error=job.%j.err
#SBATCH --partition=dgx
#SBATCH --qos=dgx

# One fine-tuning stage. Selected with STAGE=A|B|W|C, e.g.
#   sbatch --export=ALL,STAGE=A slurm_finetune.sh
# slurm_submit_all_stages.sh chains all four with the right dependencies.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config_experiment.sh"
activate_env
cd "$HERE"

STAGE=${STAGE:?set STAGE=A|B|W|C}

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch,transformers;print('torch',torch.__version__,'tf',transformers.__version__,'cuda',torch.cuda.is_available())"

# Dev set choice, per stage. Stage A adapts to adult speech, so it is model-
# selected on adult dev. Every stage that ends on children is selected on the
# WPP dev set — closest available proxy for the MPS test set, which must never
# be looked at during training.
case "$STAGE" in
  A)  BASE=$PRETRAINED;  TRAIN=$HF/iitm_train;     DEV=$HF/iitm_dev;  OUT=$STAGE_A
      LR=1e-5;    EP=6;   NOTE="stage A: pretrained -> IITM (adult)" ;;
  B)  BASE=$STAGE_A;     TRAIN=$HF/wpp_train;      DEV=$HF/wpp_dev;   OUT=$STAGE_B
      # Lower LR than stage A: the model already carries the adult adaptation
      # and a full-strength second pass tends to wash it out. Shorter warmup for
      # the same reason — the optimiser is not starting from a cold checkpoint.
      LR=7.5e-6;  EP=10;  NOTE="stage B: stageA -> WPP (children, sequential)" ;;
  W)  BASE=$PRETRAINED;  TRAIN=$HF/wpp_train;      DEV=$HF/wpp_dev;   OUT=$STAGE_W
      LR=1e-5;    EP=10;  NOTE="stage W: pretrained -> WPP only (control for B)" ;;
  C)  BASE=$PRETRAINED;  TRAIN=$HF/combined_train; DEV=$HF/wpp_dev;   OUT=$STAGE_C
      LR=1e-5;    EP=8;   NOTE="stage C: pretrained -> WPP+IITM pooled, one run" ;;
  *)  echo "[FATAL] unknown STAGE=$STAGE"; exit 1 ;;
esac

WARM=0.05
[ "$STAGE" = "B" ] && WARM=0.02

for d in "$BASE" "$TRAIN" "$DEV"; do
    [ -e "$d" ] || { echo "[FATAL] missing input: $d"; exit 1; }
done

echo "=============================================================="
echo " $NOTE"
echo "=============================================================="

python finetune_whisper_seq2seq.py \
    --base_model    "$BASE" \
    --train_dataset "$TRAIN" \
    --dev_dataset   "$DEV" \
    --output_dir    "$OUT" \
    --learning_rate "$LR" \
    --epochs        "$EP" \
    --warmup_ratio  "$WARM" \
    --batch_size    "$BS" \
    --grad_accum    "$ACC" \
    --stage_note    "$NOTE"

echo "[DONE] $STAGE -> $OUT"
cat "$OUT/dev_metrics.json" 2>/dev/null || true
