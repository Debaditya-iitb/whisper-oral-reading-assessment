#!/bin/bash
# Submit the whole experiment chain with SLURM dependencies, so it runs
# unattended and a failure stops everything downstream instead of training on
# a half-built dataset.
#
#   bash slurm_submit_all_stages.sh                    # whisper-small
#   MODEL_TAG=whisper-medium bash slurm_submit_all_stages.sh
#
# Chain:
#   prep ──┬─> A ──> B ──┐
#          ├─> W ────────┼──> eval (decode all on MPS, score per grade)
#          └─> C ────────┘
#
# A/W/C are independent and run in parallel if the queue allows; B waits for A.
# `afterok` means a failed stage stops its dependents rather than silently
# feeding them a missing checkpoint.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config_experiment.sh"
cd "$HERE"

echo "model: $MODEL_TAG"
[ -d "$PRETRAINED" ] || {
    echo "[FATAL] $PRETRAINED missing."
    echo "        Run on the LOGIN node first:  bash tool_fetch_pretrained_checkpoint.sh $MODEL_TAG"
    exit 1
}
[ -f "$DATA/train/wpp/chunks.csv" ] || {
    echo "[FATAL] data missing. Run:  bash 03_build_chunked_datasets.sh"
    exit 1
}

HAVE_IITM=0
[ -f "$DATA/train/iitm/chunks.csv" ] && HAVE_IITM=1

jid () { sbatch --parsable "$@"; }

PREP=$(jid slurm_prepare_data.sh)
echo "prep      : $PREP"

W=$(jid --dependency=afterok:$PREP --export=ALL,STAGE=W slurm_finetune.sh)
echo "stageW    : $W   (pretrained -> WPP, control)"

DEPS="afterok:$W"

if [ "$HAVE_IITM" = "1" ]; then
    A=$(jid --dependency=afterok:$PREP --export=ALL,STAGE=A slurm_finetune.sh)
    echo "stageA    : $A   (pretrained -> IITM, adult)"
    B=$(jid --dependency=afterok:$A --export=ALL,STAGE=B slurm_finetune.sh)
    echo "stageB    : $B   (stageA -> WPP, sequential)"
    C=$(jid --dependency=afterok:$PREP --export=ALL,STAGE=C slurm_finetune.sh)
    echo "stageC    : $C   (pretrained -> WPP+IITM pooled)"
    DEPS="$DEPS:$B:$C"
else
    echo "stageA/B/C: SKIPPED — IITM word transcripts not available yet."
    echo "            Re-run 03_build_chunked_datasets.sh, then re-run this script."
fi

EVAL=$(jid --dependency="$DEPS" slurm_evaluate.sh)
echo "eval      : $EVAL  (decode all on MPS + per-grade scoring)"
echo
echo "watch:   squeue -u \$USER"
echo "results: $RES/SUMMARY.md"
