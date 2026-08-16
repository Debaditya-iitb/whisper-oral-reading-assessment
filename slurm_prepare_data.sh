#!/bin/bash
#SBATCH --job-name=wh_prep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=11:59:59
#SBATCH --output=job.%j.out
#SBATCH --error=job.%j.err
#SBATCH --partition=a40
#SBATCH --qos=a40

# Build the HuggingFace arrow datasets every fine-tuning stage consumes.
# CPU only — feature extraction is numpy, a GPU here just queues longer.
#
# One dataset per (model size x corpus x split). The mel-bin count and the
# tokenizer are baked in, so these are NOT reusable across model sizes: switch
# MODEL_TAG and you must re-run this.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config_experiment.sh"
activate_env
cd "$HERE"

echo "[INFO] model=$MODEL_TAG  pretrained=$PRETRAINED"
[ -d "$PRETRAINED" ] || { echo "[FATAL] run tool_fetch_pretrained_checkpoint.sh $MODEL_TAG on the login node first"; exit 1; }

build () {   # build <corpus> <split>
    local corpus=$1 split=$2
    local csv="$DATA/$split/$corpus/chunks.csv"
    local out="$HF/${corpus}_${split}"
    if [ ! -f "$csv" ]; then
        echo "[SKIP] $csv not found (run 03_build_chunked_datasets.sh first)"
        return 0
    fi
    if [ -d "$out" ]; then
        echo "[SKIP] $out already exists"
        return 0
    fi
    echo "=== $corpus/$split -> $out ==="
    python build_arrow_datasets.py \
        --model_path "$PRETRAINED" \
        --input_csv  "$csv" \
        --save_path  "$out" \
        --language english --task transcribe \
        --num_proc 8
}

build wpp      train
build wpp      dev
build iitm     train
build iitm     dev
build combined train
build combined dev

echo
echo "[DONE] arrow datasets under $HF"
du -sh "$HF"/* 2>/dev/null || true
