#!/bin/bash
# STEP 3 — build the train/dev/test sets, then turn them into arrow datasets.
#
#   bash 04_build_arrow_datasets.sh
#   MODEL_TAG=whisper-medium bash 04_build_arrow_datasets.sh
#   SKIP_REBUILD=1 bash 04_build_arrow_datasets.sh     # data/ already correct, only arrow
#
# Two halves:
#   3a  corpora -> data/{train,dev,test}/...   (chunk manifests + Kaldi dirs)
#   3b  data/   -> hf_whisper/<model>/...      (log-Mel + BPE labels)
#
# 3b is model-size specific: mel bins (80 vs 128) and the tokenizer are baked
# into the arrow files. Switching MODEL_TAG means re-running 3b.
#
# Disk: ~1 MB per chunk. wpp_train 10.6 GB, iitm_train 19.3 GB,
# combined_train 19.5 GB, dev sets ~4 GB. Budget ~55 GB for whisper-small or
# medium, ~90 GB for large-v3.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config_experiment.sh"
activate_env
cd "$WHISPER_HOME"

echo "=============================================================="
echo " STEP 3 — data preparation   (model=$MODEL_TAG)"
echo "=============================================================="

[ -d "$PRETRAINED" ] || {
    echo "[FATAL] $PRETRAINED missing — run: bash 02_download_pretrained_models.sh $MODEL_TAG"
    exit 1
}

echo
echo "--- 3.0  path check ---------------------------------------"
python check_paths_resolve.py --sample 10 || {
    echo
    echo "[FATAL] audio not reachable."
    echo "If data/audio is present the folder should need no roots at all;"
    echo "otherwise set them and re-run:"
    echo "  export WPP_ROOT=/path/to/wpp-2020"
    echo "  export IITM_ROOT=/path/to/IITM_English_all_new"
    echo "  export MPS_ROOT=/path/to/mps_dataset"
    exit 1
}

if [ "${SKIP_REBUILD:-0}" = "1" ]; then
    echo
    echo "--- 3a  SKIPPED (SKIP_REBUILD=1) --------------------------"
else
    echo
    echo "--- 3a  corpora -> data/ ----------------------------------"
    # If data/audio already exists (audio shipped inside the folder), the
    # corpora are not needed at all and 3a is skipped: rebuilding would only
    # reproduce manifests that copy_audio_to_local_disk.py then has to re-point.
    if [ -d "$AUDIO_ROOT" ]; then
        echo "[SKIP] data/audio present — the sets are already built and"
        echo "       self-contained. Force a rebuild with FORCE_REBUILD=1."
        # wav.scp holds absolute paths and is stale after the folder moves
        # machine. Repair it here so the Kaldi dirs are correct on this box.
        python copy_audio_to_local_disk.py
        [ "${FORCE_REBUILD:-0}" = "1" ] && bash 03_build_chunked_datasets.sh
    else
        bash 03_build_chunked_datasets.sh
    fi
fi

echo
echo "--- 3b  data/ -> arrow datasets ---------------------------"
mkdir -p "$HF"

build () {   # build <corpus> <split>
    local corpus=$1 split=$2
    local csv="$DATA/$split/$corpus/chunks.csv"
    local out="$HF/${corpus}_${split}"
    if [ ! -f "$csv" ]; then echo "[SKIP] no $csv"; return 0; fi
    if [ -d "$out" ]; then echo "[SKIP] $out exists"; return 0; fi
    echo "=== $corpus/$split ($(($(wc -l < "$csv") - 1)) chunks) -> $out"
    python build_arrow_datasets.py \
        --model_path "$PRETRAINED" \
        --input_csv  "$csv" \
        --save_path  "$out" \
        --language english --task transcribe \
        --num_proc "${NUM_PROC:-8}"
}

for split in train dev; do
    for corpus in wpp iitm combined; do
        build "$corpus" "$split"
    done
done

echo
echo "=============================================================="
echo " verification"
echo "=============================================================="
python - <<'PY'
import os, sys
sys.path.insert(0, ".")
import config_paths as paths
from datasets import load_from_disk

tag = os.environ.get("MODEL_TAG", "whisper-small")
root = os.path.join(paths.HF_DATASETS, tag)
if not os.path.isdir(root):
    print(f"FATAL: {root} missing"); sys.exit(1)

print(f"{'dataset':<18} {'examples':>9} {'mel bins':>9} {'frames':>7} {'label p50':>10}")
bad = 0
for name in sorted(os.listdir(root)):
    d = os.path.join(root, name)
    if not os.path.isdir(d):
        continue
    ds = load_from_disk(d)
    ex = ds[0]
    mel = len(ex["input_features"])
    frames = len(ex["input_features"][0])
    lens = sorted(ds["n_label_tokens"])
    print(f"{name:<18} {len(ds):>9} {mel:>9} {frames:>7} {lens[len(lens)//2]:>10}")
    if frames != 3000:
        print(f"   !! expected 3000 frames (30 s), got {frames}"); bad += 1
    if max(lens) > 448:
        print(f"   !! label longer than the 448-token decoder limit"); bad += 1
sys.exit(1 if bad else 0)
PY

du -sh "$HF"/* 2>/dev/null || true
echo
echo "[DONE] step 3. Next:  bash 05_finetune_all_stages.sh"
