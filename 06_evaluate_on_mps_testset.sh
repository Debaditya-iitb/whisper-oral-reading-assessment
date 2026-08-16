#!/bin/bash
# STEP 5 — decode every model on the MPS test set and score per grade.
#
#   bash 06_evaluate_on_mps_testset.sh
#   MODEL_TAG=whisper-medium bash 06_evaluate_on_mps_testset.sh
#
# Decodes the untouched pretrained checkpoint as well. That is not optional:
# a fine-tuned WER on its own is uninterpretable — the result is the
# before/after pair on the same manifest.
#
# Models that do not exist yet are skipped, so this is useful mid-run.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config_experiment.sh"
activate_env
cd "$WHISPER_HOME"

# Same pinning convention as s4 so decoding lands on the GPU you expect.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID:-0}}"

MPS_CHUNKS=$DATA/test/mps/chunks.csv
MPS_REF=$DATA/test/mps/reference/utt_reference.csv
for f in "$MPS_CHUNKS" "$MPS_REF"; do
    [ -f "$f" ] || { echo "[FATAL] missing $f — run: bash 04_build_arrow_datasets.sh"; exit 1; }
done

echo "=============================================================="
echo " STEP 5 — evaluation on MPS   model=$MODEL_TAG"
echo " 1,600 utterances / 3,484 chunks / 19.1 h, grades 3-5"
echo "=============================================================="

decode () {   # decode <name> <model_dir>
    local name=$1 model=$2
    [ -d "$model" ] || { echo "[SKIP] $name: not trained yet"; return 0; }
    if [ -f "$DEC/$name/hyp_utterances.csv" ]; then
        echo "[SKIP] $name already decoded"; return 0
    fi
    echo "=== decoding $name ==="
    python decode_whisper_chunks.py \
        --model_dir "$model" --manifest "$MPS_CHUNKS" --out_dir "$DEC/$name" \
        --batch_size "${DECODE_BS:-32}" --num_beams 1 --max_repeat 3
}

decode pretrained "$PRETRAINED"
decode stageW     "$STAGE_W"
decode stageA     "$STAGE_A"
decode stageB     "$STAGE_B"
decode stageC     "$STAGE_C"

[ -f "$DEC/pretrained/hyp_utterances.csv" ] || {
    echo "[FATAL] the pretrained baseline failed to decode; nothing to compare against"
    exit 1
}

# devanagari=drop is the headline number (those tokens are non-words the child
# uttered). keep and placeholder are also written so the choice is visible.
for name in stageW stageA stageB stageC; do
    hyp=$DEC/$name/hyp_utterances.csv
    [ -f "$hyp" ] || continue
    for dv in drop placeholder keep; do
        python score_wer_by_grade.py \
            --reference "$MPS_REF" --hyp "$hyp" \
            --baseline_hyp "$DEC/pretrained/hyp_utterances.csv" \
            --grade_key grade --devanagari "$dv" --norm minimal \
            --out_dir "$RES/mps_${name}_${dv}" > /dev/null
    done
    echo
    echo "=== $name (devanagari=drop) ==="
    sed -n '/grade/,$p' "$RES/mps_${name}_drop/report.txt" | head -12
done

echo
python summarize_stage_results.py --results_root "$RES" --out "$RES/SUMMARY.md" || true

echo
echo "=============================================================="
echo " repetition-loop check — read this before believing any WER"
echo "=============================================================="
for d in "$DEC"/*/decode_meta.json; do
    [ -f "$d" ] || continue
    python -c "
import json,sys
m=json.load(open('$d'))
print(f\"  {m['model_dir'].split('/')[-1]:<28} looped chunks {m.get('looped_chunks',0):>4} / {m['chunks']}  RTF {m['rtf']:.4f}\")"
done

echo
echo "[DONE] results: $RES/SUMMARY.md"
