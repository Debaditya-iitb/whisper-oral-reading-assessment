#!/bin/bash
#SBATCH --job-name=wh_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=11:59:59
#SBATCH --output=job.%j.out
#SBATCH --error=job.%j.err
#SBATCH --partition=a40
#SBATCH --qos=a40

# Decode every model on the MPS test set and score per grade, each fine-tuned
# system compared against the untouched pretrained checkpoint.
#
# Only models that exist are decoded, so this is useful before the whole chain
# has finished.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config_experiment.sh"
activate_env
cd "$HERE"

MPS_CHUNKS=$DATA/test/mps/chunks.csv
MPS_REF=$DATA/test/mps/reference/utt_reference.csv
for f in "$MPS_CHUNKS" "$MPS_REF"; do
    [ -f "$f" ] || { echo "[FATAL] missing $f — run 03_build_chunked_datasets.sh"; exit 1; }
done

decode () {   # decode <name> <model_dir>
    local name=$1 model=$2
    [ -d "$model" ] || { echo "[SKIP] $name: $model not found"; return 0; }
    if [ -f "$DEC/${name}/hyp_utterances.csv" ]; then
        echo "[SKIP] $name already decoded"
        return 0
    fi
    echo "=== decoding $name ==="
    python decode_whisper_chunks.py \
        --model_dir "$model" \
        --manifest  "$MPS_CHUNKS" \
        --out_dir   "$DEC/${name}" \
        --batch_size 16 --num_beams 1 --max_repeat 3
}

decode pretrained "$PRETRAINED"
decode stageA     "$STAGE_A"
decode stageB     "$STAGE_B"
decode stageW     "$STAGE_W"
decode stageC     "$STAGE_C"

# Devanagari tokens are non-words the child uttered -> dropped from both sides
# for the headline WER. `placeholder` is also reported: it still requires the
# model to emit something non-English there, which `drop` does not.
for name in stageA stageB stageW stageC; do
    hyp=$DEC/$name/hyp_utterances.csv
    [ -f "$hyp" ] || continue
    for dv in drop placeholder keep; do
        python score_wer_by_grade.py \
            --reference "$MPS_REF" \
            --hyp "$hyp" \
            --baseline_hyp "$DEC/pretrained/hyp_utterances.csv" \
            --grade_key grade --devanagari "$dv" --norm minimal \
            --out_dir "$RES/mps_${name}_${dv}" > /dev/null
    done
    echo "=== $name (devanagari=drop) ==="
    sed -n '/grade/,$p' "$RES/mps_${name}_drop/report.txt" | head -14
done

python summarize_stage_results.py --results_root "$RES" --out "$RES/SUMMARY.md" || true
echo "[DONE] reports under $RES"
