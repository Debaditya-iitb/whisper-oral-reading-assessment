#!/bin/bash
# Build every train / dev / test set into ONE folder: whisper_finetune/data/
#
#   bash 03_build_chunked_datasets.sh
#
# Layout produced (split first, so "the train set" / "the dev set" / "the test
# set" are each one directory):
#
#   data/
#   ├── train/{wpp,iitm,combined}/
#   ├── dev/{wpp,iitm,combined}/
#   └── test/mps/
#
# Every leaf contains BOTH views of the same data:
#
#   <leaf>/                      CHUNK level  — the <=28 s units Whisper trains
#                                on / decodes. wav.scp segments text utt2spk
#                                spk2utt utt2dur, plus chunks.csv
#   <leaf>/reference/            UTTERANCE level — the unit WER is reported on.
#                                wav.scp text utt2spk spk2utt utt2dur utt2grade,
#                                plus utt_reference.csv
#
# The CSVs are what the Python scripts read; the Kaldi files are the portable,
# inspectable expression of exactly the same rows.
#
# IITM defaults: long-form read only (--iitm_keep long_read) with segments
# packed to <=28 s. Override with IITM_KEEP=long_read,short_read etc.

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config_paths.sh"
HERE=$WHISPER_HOME
MAN=$MANIFESTS
IITM_KEEP=${IITM_KEEP:-long_read}
IITM_CAP=${IITM_CAP:-iitm=60}

cd "$HERE"
mkdir -p "$DATA" "$MAN"

exp () {   # exp <dataset> <splits> [extra args...]
    local ds=$1 splits=$2; shift 2
    python export_kaldi_dirs.py --dataset "$ds" --manifest_dir "$MAN/$ds" \
        --out_root "$DATA" --splits "$splits" --layout split_first "$@"
}

echo "### WPP — children, grades 6-10"
python build_chunk_manifests.py --out_dir "$MAN/wpp" --corpora wpp --split_mode all_train
exp wpp train,dev

echo
echo "### IITM — adult, keep=$IITM_KEEP, segments packed to <=28 s"
python build_chunk_manifests.py --out_dir "$MAN/iitm" --corpora iitm \
    --split_mode all_train --iitm_keep "$IITM_KEEP"
exp iitm train,dev --use_segments_for_full

echo
echo "### COMBINED — WPP + IITM (cap $IITM_CAP)"
python build_chunk_manifests.py --out_dir "$MAN/combined" --corpora wpp,iitm \
    --split_mode all_train --iitm_keep "$IITM_KEEP" --cap_hours "$IITM_CAP"
exp combined train,dev

echo
echo "### MPS — test set, grades 3-5"
if [ -d "$MPS_ROOT" ]; then
    python prepare_mps_test_set.py --mps_root "$MPS_ROOT" --out_dir "$MAN/mps"
    exp mps test --chunk_text no
else
    echo "[SKIP] MPS not at \$MPS_ROOT ($MPS_ROOT)"
    echo "       git clone https://github.com/DAP-Lab/mps_dataset.git \"$MPS_ROOT\""
fi

# Rebuilding regenerates manifests pointing back at the EXTERNAL corpus roots.
# If the audio has already been brought inside data/audio, re-point them, or
# the folder silently stops being self-contained.
if [ -d "$AUDIO_ROOT" ] || [ "${LOCALIZE:-0}" = "1" ]; then
    echo
    echo "### Localizing audio into data/audio (folder stays self-contained)"
    python copy_audio_to_local_disk.py --mode "${LOCALIZE_MODE:-hardlink}"
fi

echo
echo "=============================================================="
printf '%-26s %9s %9s %9s\n' "set" "chunks" "utts" "speakers"
echo "=============================================================="
for split in train dev test; do
    for leaf in "$DATA/$split"/*/; do
        [ -d "$leaf" ] || continue
        c=$(wc -l < "$leaf/segments" 2>/dev/null || echo 0)
        u=$(wc -l < "$leaf/reference/text" 2>/dev/null || echo 0)
        s=$(wc -l < "$leaf/spk2utt" 2>/dev/null || echo 0)
        printf '%-26s %9s %9s %9s\n' "${leaf#$DATA/}" "$c" "$u" "$s"
    done
done
echo
echo "[DONE] $DATA"
