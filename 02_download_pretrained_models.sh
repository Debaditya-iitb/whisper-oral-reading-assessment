#!/bin/bash
# STEP 2 — download the pretrained Whisper checkpoints into pretrained/.
#
#   bash 02_download_pretrained_models.sh                        # small + medium
#   bash 02_download_pretrained_models.sh whisper-medium         # just one
#
# Must run BEFORE step 3: preparing the data needs the checkpoint's feature
# extractor and tokenizer, so this is not optional ordering.
#
# Everything downstream points at a local directory, never a hub id. On a
# cluster whose compute nodes have no outbound internet, run this on the login
# node; on a single H100 box it does not matter.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config_paths.sh"
activate_env
cd "$WHISPER_HOME"

MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=(whisper-small whisper-medium)

mkdir -p "$PRETRAINED_ROOT"

echo "=============================================================="
echo " STEP 2 — pretrained checkpoints -> $PRETRAINED_ROOT"
echo "=============================================================="
echo "Sizes: small ~1 GB, medium ~3 GB, large-v3 ~6 GB"
echo

for m in "${MODELS[@]}"; do
    dest="$PRETRAINED_ROOT/$m"
    if [ -f "$dest/model.safetensors" ] || [ -f "$dest/pytorch_model.bin" ]; then
        echo "[SKIP] $m already present"
        continue
    fi
    echo "=== openai/$m ==="
    python - "$m" "$PRETRAINED_ROOT" <<'PY'
import sys
from huggingface_hub import snapshot_download
name, dest = sys.argv[1], sys.argv[2]
p = snapshot_download(
    repo_id=f"openai/{name}",
    local_dir=f"{dest}/{name}",
    allow_patterns=[
        "config.json", "generation_config.json", "preprocessor_config.json",
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "added_tokens.json", "normalizer.json", "vocab.json", "merges.txt",
        "model.safetensors",
    ],
)
print("saved:", p)
PY
done

echo
echo "=============================================================="
echo " verification — the mel-bin count must match what step 3 builds"
echo "=============================================================="
python - <<'PY'
import os, sys
sys.path.insert(0, ".")
import config_paths as paths
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration

ok = False
for name in sorted(os.listdir(paths.PRETRAINED)):
    d = os.path.join(paths.PRETRAINED, name)
    if not os.path.isdir(d):
        continue
    try:
        fe = WhisperFeatureExtractor.from_pretrained(d)
        m = WhisperForConditionalGeneration.from_pretrained(d)
        n = sum(p.numel() for p in m.parameters())
        print(f"  OK  {name:<16} {n/1e6:7.1f}M params, {fe.feature_size} mel bins, "
              f"{fe.chunk_length}s window")
        del m
        ok = True
    except Exception as e:
        print(f"  FAIL {name}: {e}")
sys.exit(0 if ok else 1)
PY

echo
echo "[DONE] step 2. Next:  bash 04_build_arrow_datasets.sh"
