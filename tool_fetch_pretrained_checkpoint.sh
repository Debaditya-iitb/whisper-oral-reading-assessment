#!/bin/bash
# Run this ON THE LOGIN NODE, not under sbatch.
#
# Compute nodes on this cluster are not guaranteed outbound internet, and a
# training job that dies 40 minutes in because `from_pretrained("openai/...")`
# could not reach huggingface.co wastes a whole DGX allocation. Snapshot the
# checkpoints into a local directory first and point every script at the local
# path — the same convention the wav2vec2 scripts already use (BASE_MODEL_DIR
# is always a directory, never a hub id).
#
#   bash tool_fetch_pretrained_checkpoint.sh whisper-small whisper-medium
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config_paths.sh"
DEST="$PRETRAINED_ROOT"
mkdir -p "$DEST"

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
    MODELS=(whisper-small)
fi

activate_env

for m in "${MODELS[@]}"; do
    echo "=== fetching openai/$m -> $DEST/$m ==="
    python - "$m" "$DEST" <<'PY'
import sys
from huggingface_hub import snapshot_download
name, dest = sys.argv[1], sys.argv[2]
p = snapshot_download(
    repo_id=f"openai/{name}",
    local_dir=f"{dest}/{name}",
    allow_patterns=[
        "*.json", "*.txt", "*.model",
        "model.safetensors", "pytorch_model.bin",
        "preprocessor_config.json", "tokenizer*", "vocab*", "merges*",
        "normalizer.json", "added_tokens.json", "special_tokens_map.json",
        "generation_config.json", "config.json",
    ],
)
print("saved:", p)
PY
done

echo "[DONE] local checkpoints under $DEST"
ls -1 "$DEST"
