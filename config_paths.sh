#!/bin/bash
# Shell twin of config_paths.py — keep the two in sync.
# Sourced by every *.sh here. Nothing else may hardcode an absolute path.
#
# WHISPER_HOME is derived from this file's own location, so the folder can be
# copied anywhere. Corpus roots take the environment variable if set, otherwise
# the value this workspace was built on.

# Resolve this file's directory even when sourced from another directory.
WHISPER_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA="$WHISPER_HOME/data"
export AUDIO_ROOT="$DATA/audio"
MANIFESTS="$WHISPER_HOME/manifests_kaldi"
PRETRAINED_ROOT="$WHISPER_HOME/pretrained"
HF_ROOT="$WHISPER_HOME/hf_whisper"
MODELS_ROOT="$WHISPER_HOME/models"
DECODE_ROOT="$WHISPER_HOME/decode"
RESULTS_ROOT="$WHISPER_HOME/results"

export WPP_ROOT="${WPP_ROOT:-/path/to/corpora/wpp-2020}"
export IITM_ROOT="${IITM_ROOT:-/path/to/corpora/IITM_English_all_new}"
export MPS_ROOT="${MPS_ROOT:-/path/to/corpora/mps_dataset}"
export KV_ROOT="${KV_ROOT:-/path/to/corpora/KVS_6Grades_EN_MT}"

# Conda. Override CONDA_SH on a machine with a different install.
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-whisper}"

activate_env() {
    if [ -f "$CONDA_SH" ]; then
        # shellcheck disable=SC1090
        source "$CONDA_SH"
        conda activate "$CONDA_ENV"
    else
        echo "[WARN] CONDA_SH not found at $CONDA_SH — assuming the environment"
        echo "       is already active. Set CONDA_SH=<path>/etc/profile.d/conda.sh"
    fi
    export TOKENIZERS_PARALLELISM=false
}
