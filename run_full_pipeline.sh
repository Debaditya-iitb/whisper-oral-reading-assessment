#!/bin/bash
# Everything, in order, in one command.
#
#   bash run_full_pipeline.sh                          # whisper-small, all four stages
#   MODEL_TAG=whisper-medium bash run_full_pipeline.sh
#   bash run_full_pipeline.sh --from 3                 # resume from step 3
#
# Every step is individually re-runnable and skips work that is already done,
# so an interrupted run is resumed by re-issuing the same command.
#
# Set the corpus roots first (see SERVER_SETUP.md):
#   export WPP_ROOT=... IITM_ROOT=... MPS_ROOT=...

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

FROM=1
if [ "${1:-}" = "--from" ]; then FROM="${2:?--from needs a step number}"; shift 2; fi

MODEL_TAG="${MODEL_TAG:-whisper-small}"
export MODEL_TAG

banner () { echo; echo "##############################################################"; \
            echo "# $*"; echo "##############################################################"; }

t0=$(date +%s)

[ "$FROM" -le 1 ] && { banner "1/5  environment";      bash 01_setup_environment.sh; }
[ "$FROM" -le 2 ] && { banner "2/5  download models";  bash 02_download_pretrained_models.sh "$MODEL_TAG"; }
[ "$FROM" -le 3 ] && { banner "3/5  prepare data";     bash 04_build_arrow_datasets.sh; }
[ "$FROM" -le 4 ] && { banner "4/5  fine-tune";        bash 05_finetune_all_stages.sh; }
[ "$FROM" -le 5 ] && { banner "5/5  evaluate on MPS";  bash 06_evaluate_on_mps_testset.sh; }

banner "finished in $(( ($(date +%s) - t0) / 60 )) min"
echo "results: $HERE/results/$MODEL_TAG/SUMMARY.md"
