#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2 — turn a chunk manifest CSV into a HuggingFace arrow dataset for Whisper.

This is the Whisper analogue of `w2v_finetune/prepare_hf_dataset_deb.py`. Two
things differ from the wav2vec2 version and both matter:

1. wav2vec2 consumes the raw waveform (`input_values`). Whisper consumes an
   80- or 128-bin log-Mel spectrogram (`input_features`) that is ALWAYS padded
   or truncated to exactly 30 s / 3000 frames. Use the feature extractor that
   ships with the checkpoint you are fine-tuning — whisper-large-v3 uses 128
   mel bins while tiny/base/small/medium use 80, and mixing them silently
   produces garbage.
2. wav2vec2 labels are integer phone/word ids produced by `convert_vocab.py`
   against a hand-built `vocab.json`. Whisper labels are BPE token ids from its
   own fixed 50k multilingual vocabulary — there is no vocab to build, no
   `lm_head` to resize, and no `convert_vocab.py` step at all. The text column
   goes in as plain text.

Disk cost: one example is 3000 x n_mels float32 ~= 0.92 MB (small/medium) or
1.5 MB (large-v3). The 5.9k-chunk train split is therefore ~5.5 GB / ~9 GB.
Budget for it, or pass --on_the_fly to store 16-bit audio instead and compute
mel in the collator (slower per step, ~3x less disk).

Usage:
    python build_arrow_datasets.py \
        --model_path openai/whisper-small \
        --input_csv  manifests/chunks_train.csv \
        --save_path  hf_whisper/small/train
"""

import argparse
import os
import sys

import numpy as np
import soundfile as sf
from datasets import Dataset
from transformers import WhisperFeatureExtractor, WhisperTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_paths as paths  # noqa: E402

TARGET_SR = 16000
MAX_LABEL_TOKENS = 448  # Whisper decoder positional limit; longer = hard error


def build(args):
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_path)
    tokenizer = WhisperTokenizer.from_pretrained(
        args.model_path, language=args.language, task=args.task
    )
    print(
        f"[INFO] feature extractor: n_mels={feature_extractor.feature_size} "
        f"chunk={feature_extractor.chunk_length}s"
    )

    def prepare(batch):
        start = float(batch["start"])
        end = float(batch["end"])
        frame_start = int(round(start * TARGET_SR))
        frames = max(1, int(round((end - start) * TARGET_SR)))

        wav_path = paths.resolve(batch["wav_path"])
        speech, sr = sf.read(
            wav_path, start=frame_start, frames=frames, dtype="float32"
        )
        if sr != TARGET_SR:
            raise ValueError(f"Expected {TARGET_SR} Hz, got {sr} for {wav_path}")
        if speech.ndim > 1:
            speech = speech.mean(axis=1)

        if args.on_the_fly:
            batch["audio"] = (speech * 32767.0).astype(np.int16)
        else:
            batch["input_features"] = feature_extractor(
                speech, sampling_rate=TARGET_SR
            ).input_features[0]

        batch["labels"] = tokenizer(batch["text"]).input_ids
        batch["n_label_tokens"] = len(batch["labels"])
        batch["chunk_id"] = batch["chunk_id"]
        batch["utt_id"] = batch["utt_id"]
        batch["grade"] = str(batch["grade"])
        batch["story_level"] = str(batch["story_level"])
        batch["duration"] = float(batch["duration"])
        return batch

    print(f"[INFO] loading {args.input_csv}")
    ds = Dataset.from_csv(args.input_csv)
    print(f"[INFO] {len(ds)} chunks")

    keep = [
        "input_features" if not args.on_the_fly else "audio",
        "labels",
        "n_label_tokens",
        "chunk_id",
        "utt_id",
        "grade",
        "story_level",
        "duration",
    ]
    ds = ds.map(prepare, remove_columns=ds.column_names, num_proc=args.num_proc)

    too_long = ds.filter(lambda x: x["n_label_tokens"] > MAX_LABEL_TOKENS)
    if len(too_long):
        print(
            f"[WARN] dropping {len(too_long)} chunks whose label exceeds "
            f"{MAX_LABEL_TOKENS} tokens (decoder limit)"
        )
        ds = ds.filter(lambda x: x["n_label_tokens"] <= MAX_LABEL_TOKENS)

    lens = ds["n_label_tokens"]
    print(
        f"[INFO] label tokens: min {min(lens)} max {max(lens)} "
        f"mean {sum(lens)/len(lens):.1f}"
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
    ds.save_to_disk(args.save_path)
    print(f"[DONE] saved {len(ds)} examples -> {args.save_path}  (columns: {keep})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True,
                    help="whisper checkpoint dir or hub id — must be the SAME one "
                         "you fine-tune from (mel bin count has to match)")
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--save_path", required=True)
    ap.add_argument("--language", default="english")
    ap.add_argument("--task", default="transcribe")
    ap.add_argument("--num_proc", type=int, default=8)
    ap.add_argument("--on_the_fly", action="store_true",
                    help="store int16 audio instead of mel; compute mel in the "
                         "collator at train time (less disk, slower steps)")
    build(ap.parse_args())


if __name__ == "__main__":
    main()
