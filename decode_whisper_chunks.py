#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4 — decode a chunk manifest with a (fine-tuned or pretrained) Whisper model.

Reads a `chunks_*.csv` produced by `build_chunk_manifests.py`, decodes every chunk
independently, then re-joins the chunk hypotheses in time order into one
hypothesis per utterance. The utterance-level output is what `score_wer_by_grade.py`
compares against `utt_reference_*.csv` — so the WER you report is computed over
exactly the same utterances, with exactly the same references, as your
wav2vec2 word-level results. Chunking is an implementation detail of the
encoder window, not a change to the evaluation unit.

Decoding defaults are chosen for miscue fidelity, not for the prettiest WER:

    --num_beams 1        greedy. Beam search leans harder on Whisper's internal
                         LM and is measurably more likely to "repair" a
                         misread word into the word the child *should* have
                         said, which destroys the thing you are measuring.
    condition_on_prev_tokens = False
                         each chunk decoded independently; prevents a bad chunk
                         from dragging the rest of the utterance into a
                         hallucinated loop.
    --temperature 0.0    no sampling, no temperature fallback.

Run the SAME manifest through the pretrained checkpoint (`--model_dir
openai/whisper-small`) as well as your fine-tuned one — the before/after pair is
the actual result, a single fine-tuned number means nothing on its own.

Usage:
    python decode_whisper_chunks.py \
        --model_dir models/whisper-small-kv-en \
        --manifest  manifests/chunks_test.csv \
        --out_dir   decode/whisper-small-kv-en_test
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_paths as paths  # noqa: E402

TARGET_SR = 16000


def load_manifest(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def collapse_repeats(text, max_repeat=3, max_n=5):
    """Collapse runaway n-gram loops, and report whether one was found.

    Whisper's best-known failure on disfluent speech is decoding into a loop —
    `happy, happy, happy, …` until it hits max_new_tokens. Observed on this
    exact data: whisper-tiny zero-shot on an MPS grade-4 utterance produced 60+
    repetitions of one word in a single chunk, giving that utterance a 215% WER
    driven almost entirely by insertions.

    `no_repeat_ngram_size` is the usual fix and it is the WRONG one here:
    repetition is itself a reading miscue, and children genuinely do repeat
    words. So instead of forbidding repeats during generation, collapse only
    implausibly long runs afterwards (a real reader does not say a word 60
    times) and flag the chunk so you can inspect or exclude it. Nothing is
    hidden — the flag is written to the output CSV and counted in the summary.
    """
    words = text.split()
    if not words:
        return text, False
    looped = False
    for n in range(1, max_n + 1):
        out, i = [], 0
        while i < len(words):
            gram = words[i : i + n]
            if len(gram) < n:
                out.extend(words[i:])
                break
            reps = 1
            j = i + n
            while words[j : j + n] == gram:
                reps += 1
                j += n
            if reps > max_repeat:
                looped = True
                out.extend(gram * max_repeat)
            else:
                out.extend(words[i:j])
            i = j
        words = out
    return " ".join(words), looped


def read_slice(row):
    start = float(row["start"])
    end = float(row["end"])
    frame_start = int(round(start * TARGET_SR))
    frames = max(1, int(round((end - start) * TARGET_SR)))
    wav_path = paths.resolve(row["wav_path"])
    speech, sr = sf.read(wav_path, start=frame_start, frames=frames, dtype="float32")
    if sr != TARGET_SR:
        raise ValueError(f"Expected {TARGET_SR} Hz, got {sr} for {wav_path}")
    if speech.ndim > 1:
        speech = speech.mean(axis=1)
    return speech


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--language", default="english")
    ap.add_argument("--task", default="transcribe")
    ap.add_argument("--max_new_tokens", type=int, default=225)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0, help="decode only N chunks (smoke test)")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--max_repeat", type=int, default=3,
                    help="collapse any n-gram repeated more than this many times "
                         "in a chunk hypothesis (runaway-loop guard). 0 disables.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[WARN] no CUDA device — decoding on CPU. The full MPS test set "
              "will take many hours. Check `nvidia-smi` and your torch build.")
    else:
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
    torch_dtype = getattr(torch, args.dtype) if device == "cuda" else torch.float32

    processor = WhisperProcessor.from_pretrained(
        args.model_dir, language=args.language, task=args.task
    )
    try:
        model = WhisperForConditionalGeneration.from_pretrained(
            args.model_dir, torch_dtype=torch_dtype,
            attn_implementation=os.environ.get("ATTN_IMPL", "sdpa"),
        ).to(device)
    except (ValueError, TypeError, ImportError):
        model = WhisperForConditionalGeneration.from_pretrained(
            args.model_dir, torch_dtype=torch_dtype
        ).to(device)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    model.eval()
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.generation_config.language = args.language
    model.generation_config.task = args.task
    if hasattr(model.generation_config, "condition_on_prev_tokens"):
        model.generation_config.condition_on_prev_tokens = False

    rows = load_manifest(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    # Sort by duration so each batch pads to a similar length -> less wasted compute.
    order = sorted(range(len(rows)), key=lambda i: float(rows[i]["duration"]))

    hyps = {}
    looped_chunks = set()
    t0 = time.time()
    audio_sec = 0.0
    for bi in range(0, len(order), args.batch_size):
        idxs = order[bi : bi + args.batch_size]
        waves = [read_slice(rows[i]) for i in idxs]
        audio_sec += sum(len(w) for w in waves) / TARGET_SR

        inputs = processor.feature_extractor(
            waves, sampling_rate=TARGET_SR, return_tensors="pt"
        )
        feats = inputs.input_features.to(device, dtype=torch_dtype)

        with torch.no_grad():
            gen = model.generate(
                feats,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                do_sample=False,
                temperature=args.temperature if args.temperature > 0 else None,
            )
        texts = processor.tokenizer.batch_decode(gen, skip_special_tokens=True)
        for i, txt in zip(idxs, texts):
            txt = txt.strip()
            if args.max_repeat > 0:
                txt, looped = collapse_repeats(txt, args.max_repeat)
                if looped:
                    looped_chunks.add(rows[i]["chunk_id"])
            hyps[rows[i]["chunk_id"]] = txt

        done = bi + len(idxs)
        if bi % (args.batch_size * 20) == 0 or done == len(order):
            el = time.time() - t0
            print(
                f"[{done}/{len(order)}] {el:.0f}s elapsed, "
                f"RTF {el/max(audio_sec,1e-6):.3f}",
                flush=True,
            )

    chunk_csv = os.path.join(args.out_dir, "hyp_chunks.csv")
    with open(chunk_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["chunk_id", "utt_id", "start", "end", "grade",
                        "story_level", "looped", "hyp"],
        )
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "chunk_id": r["chunk_id"],
                    "utt_id": r["utt_id"],
                    "start": r["start"],
                    "end": r["end"],
                    "grade": r["grade"],
                    "story_level": r["story_level"],
                    "looped": int(r["chunk_id"] in looped_chunks),
                    "hyp": hyps.get(r["chunk_id"], ""),
                }
            )

    # Re-join chunks into one hypothesis per utterance, in start-time order.
    by_utt = {}
    for r in rows:
        by_utt.setdefault(r["utt_id"], []).append(r)
    utt_csv = os.path.join(args.out_dir, "hyp_utterances.csv")
    with open(utt_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["utt_id", "grade", "story_level", "round", "story",
                        "looped", "hyp"],
        )
        w.writeheader()
        for utt_id, items in by_utt.items():
            items.sort(key=lambda r: float(r["start"]))
            joined = " ".join(
                hyps.get(r["chunk_id"], "").strip()
                for r in items
                if hyps.get(r["chunk_id"], "").strip()
            )
            w.writerow(
                {
                    "utt_id": utt_id,
                    "grade": items[0]["grade"],
                    "story_level": items[0]["story_level"],
                    "round": items[0].get("round", ""),
                    "story": items[0].get("story", ""),
                    "looped": int(any(r["chunk_id"] in looped_chunks for r in items)),
                    "hyp": joined,
                }
            )

    meta = {
        "model_dir": args.model_dir,
        "manifest": args.manifest,
        "chunks": len(rows),
        "utterances": len(by_utt),
        "audio_hours": audio_sec / 3600,
        "wall_seconds": time.time() - t0,
        "rtf": (time.time() - t0) / max(audio_sec, 1e-6),
        "looped_chunks": len(looped_chunks),
        "looped_utterances": len(
            {r["utt_id"] for r in rows if r["chunk_id"] in looped_chunks}
        ),
        "max_repeat": args.max_repeat,
        "num_beams": args.num_beams,
        "language": args.language,
        "dtype": args.dtype,
        "device": device,
    }
    with open(os.path.join(args.out_dir, "decode_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(json.dumps(meta, indent=2))
    print(f"[DONE] {chunk_csv}\n[DONE] {utt_csv}")


if __name__ == "__main__":
    sys.exit(main())
