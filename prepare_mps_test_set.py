#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a Whisper test manifest from the MPS dataset (DAP-Lab, Interspeech 2024).

    git clone https://github.com/DAP-Lab/mps_dataset.git
    python prepare_mps_test_set.py --mps_root /path/to/mps_dataset \\
        --out_dir manifests_mps

MPS is a near-perfect drop-in for this pipeline: 1,600 utterances / 1,110
speakers / 19 h, grades 3-5, L2 English read-aloud, 16 kHz mono, and it uses
the *identical* transcription conventions to KV and WPP — same tag set
(SIL/BR/ON/FP/IR/MB/WH/HS, all already in `text_norm.LABEL_TAGS`) and the same
Devanagari-for-invalid-English-words rule (3.43% of tokens, 89.4% of
utterances). `manualTranscript` is verbatim, so it is directly comparable with
the KV/WPP references — no convention mismatch to correct for.

WHY NO FORCED ALIGNMENT IS NEEDED HERE EITHER
---------------------------------------------
`data.json` has no timestamps, and MPS utterances are 35-60 s, so they exceed
Whisper's 30 s encoder window. But this is a *test* set, and testing has a
weaker requirement than training:

    training needs (audio, text) pairs that are aligned AND <= 30 s
    decoding needs only audio <= 30 s -- the reference stays whole

So the audio is cut into <= 28 s pieces, each piece is decoded independently,
and the hypotheses are re-joined into one hypothesis per utterance before
scoring against the full `manualTranscript`. The text is never cut, so it never
needs to be aligned. Cuts are placed at the quietest frame inside a search
window, so they land in pauses rather than mid-word.

Outputs:
    chunks_test.csv          audio-only chunk manifest for decode_whisper_chunks.py
    utt_reference_test.csv   utterance-level references for score_wer_by_grade.py
    mps_stats.txt
"""

import argparse
import collections
import csv
import json
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from text_normalisation import strip_tags  # noqa: E402
import config_paths as paths  # noqa: E402

TARGET_SR = 16000
MAX_CHUNK_SEC = 28.0
MIN_CHUNK_SEC = 0.6
# Cuts are searched in [MAX-SEARCH, MAX]; a wider window finds quieter cut
# points but makes chunks more variable in length.
SEARCH_SEC = 10.0
FRAME_SEC = 0.025
HOP_SEC = 0.010


def energy_cut_points(speech, max_sec, search_sec):
    """Greedy: walk forward in max_sec strides, and cut at the lowest-energy
    frame within the last `search_sec` of each stride. Returns sample indices."""
    n = len(speech)
    if n <= int(max_sec * TARGET_SR):
        return [0, n]

    hop = int(HOP_SEC * TARGET_SR)
    win = int(FRAME_SEC * TARGET_SR)
    n_frames = max(1, 1 + (n - win) // hop)
    rms = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        seg = speech[i * hop : i * hop + win]
        rms[i] = float(np.sqrt(np.mean(seg * seg)) if seg.size else 0.0)

    cuts = [0]
    while n - cuts[-1] > int(max_sec * TARGET_SR):
        start = cuts[-1]
        hard_end = start + int(max_sec * TARGET_SR)
        soft_start = max(start + int(MIN_CHUNK_SEC * TARGET_SR),
                         hard_end - int(search_sec * TARGET_SR))
        f0, f1 = soft_start // hop, min(hard_end // hop, n_frames)
        if f1 <= f0:
            cuts.append(hard_end)
            continue
        best = int(np.argmin(rms[f0:f1])) + f0
        cut = min(best * hop + win // 2, hard_end)
        if cut <= start + int(MIN_CHUNK_SEC * TARGET_SR):
            cut = hard_end
        cuts.append(cut)
    cuts.append(n)
    return cuts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mps_root", default=paths.ROOTS["MPS_ROOT"],
                    help="cloned mps_dataset repo (default: $MPS_ROOT)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--data_json", default="", help="defaults to <mps_root>/data.json")
    ap.add_argument("--max_chunk_sec", type=float, default=MAX_CHUNK_SEC)
    ap.add_argument("--search_sec", type=float, default=SEARCH_SEC)
    args = ap.parse_args()

    data_json = args.data_json or os.path.join(args.mps_root, "data.json")
    if not os.path.exists(data_json):
        sys.exit(f"[FATAL] not found: {data_json}")
    records = json.load(open(data_json, encoding="utf-8"))
    print(f"[INFO] {len(records)} MPS records")

    os.makedirs(args.out_dir, exist_ok=True)
    stats = collections.Counter()
    chunk_rows, utt_rows = [], []

    for rec in records:
        utt_id = rec["audioID"]
        meta = rec.get("metaData", {})
        wav_path = os.path.join(args.mps_root, rec["audioPath"])
        if not os.path.exists(wav_path):
            stats["wav_missing"] += 1
            continue

        reference = strip_tags(rec.get("manualTranscript", "")).strip()
        if not reference:
            stats["empty_reference"] += 1
            continue

        try:
            speech, sr = sf.read(wav_path, dtype="float32")
        except Exception:
            stats["wav_unreadable"] += 1
            continue
        if sr != TARGET_SR:
            stats["wav_not_16k"] += 1
            continue
        if speech.ndim > 1:
            speech = speech.mean(axis=1)

        cuts = energy_cut_points(speech, args.max_chunk_sec, args.search_sec)
        kept = 0
        for i in range(len(cuts) - 1):
            start = cuts[i] / TARGET_SR
            end = cuts[i + 1] / TARGET_SR
            if end - start < MIN_CHUNK_SEC:
                stats["chunk_too_short"] += 1
                continue
            if end - start > args.max_chunk_sec + 0.05:
                stats["chunk_oversized"] += 1
            chunk_rows.append(
                {
                    "chunk_id": f"{utt_id}__c{i:03d}",
                    "utt_id": utt_id,
                    "recording": utt_id,
                    "wav_path": paths.templatize(wav_path),
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "duration": round(end - start, 3),
                    "text": "",  # test set: audio-only chunking, text never cut
                    "grade": str(meta.get("grade", "?")),
                    "corpus": "mps",
                    "round": "mps",
                    "story": meta.get("storyID", "?"),
                    "story_level": "?",  # MPS story ids do not encode a level
                    "child_key": f"mps:{meta.get('speakerID', utt_id)}",
                    "para": str(meta.get("paragraphID", "")),
                }
            )
            kept += 1
            stats["chunk_sec"] += end - start

        if kept == 0:
            stats["utt_no_usable_chunk"] += 1
            continue

        utt_rows.append(
            {
                "utt_id": utt_id,
                "grade": str(meta.get("grade", "?")),
                "corpus": "mps",
                "round": "mps",
                "story": meta.get("storyID", "?"),
                "story_level": "?",
                "child_key": f"mps:{meta.get('speakerID', utt_id)}",
                "reference": reference,
                "gender": meta.get("gender", ""),
            }
        )
        stats["utts"] += 1

    chunk_fields = [
        "chunk_id", "utt_id", "recording", "wav_path", "start", "end",
        "duration", "text", "grade", "corpus", "round", "story", "story_level",
        "child_key", "para",
    ]
    utt_fields = [
        "utt_id", "grade", "corpus", "round", "story", "story_level",
        "child_key", "reference", "gender",
    ]
    for path, rows, fields in (
        (os.path.join(args.out_dir, "chunks_test.csv"), chunk_rows, chunk_fields),
        (os.path.join(args.out_dir, "utt_reference_test.csv"), utt_rows, utt_fields),
    ):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    per_grade = collections.Counter()
    grade_sec = collections.Counter()
    dur_by_utt = collections.Counter()
    for c in chunk_rows:
        dur_by_utt[c["utt_id"]] += c["duration"]
    for u in utt_rows:
        per_grade[u["grade"]] += 1
        grade_sec[u["grade"]] += dur_by_utt[u["utt_id"]]

    out = ["MPS TEST-SET REPORT", f"data.json: {data_json}",
           f"max_chunk_sec = {args.max_chunk_sec}  search_sec = {args.search_sec}", ""]
    for k in sorted(stats):
        out.append(f"{k:>22}: {stats[k]/3600:.2f} h" if k == "chunk_sec"
                   else f"{k:>22}: {stats[k]}")
    out.append("")
    out.append(f"{'grade':>7} {'utts':>7} {'chunks':>8} {'hours':>8}")
    ch_by_grade = collections.Counter(c["grade"] for c in chunk_rows)
    for g in sorted(per_grade):
        out.append(f"{g:>7} {per_grade[g]:>7} {ch_by_grade[g]:>8} "
                   f"{grade_sec[g]/3600:>8.2f}")
    out.append(f"{'ALL':>7} {len(utt_rows):>7} {len(chunk_rows):>8} "
               f"{sum(grade_sec.values())/3600:>8.2f}")
    out.append(f"speakers: {len({u['child_key'] for u in utt_rows})}")

    report = "\n".join(out)
    print(report)
    with open(os.path.join(args.out_dir, "mps_stats.txt"), "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print(f"\n[DONE] {args.out_dir}/chunks_test.csv, utt_reference_test.csv")


if __name__ == "__main__":
    main()
