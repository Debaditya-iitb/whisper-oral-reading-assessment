#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export the Whisper manifests as Kaldi-format data directories.

One source of truth: `build_chunk_manifests.py` / `prepare_mps_test_set.py` decide what
the data *is*; this script only re-expresses it in Kaldi layout so the same
splits can be inspected with Kaldi tooling, diffed against the existing
`KV_English_MTP1/Data/**` dirs, and handed to collaborators.

Layout produced per dataset:

    <out>/<dataset>/full/<split>/     utterance level (the scoring unit)
        wav.scp    <utt_id> <abs path to wav>        (or <rec_id> for iitm)
        text       <utt_id> <verbatim transcript>
        utt2spk / spk2utt / utt2dur
        utt2grade  <utt_id> <grade>          (non-standard, kept for §7 scoring)
        segments   only when utterances are spans of a longer recording (iitm)

    <out>/<dataset>/chunks/<split>/   <=28 s Whisper units
        wav.scp    <rec_id> <abs path>   rec_id is the utterance id
        segments   <chunk_id> <rec_id> <start> <end>
        text       <chunk_id> <transcript for that chunk>
        utt2spk / spk2utt / utt2dur

The chunk dirs are the Kaldi expression of exactly what
`build_arrow_datasets.py` consumes: `wav.scp` + `segments` is the standard
Kaldi idiom for "a piece of a longer recording", which is precisely what a
Whisper chunk is.

MPS test chunks carry no `text` (audio-only chunking — see README §12); the
reference lives at utterance level in `full/test/text`, which is the unit the
WER is computed on.

All files are written sorted by key, which Kaldi requires.
"""

import argparse
import collections
import csv
import os
import shutil
import sys

import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_paths as paths  # noqa: E402


def read_csv(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_sorted(path, pairs):
    """pairs: iterable of (key, value-string). Written sorted, one per line."""
    rows = sorted(pairs, key=lambda kv: kv[0])
    with open(path, "w", encoding="utf-8") as fh:
        for k, v in rows:
            fh.write(f"{k} {v}\n")
    return len(rows)


def write_spk2utt(path, utt2spk):
    by_spk = collections.defaultdict(list)
    for utt, spk in utt2spk:
        by_spk[spk].append(utt)
    with open(path, "w", encoding="utf-8") as fh:
        for spk in sorted(by_spk):
            fh.write(f"{spk} {' '.join(sorted(by_spk[spk]))}\n")
    return len(by_spk)


def wav_duration(path, cache):
    if path not in cache:
        try:
            cache[path] = sf.info(path).duration
        except Exception:
            cache[path] = None
    return cache[path]


def export_full(out_dir, utt_rows, chunk_rows, dur_cache, use_segments):
    """Utterance-level Kaldi dir."""
    os.makedirs(out_dir, exist_ok=True)
    wav_of, span_of = {}, {}
    for c in chunk_rows:
        wav_of.setdefault(c["utt_id"], paths.resolve(c["wav_path"]))
        s, e = float(c["start"]), float(c["end"])
        if c["utt_id"] in span_of:
            s0, e0 = span_of[c["utt_id"]]
            span_of[c["utt_id"]] = (min(s0, s), max(e0, e))
        else:
            span_of[c["utt_id"]] = (s, e)

    utt_rows = [r for r in utt_rows if r["utt_id"] in wav_of]

    text, utt2spk, utt2dur, utt2grade, segments, wav_scp = [], [], [], [], [], {}
    for r in utt_rows:
        utt = r["utt_id"]
        wav = wav_of[utt]
        if use_segments:
            # utterance is a span of a longer recording: key wav.scp by recording
            rec = r.get("story") or utt
            wav_scp[rec] = wav
            s, e = span_of[utt]
            segments.append((utt, f"{rec} {s:.3f} {e:.3f}"))
            utt2dur.append((utt, f"{e - s:.3f}"))
        else:
            wav_scp[utt] = wav
            d = wav_duration(wav, dur_cache)
            utt2dur.append((utt, f"{d:.3f}" if d else "0.000"))
        text.append((utt, r["reference"]))
        utt2spk.append((utt, r["child_key"]))
        utt2grade.append((utt, r.get("grade", "?")))

    n = write_sorted(os.path.join(out_dir, "wav.scp"), wav_scp.items())
    write_sorted(os.path.join(out_dir, "text"), text)
    write_sorted(os.path.join(out_dir, "utt2spk"), utt2spk)
    write_sorted(os.path.join(out_dir, "utt2dur"), utt2dur)
    write_sorted(os.path.join(out_dir, "utt2grade"), utt2grade)
    nspk = write_spk2utt(os.path.join(out_dir, "spk2utt"), utt2spk)
    if segments:
        write_sorted(os.path.join(out_dir, "segments"), segments)
    return {"utts": len(utt_rows), "recordings": n, "speakers": nspk}


def export_chunks(out_dir, chunk_rows, with_text=True):
    """Chunk-level Kaldi dir: wav.scp keyed by utterance, segments per chunk."""
    os.makedirs(out_dir, exist_ok=True)
    wav_scp, segments, text, utt2spk, utt2dur = {}, [], [], [], []
    for c in chunk_rows:
        # `recording` is the physical wav; for KV/WPP/MPS that is the utterance
        # itself, for IITM many segments share one long recording. Keying
        # wav.scp by it keeps the file the size of the audio set, not the
        # segment set (37k lines instead of 104k for IITM).
        rec = c.get("recording") or c["utt_id"]
        wav_scp[rec] = paths.resolve(c["wav_path"])
        segments.append(
            (c["chunk_id"], f"{rec} {float(c['start']):.3f} {float(c['end']):.3f}")
        )
        utt2dur.append((c["chunk_id"], f"{float(c['duration']):.3f}"))
        utt2spk.append((c["chunk_id"], c["child_key"]))
        if with_text:
            text.append((c["chunk_id"], c["text"]))

    write_sorted(os.path.join(out_dir, "wav.scp"), wav_scp.items())
    write_sorted(os.path.join(out_dir, "segments"), segments)
    write_sorted(os.path.join(out_dir, "utt2spk"), utt2spk)
    write_sorted(os.path.join(out_dir, "utt2dur"), utt2dur)
    nspk = write_spk2utt(os.path.join(out_dir, "spk2utt"), utt2spk)
    if with_text:
        write_sorted(os.path.join(out_dir, "text"), text)
    return {"chunks": len(segments), "recordings": len(wav_scp), "speakers": nspk}


def validate(out_dir, expect_text=True):
    """Cheap consistency checks; Kaldi's own validate_data_dir.sh is stricter but
    is not installed here, and these catch the mistakes that actually happen."""
    problems = []

    def keys(name):
        p = os.path.join(out_dir, name)
        if not os.path.exists(p):
            return None
        with open(p, encoding="utf-8") as fh:
            return [ln.split(None, 1)[0] for ln in fh if ln.strip()]

    seg, utt2spk = keys("segments"), keys("utt2spk")
    text, wav = keys("text"), keys("wav.scp")
    ids = seg if seg is not None else wav

    for name, k in (("utt2spk", utt2spk), ("text", text)):
        if k is None:
            if name == "text" and not expect_text:
                continue
            problems.append(f"missing {name}")
            continue
        if k != sorted(k):
            problems.append(f"{name} is not sorted by key")
        if ids is not None and set(k) != set(ids):
            problems.append(
                f"{name} keys differ from {'segments' if seg is not None else 'wav.scp'} "
                f"({len(set(k) ^ set(ids))} mismatched)"
            )
    if seg is not None and wav is not None:
        with open(os.path.join(out_dir, "segments"), encoding="utf-8") as fh:
            recs = {ln.split()[1] for ln in fh if ln.strip()}
        missing = recs - set(wav)
        if missing:
            problems.append(f"{len(missing)} segment recordings absent from wav.scp")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="wpp | iitm | mps | kv")
    ap.add_argument("--manifest_dir", required=True,
                    help="directory holding chunks_<split>.csv + utt_reference_<split>.csv")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--splits", default="train,dev,test",
                    type=lambda v: [x.strip() for x in v.split(",") if x.strip()])
    ap.add_argument("--chunk_text", default="yes", choices=["yes", "no"],
                    help="no = audio-only chunks (MPS test set)")
    ap.add_argument("--use_segments_for_full", action="store_true",
                    help="utterances are spans of a longer recording (iitm)")
    ap.add_argument("--layout", choices=["corpus_first", "split_first"],
                    default="corpus_first",
                    help="corpus_first: <out>/<dataset>/{full,chunks}/<split>. "
                         "split_first: <out>/<split>/<dataset>/ with the "
                         "utterance-level view under reference/ (deliverable layout)")
    args = ap.parse_args()

    dur_cache = {}
    report = []
    for split in args.splits:
        cpath = os.path.join(args.manifest_dir, f"chunks_{split}.csv")
        upath = os.path.join(args.manifest_dir, f"utt_reference_{split}.csv")
        if not os.path.exists(cpath):
            continue
        chunk_rows = read_csv(cpath)
        if not chunk_rows:
            continue
        utt_rows = read_csv(upath) if os.path.exists(upath) else []

        if args.layout == "split_first":
            chunk_dir = os.path.join(args.out_root, split, args.dataset)
            full_dir = os.path.join(chunk_dir, "reference")
        else:
            full_dir = os.path.join(args.out_root, args.dataset, "full", split)
            chunk_dir = os.path.join(args.out_root, args.dataset, "chunks", split)

        s_full = (
            export_full(full_dir, utt_rows, chunk_rows, dur_cache,
                        args.use_segments_for_full)
            if utt_rows else {"utts": 0}
        )
        s_chunk = export_chunks(chunk_dir, chunk_rows, with_text=(args.chunk_text == "yes"))

        p_full = validate(full_dir) if utt_rows else []
        p_chunk = validate(chunk_dir, expect_text=(args.chunk_text == "yes"))

        if args.layout == "split_first":
            # Keep the CSV manifests next to the Kaldi files: the CSVs are what
            # build_arrow_datasets.py / decode_whisper_chunks.py actually read,
            # the Kaldi files are the inspectable/portable expression of them.
            shutil.copyfile(cpath, os.path.join(chunk_dir, "chunks.csv"))
            if os.path.exists(upath):
                shutil.copyfile(upath, os.path.join(full_dir, "utt_reference.csv"))

        report.append(
            f"{args.dataset}/{split}: full={s_full.get('utts')} utts / "
            f"{s_full.get('speakers','-')} spk   chunks={s_chunk['chunks']}"
        )
        for where, probs in ((full_dir, p_full), (chunk_dir, p_chunk)):
            for p in probs:
                report.append(f"   !! {where}: {p}")

    print("\n".join(report) if report else f"[WARN] nothing exported for {args.dataset}")
    if any("!!" in line for line in report):
        sys.exit(1)


if __name__ == "__main__":
    main()
