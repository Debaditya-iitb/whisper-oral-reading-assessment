#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1 — build Whisper-ready manifests from the KV 6-grades label tracks.

Why this step exists at all
---------------------------
The wav2vec2 pipeline feeds whole utterances (mean 43 s, max ~62 s, 94% of them
longer than 30 s) straight into a CTC model, which is fine because wav2vec2 is
fully convolutional+attentive over an arbitrary-length waveform. Whisper is NOT:
its encoder has a *fixed* 30-second receptive field (3000 mel frames). Anything
past 30 s is silently truncated by the feature extractor, so training on the
full utterance would pair 30 s of audio with a 43 s transcript and teach the
model to hallucinate the missing tail. Every serious Whisper fine-tune on
long-form data has to segment first.

`KVS_6Grades_EN_MT/*/label_tracks/*_label_track.txt` already contains
`start \t end \t verbatim_text` lines at breath-group granularity (2-6 s each),
which is exactly the alignment needed. This script greedily merges consecutive
lines into <= MAX_CHUNK_SEC windows without ever splitting a line, so audio and
text stay aligned by construction — no forced aligner needed.

Outputs (into --out_dir):
    chunks_train.csv / chunks_dev.csv / chunks_test.csv
        chunk_id, utt_id, wav_path, start, end, duration, text,
        grade, round, story, child_key, para
    utt_reference_{train,dev,test}.csv
        utt_id, grade, round, story, child_key, reference   <- utterance-level
        ground truth, used at scoring time so that Whisper's numbers are
        directly comparable with the utterance-level wav2vec2 PER/WER.
    manifest_stats.txt

Usage:
    python build_chunk_manifests.py --out_dir manifests --split_mode established
"""

import argparse
import collections
import csv
import glob
import os
import random
import re
import sys

import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from text_normalisation import is_tag_only, strip_tags  # noqa: E402
import config_paths as paths  # noqa: E402

ESTABLISHED_TEST_TEXT = os.path.join(
    os.path.dirname(paths.ROOTS["KV_ROOT"].rstrip("/")),
    "KV_English_MTP1/Data/Baseline_Test_set/Combined_data_B24_B25/text",
)

# 28 s not 30 s: leaves headroom so that a chunk whose last line ends slightly
# late, or whose audio file is a few frames longer than the label track says,
# still fits inside the encoder window instead of being truncated.
MAX_CHUNK_SEC = 28.0
MIN_CHUNK_SEC = 0.6
DROP_OVERSIZED = True

# ---------------------------------------------------------------------------
# Corpus registry. Any corpus laid out as <root>/<subdir>/{audios,label_tracks}
# with `start \t end \t verbatim_text` label tracks plugs in here — that is the
# only thing this script needs, and both KV and WPP already have it.
# ---------------------------------------------------------------------------

KV_UTT_RE = re.compile(
    r"^(?P<ini>[a-z]+)_(?P<grade>\d+)-(?P<sec>[a-z0-9]+)-(?P<roll>\d+)_"
    r"(?P<school>.+?)_(?P<date>\d{8}-\d{6}-\d+)_"
    r"(?P<story>EN-[A-Z0-9-]+)_(?P<para>\d+)$"
)

# WPP ids differ from KV: no roll number, school and city are separate fields,
# and the story field is either `EN-IITB-009` / `EN-WPP-LLF-018` or a bare
# `s010` / `m010` (844 utterances, all from CSV_Dharavi).
WPP_UTT_RE = re.compile(
    r"^(?P<ini>[0-9A-Za-z]+)_(?P<grade>\d+)-(?P<sec>[A-Za-z0-9]+)_"
    r"(?P<school>.+?)_(?P<city>[A-Za-z-]+)_(?P<date>\d{8}-\d{6}-\d+)_"
    r"(?P<story>[A-Za-z0-9-]+)_(?P<para>\d+)$"
)


def _story_level_from_number(story):
    """KV story numbering encodes passage difficulty: EN-OL-RC-2xx = grade 3,
    3xx = grade 4, ... 7xx = grade 8."""
    num = story.rsplit("-", 1)[-1]
    return str(int(num[0]) + 1) if num[:1].isdigit() else "?"


def parse_kv_id(utt_id):
    m = KV_UTT_RE.match(utt_id)
    if not m:
        return None
    d = m.groupdict()
    # child_key deliberately excludes the date: the same child is recorded in
    # Baseline/Midline/Endline of the same academic year, and 557 children in
    # this corpus appear in more than one round. Keying on the date would let
    # the same child land in both train and test.
    d["child_key"] = f"kv:{d['ini']}_{d['grade']}-{d['sec']}-{d['roll']}_{d['school']}"
    d["utt_id"] = utt_id
    # Story difficulty level. This is NOT always the child's own grade: 200
    # utterances in the existing B24+B25 test set are grade-5 children reading
    # the grade-6 passage EN-OL-RC-500, and those utterances are physically
    # duplicated into both `grade_5/` and `grade_6/` of that Kaldi dir.
    # Carrying both keys means the per-grade report can say which definition it
    # is using instead of silently double counting.
    d["story_level"] = _story_level_from_number(d["story"])
    return d


def parse_wpp_id(utt_id):
    m = WPP_UTT_RE.match(utt_id)
    if not m:
        return None
    d = m.groupdict()
    d["child_key"] = (
        f"wpp:{d['ini']}_{d['grade']}-{d['sec']}_{d['school']}_{d['city']}"
    )
    d["utt_id"] = utt_id
    # WPP story ids carry no difficulty level, so story_level is unknown rather
    # than guessed. Group WPP by `grade` (the child's), never by `story_level`.
    d["story_level"] = "?"
    return d


def parse_iitm_id(utt_id, recording=None):
    """IITM utterance ids are `<recording>_<NNNN>`; the recording is the
    speaker for leakage purposes (one talker per recording)."""
    rec = recording or utt_id.rsplit("_", 1)[0]
    return {
        "utt_id": utt_id,
        "grade": "adult",   # not children; no grade exists for this corpus
        "story": rec,
        "story_level": "?",
        "child_key": f"iitm:{rec}",
        "para": utt_id.rsplit("_", 1)[-1],
    }


# The primary IITM-English release, with WORD-LEVEL Kaldi transcription.
# Note this is NOT the copy under Raj_backup/.../Dataset/IITM-English — that one
# is a phone-only derivative built for the wav2vec2 phone recogniser (its
# `text` equivalents are ARPAbet, and its 110,504 CTMs are all *_phone_ctm).
# Same segment ids in both, so they are the same underlying data; only this
# folder has the words Whisper needs.
IITM_ROOT = paths.ROOTS["IITM_ROOT"]

CORPORA = {
    "kv": {
        "type": "label_tracks",
        "root": paths.ROOTS["KV_ROOT"],
        "subdirs": [
            "KV_AllIndia6Grades_Baseline2024",
            "KV_AllIndia6Grades_Baseline2025",
            "KV_AllIndia6Grades_Midline2024",
            "KV_AllIndia6Grades_Endline2025",
        ],
        "parser": parse_kv_id,
    },
    "wpp": {
        "type": "label_tracks",
        "root": paths.ROOTS["WPP_ROOT"],
        "subdirs": None,  # glob WPP_*_2020
        "subdir_glob": "WPP_*_2020",
        "parser": parse_wpp_id,
    },
    # Kaldi-style corpus: wav.scp + segments + text, one chunk per segment.
    # IITM-English already ships wav_*.scp and segments_* (110,508 segments,
    # max 23.0 s -> nothing needs chunking and nothing needs forced alignment).
    # `text` must be WORD-LEVEL (uppercase here; `text_norm` lowercases both
    # sides at scoring time, so that is harmless).
    "iitm": {
        "type": "kaldi",
        "parts": [
            {
                "wav_scp": os.path.join(IITM_ROOT, "transcription/train_English/wav.scp"),
                "segments": os.path.join(IITM_ROOT, "transcription/train_English/segments"),
                "text": os.path.join(IITM_ROOT, "transcription/train_English/text"),
                "name": "train",
            },
            {
                "wav_scp": os.path.join(IITM_ROOT, "transcription/dev_English/wav.scp"),
                "segments": os.path.join(IITM_ROOT, "transcription/dev_English/segments"),
                "text": os.path.join(IITM_ROOT, "transcription/dev_English/text"),
                "name": "dev",
            },
        ],
        # wav.scp paths are RELATIVE ("Audio/01_Financial_...wav") and the dev
        # recording ids are zero-padded while the filenames are not
        # (ahd_00300_long_00011_eng -> ahd_300_long_11_eng.wav), so resolve by
        # basename against this directory rather than trusting the scp path.
        "audio_dir": os.path.join(IITM_ROOT, "Audio"),
        "parser": parse_iitm_id,
        # IITM-English is three different speaking styles in one corpus,
        # separable by recording/file name and confirmed by filler-word rate:
        #   long_read      5,383 recs / 127.6 h / 85 s per rec / 0.0 fillers per 1k
        #                  long-form read news+articles
        #   short_read    31,776 recs /  41.9 h / 4.7 s per rec / 0.0 fillers per 1k
        #                  one-sentence read prompts -- too short to pack into the
        #                  30 s window, so they cost the most compute per hour
        #   conversational    64 recs /  15.3 h / 851 s per rec / 12.7-40.6 fillers
        #                  IE_vy_* interviews and *_text podcasts
        # Default keeps long_read only: best match to the read-aloud target task
        # and by far the best compute-per-hour after merging.
        "categories": {
            "long_read": lambda rid, wav: "_long_" in wav,
            "short_read": lambda rid, wav: bool(
                re.match(r"^\d+_eng_\d+_\d+\.wav$", wav)
            ),
            "conversational": lambda rid, wav: not (
                "_long_" in wav or re.match(r"^\d+_eng_\d+_\d+\.wav$", wav)
            ),
        },
        "keep_categories": ["long_read"],
        # Merge consecutive segments up to MAX_CHUNK_SEC. Gaps between segments
        # are silence the transcribers skipped, so crossing them is safe; the
        # merge is otherwise identical to the label-track packing used for WPP.
        "merge_segments": True,
        "merge_gap_tol": float("inf"),
    },
}


def read_kaldi_two_col(path):
    """utt_id -> rest-of-line, for wav.scp / text."""
    out = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split(None, 1)
            if len(parts) == 2:
                out[parts[0]] = parts[1].strip()
    return out


def merge_runs(items, max_sec, gap_tol):
    """items: sorted [(start, end, text)] within ONE recording.
    Greedily pack into <= max_sec spans; never splits an item."""
    out, cur = [], None
    for s_, e_, t_ in items:
        if cur and (e_ - cur[0]) <= max_sec and (s_ - cur[1]) <= gap_tol:
            cur = (cur[0], e_, cur[2] + " " + t_)
        else:
            if cur:
                out.append(cur)
            cur = (s_, e_, t_)
    if cur:
        out.append(cur)
    return out


def collect_kaldi(corpus, spec, stats):
    """One chunk per Kaldi segment. Segments longer than MAX_CHUNK_SEC are
    dropped, not split: without intra-segment word timings there is no way to
    cut the transcript to match, and truncating the audio while keeping the
    full text is exactly the failure mode described at the top of this file."""
    chunk_rows, utt_rows = [], []
    audio_dir = spec.get("audio_dir")
    sr_cache = {}
    cats = spec.get("categories") or {}
    keep = set(spec.get("keep_categories") or cats.keys())
    do_merge = spec.get("merge_segments", False)
    gap_tol = spec.get("merge_gap_tol", 0.0)
    pending = collections.defaultdict(list)   # recording -> [(start, end, text)]
    rec_wav = {}

    for part in spec["parts"]:
        for key in ("wav_scp", "segments", "text"):
            if not os.path.exists(part[key]):
                stats[f"{corpus}[{part['name']}]_missing_{key}"] += 1
        if not all(os.path.exists(part[k]) for k in ("wav_scp", "segments", "text")):
            continue

        wav_scp = read_kaldi_two_col(part["wav_scp"])
        texts = read_kaldi_two_col(part["text"])

        # Repair stale paths by basename against the local audio directory.
        for rec, path in list(wav_scp.items()):
            if os.path.exists(path):
                continue
            if audio_dir:
                cand = os.path.join(audio_dir, os.path.basename(path))
                if os.path.exists(cand):
                    wav_scp[rec] = cand
                    stats[f"{corpus}_wav_path_repaired"] += 1
                    continue
            wav_scp.pop(rec)
            stats[f"{corpus}_wav_unresolved"] += 1

        with open(part["segments"], encoding="utf-8") as fh:
            for line in fh:
                f = line.split()
                if len(f) < 4:
                    continue
                utt_id, rec = f[0], f[1]
                try:
                    start, end = float(f[2]), float(f[3])
                except ValueError:
                    stats[f"{corpus}_segment_unparsed"] += 1
                    continue

                if utt_id not in texts:
                    stats[f"{corpus}_no_text"] += 1
                    continue
                if rec not in wav_scp:
                    stats[f"{corpus}_no_wav"] += 1
                    continue

                text = texts[utt_id].strip()
                if not text:
                    stats[f"{corpus}_empty_text"] += 1
                    continue

                dur = end - start
                if dur < MIN_CHUNK_SEC:
                    stats["chunk_too_short"] += 1
                    continue
                if dur > MAX_CHUNK_SEC:
                    stats["oversized_single_lines"] += 1
                    if DROP_OVERSIZED:
                        stats["chunk_dropped_oversized"] += 1
                        continue

                wav_path = wav_scp[rec]
                if cats:
                    base = os.path.basename(wav_path)
                    cat = next((c for c, fn in cats.items() if fn(rec, base)), "?")
                    if cat not in keep:
                        stats[f"{corpus}_skipped[{cat}]"] += 1
                        continue
                if wav_path not in sr_cache:
                    try:
                        sr_cache[wav_path] = sf.info(wav_path).samplerate
                    except Exception:
                        sr_cache[wav_path] = None
                sr = sr_cache[wav_path]
                if sr is None:
                    stats["wav_unreadable"] += 1
                    continue
                if sr != 16000:
                    stats["wav_not_16k"] += 1
                    continue

                if do_merge:
                    # Key by recording ONLY, never by part. IITM's official
                    # train/dev split is segment-level: all 2,414 dev recordings
                    # also appear in train (segments are disjoint, recordings are
                    # not). Merging per (part, recording) would produce two
                    # interleaved, time-overlapping chunk sets from one wav and
                    # duplicate ids. Pooling first gives one coherent timeline
                    # per recording; the train/dev split is then redone by
                    # speaker below, which also fixes the leakage in their split.
                    pending[rec].append((start, end, text))
                    rec_wav[rec] = wav_path
                    continue

                meta = spec["parser"](utt_id, rec)
                row = {
                    "chunk_id": f"{utt_id}__c000",
                    "utt_id": utt_id,
                    # the true Kaldi recording: many segments share one long wav
                    "recording": rec,
                    "wav_path": paths.templatize(wav_path),
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "duration": round(dur, 3),
                    "text": text,
                    "grade": meta["grade"],
                    "corpus": corpus,
                    "round": part["name"],
                    "story": meta["story"],
                    "story_level": meta["story_level"],
                    "child_key": meta["child_key"],
                    "para": meta["para"],
                }
                chunk_rows.append(row)
                stats["chunk_sec"] += dur
                utt_rows.append(
                    {
                        "utt_id": utt_id,
                        "grade": meta["grade"],
                        "corpus": corpus,
                        "round": part["name"],
                        "story": meta["story"],
                        "story_level": meta["story_level"],
                        "child_key": meta["child_key"],
                        "reference": text,
                    }
                )
                stats["utts"] += 1

    for rec, items in pending.items():
        items.sort()
        wav_path = rec_wav[rec]
        part_name = "pooled"
        for i, (s_, e_, t_) in enumerate(merge_runs(items, MAX_CHUNK_SEC, gap_tol)):
            uid = f"{rec}__m{i:03d}"
            meta = spec["parser"](uid, rec)
            base = {
                "utt_id": uid,
                "grade": meta["grade"],
                "corpus": corpus,
                "round": part_name,
                "story": meta["story"],
                "story_level": meta["story_level"],
                "child_key": meta["child_key"],
            }
            chunk_rows.append({
                "chunk_id": f"{uid}__c000",
                "recording": rec,
                "wav_path": paths.templatize(wav_path),
                "start": round(s_, 3),
                "end": round(e_, 3),
                "duration": round(e_ - s_, 3),
                "text": t_,
                "para": str(i),
                **base,
            })
            utt_rows.append({**base, "reference": t_})
            stats["chunk_sec"] += e_ - s_
            stats["utts"] += 1
            stats[f"{corpus}_merged_chunks"] += 1

    return chunk_rows, utt_rows


def read_label_track(path):
    """-> list of (start, end, cleaned_text); tag-only lines are dropped."""
    lines = []
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            parts = raw.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            try:
                start, end = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            text = "\t".join(parts[2:])
            if is_tag_only(text):
                continue
            cleaned = strip_tags(text)
            if not cleaned:
                continue
            lines.append((start, end, cleaned))
    return lines


def chunk_lines(lines, max_sec):
    """Greedily merge consecutive label-track lines into <= max_sec windows.

    A line is never split, so text and audio stay aligned. A single line that is
    itself longer than max_sec is emitted alone and reported as `oversized` —
    it will be truncated by the feature extractor, so you want that count to be
    ~0 (it is, on this corpus).
    """
    chunks, cur, oversized = [], [], 0
    for start, end, text in lines:
        if cur and (end - cur[0][0]) > max_sec:
            chunks.append(cur)
            cur = []
        if not cur and (end - start) > max_sec:
            chunks.append([(start, end, text)])
            oversized += 1
            continue
        cur.append((start, end, text))
    if cur:
        chunks.append(cur)
    return chunks, oversized


def corpus_subdirs(spec):
    if spec.get("subdirs"):
        return spec["subdirs"]
    return sorted(
        os.path.basename(p)
        for p in glob.glob(os.path.join(spec["root"], spec["subdir_glob"]))
        if os.path.isdir(p)
    )


def collect(corpus_names):
    """Walk the label tracks and produce chunk rows + utterance references."""
    chunk_rows, utt_rows = [], []
    stats = collections.Counter()

    units = []
    for name in corpus_names:
        spec = CORPORA[name]
        if spec.get("type") == "kaldi":
            c, u = collect_kaldi(name, spec, stats)
            chunk_rows.extend(c)
            utt_rows.extend(u)
            continue
        if not os.path.isdir(spec["root"]):
            sys.exit(f"[FATAL] corpus root not found: {spec['root']}")
        for sub in corpus_subdirs(spec):
            units.append((name, spec, sub))

    for corpus, spec, rnd in units:
        lt_dir = os.path.join(spec["root"], rnd, "label_tracks")
        au_dir = os.path.join(spec["root"], rnd, "audios")
        for lt_path in sorted(glob.glob(os.path.join(lt_dir, "*_label_track.txt"))):
            utt_id = os.path.basename(lt_path)[: -len("_label_track.txt")]
            meta = spec["parser"](utt_id)
            if meta is None:
                stats[f"utt_id_unparsed[{corpus}]"] += 1
                continue

            wav_path = os.path.join(au_dir, utt_id + ".wav")
            if not os.path.exists(wav_path):
                stats["wav_missing"] += 1
                continue

            lines = read_label_track(lt_path)
            if not lines:
                stats["utt_empty_transcript"] += 1
                continue

            try:
                info = sf.info(wav_path)
            except Exception:
                stats["wav_unreadable"] += 1
                continue
            if info.samplerate != 16000:
                stats["wav_not_16k"] += 1
                continue
            wav_dur = info.duration

            chunks, oversized = chunk_lines(lines, MAX_CHUNK_SEC)
            stats["oversized_single_lines"] += oversized

            kept = 0
            for i, grp in enumerate(chunks):
                start = max(0.0, grp[0][0])
                end = min(wav_dur, grp[-1][1])
                if end - start < MIN_CHUNK_SEC:
                    stats["chunk_too_short"] += 1
                    continue
                if end - start > MAX_CHUNK_SEC and DROP_OVERSIZED:
                    # A single label-track line longer than the encoder window.
                    # 23 of these exist across KV+WPP (all WPP, up to 57 s).
                    # Keeping them means the feature extractor silently drops
                    # the tail while the label keeps the full text — exactly the
                    # train-to-hallucinate pattern this script exists to avoid.
                    stats["chunk_dropped_oversized"] += 1
                    continue
                text = " ".join(t for _, _, t in grp).strip()
                if not text:
                    stats["chunk_empty_text"] += 1
                    continue
                chunk_rows.append(
                    {
                        "chunk_id": f"{utt_id}__c{i:03d}",
                        "utt_id": utt_id,
                        "recording": utt_id,  # one file per utterance here
                        "wav_path": paths.templatize(wav_path),
                        "start": round(start, 3),
                        "end": round(end, 3),
                        "duration": round(end - start, 3),
                        "text": text,
                        "grade": meta["grade"],
                        "corpus": corpus,
                        "round": rnd,
                        "story": meta["story"],
                        "story_level": meta["story_level"],
                        "child_key": meta["child_key"],
                        "para": meta["para"],
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
                    "grade": meta["grade"],
                    "corpus": corpus,
                    "round": rnd,
                    "story": meta["story"],
                    "story_level": meta["story_level"],
                    "child_key": meta["child_key"],
                    "reference": " ".join(t for _, _, t in lines).strip(),
                }
            )
            stats["utts"] += 1

    return chunk_rows, utt_rows, stats


def split_established(utt_rows):
    """test = the B24+B25 set already used for the wav2vec2 results.

    Children who appear in that test set are removed from train/dev entirely,
    even when their Midline/Endline recordings are what would otherwise be used
    for training — otherwise the same child is on both sides of the split.
    """
    if not os.path.exists(ESTABLISHED_TEST_TEXT):
        sys.exit(f"[FATAL] established test list not found: {ESTABLISHED_TEST_TEXT}")

    test_ids = set()
    with open(ESTABLISHED_TEST_TEXT, encoding="utf-8") as fh:
        for line in fh:
            uid = re.split(r"[\t ]", line.strip(), maxsplit=1)[0]
            if uid:
                test_ids.add(uid)

    test = [r for r in utt_rows if r["utt_id"] in test_ids]
    test_children = {r["child_key"] for r in test}
    pool = [
        r
        for r in utt_rows
        if r["utt_id"] not in test_ids and r["child_key"] not in test_children
    ]
    leaked = sum(
        1
        for r in utt_rows
        if r["utt_id"] not in test_ids and r["child_key"] in test_children
    )
    return test, pool, leaked


def split_children(rows, frac, seed, exclude_children=frozenset()):
    """Child-disjoint, grade-stratified split. Returns (held_out, remainder)."""
    rng = random.Random(seed)
    by_grade = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        by_grade[r["grade"]][r["child_key"]].append(r)

    held, rest = [], []
    for grade in sorted(by_grade):
        children = sorted(c for c in by_grade[grade] if c not in exclude_children)
        rng.shuffle(children)
        n_hold = max(1, int(round(frac * len(children)))) if children else 0
        hold_set = set(children[:n_hold])
        for child, items in by_grade[grade].items():
            (held if child in hold_set else rest).extend(items)
    return held, rest


def write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def summarize(name, utt_rows, chunk_rows, out):
    by_utt = collections.defaultdict(list)
    for c in chunk_rows:
        by_utt[c["utt_id"]].append(c)
    g_utt = collections.Counter(r["grade"] for r in utt_rows)
    g_sec = collections.Counter()
    g_chunk = collections.Counter()
    for r in utt_rows:
        for c in by_utt[r["utt_id"]]:
            g_sec[r["grade"]] += c["duration"]
            g_chunk[r["grade"]] += 1
    out.append(f"\n=== {name} ===")
    out.append(f"{'grade':>7} {'utts':>7} {'chunks':>8} {'hours':>8}")
    for g in sorted(g_utt):
        out.append(f"{g:>7} {g_utt[g]:>7} {g_chunk[g]:>8} {g_sec[g]/3600:>8.2f}")
    out.append(
        f"{'ALL':>7} {sum(g_utt.values()):>7} {sum(g_chunk.values()):>8} "
        f"{sum(g_sec.values())/3600:>8.2f}"
    )
    out.append(f"speakers (child_key): {len({r['child_key'] for r in utt_rows})}")
    by_corpus = collections.Counter(r.get("corpus", "?") for r in utt_rows)
    if len(by_corpus) > 1:
        out.append(
            "by corpus: "
            + ", ".join(f"{c}={n}" for c, n in sorted(by_corpus.items()))
        )


def main():
    global MAX_CHUNK_SEC, DROP_OVERSIZED
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=os.path.join(paths.MANIFESTS, "default"))
    ap.add_argument(
        "--split_mode",
        choices=["established", "child_random", "all_train"],
        default="established",
        help="established = reuse the B24+B25 test set the wav2vec2 numbers were "
        "computed on (recommended, keeps results comparable). child_random = "
        "fresh child-disjoint 80/10/10 over all four rounds. all_train = use "
        "ALL four rounds for train/dev and emit an empty test split — for when "
        "you are testing on an external corpus (e.g. MPS) instead; run "
        "check_speaker_overlap.py against that corpus before training.",
    )
    ap.add_argument(
        "--corpora",
        default="kv",
        type=lambda v: [x.strip() for x in v.split(",") if x.strip()],
        help="comma-separated corpora to pool: kv, wpp (both share the same "
        "audios/ + label_tracks/ layout, so no extra work is needed to mix them)",
    )
    ap.add_argument(
        "--iitm_root", default="",
        help="override the IITM-English directory (use once the upload lands)",
    )
    ap.add_argument(
        "--iitm_text_train", default="",
        help="WORD-LEVEL transcripts for IITM train, Kaldi `utt_id text` format. "
        "The phone files already in that directory are ARPAbet and will not work.",
    )
    ap.add_argument("--iitm_text_dev", default="")
    ap.add_argument(
        "--iitm_keep",
        default="long_read",
        help="comma-separated IITM speaking-style categories to keep: "
        "long_read (127.6 h, long-form read), short_read (41.9 h, ~4.7 s "
        "one-sentence prompts), conversational (15.3 h, interviews/podcasts). "
        "Default long_read.",
    )
    ap.add_argument(
        "--iitm_no_merge", action="store_true",
        help="keep IITM Kaldi segments as-is instead of packing them to "
        "<= --max_chunk_sec (packing cuts steps/epoch ~5x, see README)",
    )
    ap.add_argument("--iitm_segments_train", default="")
    ap.add_argument("--iitm_segments_dev", default="")
    ap.add_argument("--iitm_wav_scp_train", default="")
    ap.add_argument("--iitm_wav_scp_dev", default="")
    ap.add_argument(
        "--cap_hours", default="", action="append" if False else "store",
        help="cap a corpus's contribution, e.g. --cap_hours iitm=60. Whole "
        "speakers are dropped (never partial ones) so the split stays "
        "speaker-disjoint. Use this when pooling a large out-of-domain corpus "
        "with a smaller in-domain one so the big one does not dominate.",
    )
    ap.add_argument("--dev_frac", type=float, default=0.08)
    ap.add_argument("--test_frac", type=float, default=0.10, help="child_random only")
    ap.add_argument("--max_chunk_sec", type=float, default=MAX_CHUNK_SEC)
    ap.add_argument(
        "--keep_oversized",
        action="store_true",
        help="keep chunks made of a single label-track line longer than "
        "--max_chunk_sec. They WILL be truncated by the feature extractor; the "
        "default drops them instead.",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    MAX_CHUNK_SEC = args.max_chunk_sec
    DROP_OVERSIZED = not args.keep_oversized

    if args.iitm_root:
        spec = CORPORA["iitm"]
        spec["audio_dir"] = os.path.join(args.iitm_root, "Audio")
        for part in spec["parts"]:
            sub = f"transcription/{part['name']}_English"
            for key, fname in (
                ("wav_scp", "wav.scp"), ("segments", "segments"), ("text", "text"),
            ):
                part[key] = os.path.join(args.iitm_root, sub, fname)
    for part in CORPORA["iitm"]["parts"]:
        for key, flag in (
            ("text", f"iitm_text_{part['name']}"),
            ("segments", f"iitm_segments_{part['name']}"),
            ("wav_scp", f"iitm_wav_scp_{part['name']}"),
        ):
            override = getattr(args, flag, "")
            if override:
                part[key] = override

    CORPORA["iitm"]["keep_categories"] = [
        x.strip() for x in args.iitm_keep.split(",") if x.strip()
    ]
    CORPORA["iitm"]["merge_segments"] = not args.iitm_no_merge

    unknown = [c for c in args.corpora if c not in CORPORA]
    if unknown:
        sys.exit(f"[FATAL] unknown corpus {unknown}; known: {sorted(CORPORA)}")
    if args.split_mode == "established" and "kv" not in args.corpora:
        sys.exit(
            "[FATAL] --split_mode established needs 'kv' in --corpora: its test "
            "set is the KV B24+B25 list."
        )

    print("[INFO] scanning label tracks ...")
    chunk_rows, utt_rows, stats = collect(args.corpora)

    caps = {}
    for item in filter(None, (x.strip() for x in args.cap_hours.split(","))):
        name, _, hours = item.partition("=")
        if not hours:
            sys.exit(f"[FATAL] --cap_hours expects corpus=HOURS, got '{item}'")
        caps[name.strip()] = float(hours)

    # A requested corpus that contributes nothing is always a configuration
    # error (missing transcripts, wrong path) — never let it pass as a smaller
    # training set than you asked for.
    contributed = collections.Counter(r["corpus"] for r in utt_rows)
    for name in args.corpora:
        if contributed[name] == 0:
            detail = "; ".join(f"{k}={v}" for k, v in sorted(stats.items())
                               if k.startswith(name))
            sys.exit(
                f"[FATAL] corpus '{name}' contributed 0 utterances. "
                f"{detail or 'no diagnostics recorded'}\n"
                f"        For 'iitm' this almost always means the WORD-LEVEL "
                f"text file is missing — the ARPAbet phone files in that "
                f"directory cannot be used. Pass --iitm_text_train / "
                f"--iitm_text_dev (or --iitm_root) once the upload lands."
            )
    print(f"[INFO] {len(utt_rows)} utterances -> {len(chunk_rows)} chunks")

    out = ["MANIFEST BUILD REPORT",
           f"corpora = {','.join(args.corpora)}",
           f"split_mode = {args.split_mode}",
           f"max_chunk_sec = {MAX_CHUNK_SEC}", f"seed = {args.seed}", "",
           "--- scan counters ---"]
    for k in sorted(stats):
        v = stats[k]
        out.append(f"{k:>28}: {v/3600:.2f} h" if k == "chunk_sec" else f"{k:>28}: {v}")

    if caps:
        dur_by_utt = collections.Counter()
        for c in chunk_rows:
            dur_by_utt[c["utt_id"]] += c["duration"]
        keep_utts, cap_report = set(), []
        for name in args.corpora:
            rows = [r for r in utt_rows if r["corpus"] == name]
            if name not in caps:
                keep_utts.update(r["utt_id"] for r in rows)
                continue
            by_child = collections.defaultdict(list)
            for r in rows:
                by_child[r["child_key"]].append(r)
            children = sorted(by_child)
            random.Random(args.seed).shuffle(children)
            budget, used, kept_children = caps[name] * 3600, 0.0, 0
            for child in children:
                child_sec = sum(dur_by_utt[r["utt_id"]] for r in by_child[child])
                if used + child_sec > budget and kept_children:
                    continue
                used += child_sec
                kept_children += 1
                keep_utts.update(r["utt_id"] for r in by_child[child])
            cap_report.append(
                f"{name}: capped at {caps[name]:.1f} h -> kept {used/3600:.2f} h "
                f"from {kept_children}/{len(children)} speakers"
            )
        utt_rows = [r for r in utt_rows if r["utt_id"] in keep_utts]
        chunk_rows = [r for r in chunk_rows if r["utt_id"] in keep_utts]
        out.append("")
        out.extend(cap_report)

    if args.split_mode == "all_train":
        test = []
        dev, train = split_children(utt_rows, args.dev_frac, args.seed)
        out.append(
            "\nall_train: every round goes to train/dev; the test split is empty "
            "by design. Testing happens on an external corpus — verify speaker "
            "disjointness with check_speaker_overlap.py first."
        )
    elif args.split_mode == "established":
        test, pool, leaked = split_established(utt_rows)
        out.append(
            f"\nutterances excluded from train/dev because the child also "
            f"appears in the test set: {leaked}"
        )
        dev, train = split_children(pool, args.dev_frac, args.seed)
    else:
        test, pool = split_children(utt_rows, args.test_frac, args.seed)
        test_children = {r["child_key"] for r in test}
        dev, train = split_children(
            pool, args.dev_frac, args.seed + 1, exclude_children=test_children
        )

    split_of = {}
    for name, rows in (("train", train), ("dev", dev), ("test", test)):
        for r in rows:
            split_of[r["utt_id"]] = name

    # sanity: no child may straddle two splits
    child_splits = collections.defaultdict(set)
    for rows, name in ((train, "train"), (dev, "dev"), (test, "test")):
        for r in rows:
            child_splits[r["child_key"]].add(name)
    straddling = [c for c, s in child_splits.items() if len(s) > 1]
    out.append(f"children appearing in more than one split: {len(straddling)}")
    if straddling:
        out.append("  !! speaker leakage — investigate before trusting the WERs")

    chunk_fields = [
        "chunk_id", "utt_id", "recording", "wav_path", "start", "end",
        "duration", "text", "grade", "corpus", "round", "story", "story_level",
        "child_key", "para",
    ]
    utt_fields = [
        "utt_id", "grade", "corpus", "round", "story", "story_level",
        "child_key", "reference",
    ]

    for name in ("train", "dev", "test"):
        u = [r for r in utt_rows if split_of.get(r["utt_id"]) == name]
        c = [r for r in chunk_rows if split_of.get(r["utt_id"]) == name]
        write_csv(os.path.join(args.out_dir, f"chunks_{name}.csv"), c, chunk_fields)
        write_csv(os.path.join(args.out_dir, f"utt_reference_{name}.csv"), u, utt_fields)
        summarize(name, u, c, out)

    report = "\n".join(out)
    with open(os.path.join(args.out_dir, "manifest_stats.txt"), "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print(report)
    print(f"\n[DONE] manifests written to {args.out_dir}")


if __name__ == "__main__":
    main()
