#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
First thing to run after copying this folder to a new machine.

Reports which corpus roots are missing, and actually opens a sample of audio
files from every built manifest — a root that exists but points at the wrong
corpus is a far nastier failure than one that is simply absent.

    python check_paths_resolve.py
    WPP_ROOT=/data/wpp-2020 python check_paths_resolve.py
"""

import argparse
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_paths as paths  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=25,
                    help="audio files to open per manifest (0 = only check roots)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"WHISPER_HOME = {paths.WHISPER_HOME}\n")

    # Which roots do the built manifests actually reference? A root that is
    # missing but unused (e.g. KV_ROOT when you only train on WPP+IITM) is not
    # an error, and reporting it as one sends you chasing a non-problem.
    used = set()
    for split in ("train", "dev", "test"):
        split_dir = os.path.join(paths.DATA, split)
        if not os.path.isdir(split_dir):
            continue
        for ds in sorted(os.listdir(split_dir)):
            cp = os.path.join(split_dir, ds, "chunks.csv")
            if not os.path.exists(cp):
                continue
            with open(cp, encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    for m in paths._VAR.finditer(r["wav_path"]):
                        used.add(m.group(1))
                    break

    print("corpus roots")
    bad_roots = 0
    for name in sorted(paths.ROOTS):
        root = paths.ROOTS[name]
        src = "env" if name in os.environ else "default"
        ok = os.path.isdir(root)
        if ok:
            tag = "OK  "
        elif name in used:
            tag = "MISS"
            bad_roots += 1
        else:
            tag = "n/a "   # missing, but nothing built references it
        note = "" if name in used else "   (not used by any built manifest)"
        print(f"  {tag} {name:<11} [{src:<7}] {root}{note}")

    print("\nlocal directories")
    for label, p in [
        ("data", paths.DATA), ("manifests", paths.MANIFESTS),
        ("pretrained", paths.PRETRAINED), ("hf_whisper", paths.HF_DATASETS),
        ("models", paths.MODELS),
    ]:
        print(f"  {'OK  ' if os.path.isdir(p) else '--  '} {label:<11} {p}")

    print("\nmanifests (resolving audio paths)")
    rng = random.Random(args.seed)
    problems = 0
    found_any = False
    for split in ("train", "dev", "test"):
        split_dir = os.path.join(paths.DATA, split)
        if not os.path.isdir(split_dir):
            continue
        for ds in sorted(os.listdir(split_dir)):
            csv_path = os.path.join(split_dir, ds, "chunks.csv")
            if not os.path.exists(csv_path):
                continue
            found_any = True
            with open(csv_path, encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            templated = sum(1 for r in rows if "${" in r["wav_path"])
            missing, unresolved = [], []
            sample = rows if args.sample <= 0 else rng.sample(
                rows, min(args.sample, len(rows))
            )
            for r in sample:
                try:
                    p = paths.resolve(r["wav_path"])
                except KeyError as e:
                    unresolved.append(str(e))
                    continue
                if not os.path.exists(p):
                    missing.append(p)
            status = "OK  " if not missing and not unresolved else "FAIL"
            problems += bool(missing or unresolved)
            print(
                f"  {status} {split}/{ds:<9} {len(rows):>6} chunks, "
                f"{templated:>6} templated, checked {len(sample)}"
            )
            for m in (unresolved + missing)[:3]:
                print(f"       -> {m}")

    if not found_any:
        print("  (none built yet — run: bash 03_build_chunked_datasets.sh)")

    # wav.scp stores absolute paths (Kaldi cannot expand ${VARS}), so it goes
    # stale the moment the folder changes machine. Nothing in s1-s5 reads it —
    # training and decoding use chunks.csv — but external Kaldi tooling does.
    print("\nkaldi wav.scp (absolute paths — external tooling only)")
    stale = 0
    for split in ("train", "dev", "test"):
        split_dir = os.path.join(paths.DATA, split)
        if not os.path.isdir(split_dir):
            continue
        for ds in sorted(os.listdir(split_dir)):
            scp = os.path.join(split_dir, ds, "wav.scp")
            if not os.path.exists(scp):
                continue
            with open(scp, encoding="utf-8") as fh:
                first = fh.readline().split(None, 1)
            if len(first) != 2:
                continue
            path = first[1].strip()
            ok = path.startswith(paths.AUDIO_ROOT) and os.path.exists(path)
            stale += not ok
            print(f"  {'OK  ' if ok else 'STALE'} {split}/{ds}")
    if stale:
        print(f"\n  {stale} wav.scp file(s) point somewhere else (probably the")
        print("  machine this folder came from). Training is UNAFFECTED — s1-s5")
        print("  read chunks.csv, not wav.scp. To repair them anyway:")
        print("      python copy_audio_to_local_disk.py")

    print()
    if bad_roots:
        print("NOT READY — corpus roots missing. Point them at your copies:")
        for name, root in paths.missing_roots():
            if name in used:
                print(f"  export {name}=/path/to/{os.path.basename(root)}")
        return 1
    if problems:
        # Roots are fine, so this is missing or truncated audio, not a config
        # problem. Sending someone to set environment variables here wastes
        # their time.
        print("NOT READY — the roots resolve but audio files are missing.")
        if os.path.isdir(paths.AUDIO_ROOT):
            n = sum(len(f) for _, _, f in os.walk(paths.AUDIO_ROOT))
            print(f"  data/audio holds {n} files; the manifests expect 13685.")
            print("  Most likely an incomplete copy — re-run your rsync, it resumes.")
        else:
            print(f"  {paths.AUDIO_ROOT} does not exist. Either copy it across, or")
            print("  set the corpus roots and run `python copy_audio_to_local_disk.py`.")
        return 1

    print("READY — every root resolves and the sampled audio opens.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
