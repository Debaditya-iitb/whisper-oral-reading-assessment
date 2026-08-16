#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run this on the CURRENT server to find out exactly which audio files the new
server needs, and to emit ready-to-paste rsync commands.

Copying whole corpus directories moves 39 GB and ~45,000 files. The built
manifests only reference a subset — notably IITM, where the long-form-read
selection uses 5,383 of 37,223 wavs. This writes one file list per corpus root
so rsync can transfer only those.

    python make_audio_transfer_list.py --out transfer/
    # then, per the printed commands:
    rsync -av --files-from=transfer/IITM_ROOT.files "$IITM_ROOT/" newhost:/data/iitm/
"""

import argparse
import collections
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_paths as paths  # noqa: E402


def human(n):
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}"
        n /= 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="transfer")
    ap.add_argument("--dest_host", default="NEWHOST")
    ap.add_argument("--dest_dir", default="/data/corpora")
    ap.add_argument("--no_size", action="store_true", help="skip stat() (faster)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    by_root = collections.defaultdict(set)
    n_rows = 0
    for split in ("train", "dev", "test"):
        split_dir = os.path.join(paths.DATA, split)
        if not os.path.isdir(split_dir):
            continue
        for ds in sorted(os.listdir(split_dir)):
            cpath = os.path.join(split_dir, ds, "chunks.csv")
            if not os.path.exists(cpath):
                continue
            with open(cpath, encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    n_rows += 1
                    wp = r["wav_path"]
                    m = paths._VAR.match(wp)
                    if not m:
                        by_root["ABSOLUTE"].add(wp)
                        continue
                    name = m.group(1)
                    rel = wp[m.end():].lstrip("/")
                    by_root[name].add(rel)

    if not by_root:
        sys.exit("[FATAL] no manifests found — run `bash 03_build_chunked_datasets.sh` first")

    print(f"scanned {n_rows} chunk rows\n")
    print(f"{'root':<12} {'files':>8} {'size':>12}   list")
    total = 0
    cmds = []
    for name in sorted(by_root):
        rels = sorted(by_root[name])
        root = paths.ROOTS.get(name, "")
        size = 0
        if not args.no_size and root:
            for rel in rels:
                p = os.path.join(root, rel)
                try:
                    size += os.path.getsize(p)
                except OSError:
                    pass
        total += size
        listfile = os.path.join(args.out, f"{name}.files")
        with open(listfile, "w", encoding="utf-8") as fh:
            fh.write("\n".join(rels) + "\n")
        print(f"{name:<12} {len(rels):>8} {human(size):>12}   {listfile}")
        if root:
            dest = os.path.join(args.dest_dir, os.path.basename(root.rstrip("/")))
            cmds.append(
                f"rsync -av --files-from={listfile} "
                f"{root.rstrip('/')}/ {args.dest_host}:{dest}/"
            )

    print(f"{'TOTAL':<12} {sum(len(v) for v in by_root.values()):>8} {human(total):>12}")

    # IITM also needs its Kaldi transcription dir; MPS needs data.json. Neither
    # is referenced from chunks.csv (only the wavs are) but 03_build_chunked_datasets.sh needs
    # them if you ever rebuild the manifests on the new server.
    extra = [
        (paths.ROOTS["IITM_ROOT"], "transcription"),
        (paths.ROOTS["MPS_ROOT"], "data.json"),
    ]
    print("\nalso copy (needed only if you rebuild data/ on the new server):")
    for root, sub in extra:
        p = os.path.join(root, sub)
        mark = "OK " if os.path.exists(p) else "?? "
        print(f"  {mark} {p}")

    print("\n--- rsync commands ---")
    for c in cmds:
        print(c)
    print(
        f"\nrsync -av --exclude hf_whisper --exclude models --exclude __pycache__ "
        f"{paths.WHISPER_HOME}/ {args.dest_host}:{args.dest_dir}/../whisper_finetune/"
    )
    print("\nThen on the new server set the roots to match --dest_dir and run "
          "`python check_paths_resolve.py`.")


if __name__ == "__main__":
    main()
