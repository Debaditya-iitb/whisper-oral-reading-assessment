#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bring every referenced audio file INSIDE this folder, and repoint the manifests
at the internal copy.

After this runs the project is completely self-contained: copy the folder to any
machine and it works with **no environment variables and no external corpora**.
Manifests reference `${AUDIO_ROOT}/...`, and `AUDIO_ROOT` is always
`<this folder>/data/audio`, derived from `config_paths.py`.

    python copy_audio_to_local_disk.py                 # hardlink where possible
    python copy_audio_to_local_disk.py --mode copy     # force real copies
    python copy_audio_to_local_disk.py --dry_run

Layout created:

    data/audio/wpp/WPP_VV_Mumbai_2020/audios/xxx.wav
    data/audio/iitm/Audio/xxx.wav
    data/audio/mps/audios/xxx.wav

MODE
----
`hardlink` (default) makes the files appear inside the folder while sharing
disk blocks with the originals, so it costs ~0 extra bytes on a machine where
both live on one filesystem. A hardlink is a real directory entry, not a
symlink: `rsync`, `tar`, `scp` and `zip` all read through it and transfer full
file content, which is exactly what you want when shipping the folder. It falls
back to a real copy automatically when the source is on another filesystem.

Use `--mode copy` if you intend to delete or move the original corpora and want
belt-and-braces independence (hardlinks survive deletion of the original too —
the data is only freed when the last link goes — but copies are easier to
reason about).

Re-run this after any `03_build_chunked_datasets.sh`, which regenerates the manifests pointing
back at the external roots. `03_build_chunked_datasets.sh LOCALIZE=1` does both.
"""

import argparse
import csv
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_paths as paths  # noqa: E402

# ${ROOT} -> subdirectory name under data/audio/
SUBDIR = {
    "WPP_ROOT": "wpp",
    "IITM_ROOT": "iitm",
    "MPS_ROOT": "mps",
    "KV_ROOT": "kv",
}


def human(n):
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}"
        n /= 1024


def manifest_paths():
    out = []
    for split in ("train", "dev", "test"):
        d = os.path.join(paths.DATA, split)
        if not os.path.isdir(d):
            continue
        for ds in sorted(os.listdir(d)):
            p = os.path.join(d, ds, "chunks.csv")
            if os.path.exists(p):
                out.append(p)
    return out


def plan(csv_paths):
    """-> {source_abs: dest_abs}, and per-manifest row rewrites."""
    mapping = {}
    unknown = set()
    for cp in csv_paths:
        with open(cp, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                wp = r["wav_path"]
                m = paths._VAR.match(wp)
                if not m:
                    unknown.add(wp)
                    continue
                root = m.group(1)
                if root == "AUDIO_ROOT":
                    continue  # already localized
                sub = SUBDIR.get(root)
                if sub is None:
                    unknown.add(wp)
                    continue
                rel = wp[m.end():].lstrip("/")
                src = os.path.join(paths.ROOTS[root], rel)
                dst = os.path.join(paths.AUDIO_ROOT, sub, rel)
                mapping[src] = dst
    return mapping, unknown


def transfer(src, dst, mode):
    """-> ('hardlink'|'copy'|'exists'|'missing', bytes_added)"""
    if os.path.exists(dst):
        return "exists", 0
    if not os.path.exists(src):
        return "missing", 0
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    size = os.path.getsize(src)
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return "hardlink", 0
        except OSError:
            pass  # cross-device, or a filesystem without hardlinks
    shutil.copy2(src, dst)
    return "copy", size


def rewrite(cp, dry_run):
    """Point every row at ${AUDIO_ROOT}. Returns rows changed."""
    with open(cp, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames
        rows = list(reader)

    changed = 0
    for r in rows:
        m = paths._VAR.match(r["wav_path"])
        if not m or m.group(1) == "AUDIO_ROOT":
            continue
        sub = SUBDIR.get(m.group(1))
        if sub is None:
            continue
        rel = r["wav_path"][m.end():].lstrip("/")
        r["wav_path"] = f"${{AUDIO_ROOT}}/{sub}/{rel}"
        changed += 1

    if changed and not dry_run:
        tmp = cp + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, cp)
    return changed


def rewrite_wav_scp(dry_run):
    """Kaldi wav.scp holds absolute paths (Kaldi does not expand variables), so
    regenerate them from the now-internal locations."""
    n = 0
    for split in ("train", "dev", "test"):
        d = os.path.join(paths.DATA, split)
        if not os.path.isdir(d):
            continue
        for ds in sorted(os.listdir(d)):
            leaf = os.path.join(d, ds)
            cp = os.path.join(leaf, "chunks.csv")
            if not os.path.exists(cp):
                continue
            with open(cp, encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            wav = {}
            for r in rows:
                rec = r.get("recording") or r["utt_id"]
                wav[rec] = paths.resolve(r["wav_path"])
            for scp in (os.path.join(leaf, "wav.scp"),
                        os.path.join(leaf, "reference", "wav.scp")):
                if not os.path.exists(scp):
                    continue
                if dry_run:
                    n += 1
                    continue
                # reference/wav.scp may be keyed by utterance rather than
                # recording, so rewrite only the path column, keeping the keys.
                out = []
                with open(scp, encoding="utf-8") as fh:
                    for line in fh:
                        parts = line.rstrip("\n").split(None, 1)
                        if len(parts) != 2:
                            continue
                        key, old = parts
                        new = wav.get(key)
                        if new is None:
                            # utterance-keyed: find it via its chunk rows
                            new = next(
                                (paths.resolve(r["wav_path"])
                                 for r in rows if r["utt_id"] == key),
                                old,
                            )
                        out.append(f"{key} {new}\n")
                with open(scp, "w", encoding="utf-8") as fh:
                    fh.writelines(out)
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["hardlink", "copy"], default="hardlink")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    csv_paths = manifest_paths()
    if not csv_paths:
        sys.exit("[FATAL] no manifests under data/ — run `bash 03_build_chunked_datasets.sh` first")

    mapping, unknown = plan(csv_paths)
    if unknown:
        print(f"[WARN] {len(unknown)} paths reference an unknown root and will be "
              f"left alone, e.g. {sorted(unknown)[0]}")

    if not mapping:
        # The CSVs are portable (${AUDIO_ROOT} is resolved at read time) but
        # wav.scp holds ABSOLUTE paths, because Kaldi does not expand variables.
        # After the folder moves machine those are stale, and this is the only
        # place that repairs them — so do it instead of returning early.
        print("[OK] manifests already point at ${AUDIO_ROOT}")
        n_scp = rewrite_wav_scp(args.dry_run)
        print(f"refreshed {n_scp} wav.scp files to this machine's paths "
              f"({paths.AUDIO_ROOT})")
        return 0

    print(f"{len(mapping)} unique audio files -> {paths.AUDIO_ROOT}")
    print(f"mode: {args.mode}{'  (DRY RUN)' if args.dry_run else ''}\n")

    stats = {"hardlink": 0, "copy": 0, "exists": 0, "missing": 0}
    added = 0
    missing_examples = []
    for i, (src, dst) in enumerate(sorted(mapping.items()), 1):
        if args.dry_run:
            stats["exists" if os.path.exists(dst) else "hardlink"] += 1
            continue
        kind, nbytes = transfer(src, dst, args.mode)
        stats[kind] += 1
        added += nbytes
        if kind == "missing" and len(missing_examples) < 3:
            missing_examples.append(src)
        if i % 2000 == 0:
            print(f"  {i}/{len(mapping)} ...", flush=True)

    print()
    for k, v in stats.items():
        if v:
            print(f"  {k:<9} {v}")
    if missing_examples:
        print("\n[FATAL] source files missing, e.g.:")
        for m in missing_examples:
            print(f"  {m}")
        return 1
    print(f"  extra disk used: {human(added)}")

    changed = sum(rewrite(cp, args.dry_run) for cp in csv_paths)
    n_scp = rewrite_wav_scp(args.dry_run)
    print(f"\nrewrote {changed} manifest rows and {n_scp} wav.scp files "
          f"to ${{AUDIO_ROOT}}")

    if not args.dry_run:
        total = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fs in os.walk(paths.AUDIO_ROOT) for f in fs
        )
        print(f"data/audio now holds {human(total)} of audio")
        print("\nThe folder is now self-contained: no environment variables, no "
              "external corpora.\nVerify with:  python check_paths_resolve.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
