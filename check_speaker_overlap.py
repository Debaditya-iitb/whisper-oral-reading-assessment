#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Guard against speaker overlap between a training manifest and an external test
set (e.g. the MPS children's-speech corpus), and optionally emit a cleaned
training manifest with the offending utterances removed.

Why this is not just a set intersection
---------------------------------------
Speaker ids in two corpora almost never share a format. If MPS is a genuinely
different collection of children, the exact-id intersection will be empty and
that tells you nothing — the real risks are *near* matches (same child, id
written differently) and shared recording sessions. So this script:

  * extracts a speaker key from each side using a configurable rule,
  * reports exact key collisions,
  * reports near-collisions (normalised key equality, then a token/edit
    similarity pass), which is where re-registered or re-spelled children show up,
  * prints a sample of keys from both sides FIRST, so you can see whether your
    extraction rule actually produced something meaningful before believing a
    "0 overlaps" result.

A "0 overlaps" verdict from a rule that parsed nothing is the failure mode this
script exists to prevent — it exits non-zero if either side yields fewer than
two distinct keys.

Speaker-key rules (--train_rule / --test_rule):
    kv        KV convention: <initials>_<grade>-<sec>-<roll>_<school>, i.e. the
              utterance id up to the story field, date excluded. Same key the
              KV splits use.
    field:<name>       take the CSV column <name> verbatim
    regex:<pattern>    first capture group of <pattern> applied to the id
    prefix:<n>         first n underscore-separated fields of the id
    whole              the id itself

Usage:
    # inspect first — see what the rules actually extracted
    python check_speaker_overlap.py \
        --train manifests/chunks_train.csv --train_id_field utt_id --train_rule kv \
        --test  /path/to/mps_manifest.csv  --test_id_field  utt_id --test_rule whole \
        --report_only

    # then, once the rules look right, write a cleaned training manifest
    python check_speaker_overlap.py ... --out_train manifests/chunks_train_nooverlap.csv
"""

import argparse
import collections
import csv
import difflib
import os
import re
import sys
import unicodedata


def load_ids(path, id_field):
    """Accepts a CSV (needs id_field), or a Kaldi text/wav.scp/utt2spk, or a
    plain list of ids, one per line."""
    if not os.path.exists(path):
        sys.exit(f"[FATAL] not found: {path}")

    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        head = fh.readline()
        fh.seek(0)
        if "," in head and id_field and id_field in head:
            for row in csv.DictReader(fh):
                if row.get(id_field):
                    rows.append((row[id_field], row))
        else:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                uid = re.split(r"[\t ]", line, maxsplit=1)[0]
                rows.append((uid, {}))
    return rows


KV_RE = re.compile(
    r"^(?P<ini>[a-z]+)_(?P<grade>\d+)-(?P<sec>[a-z0-9]+)-(?P<roll>\d+)_"
    r"(?P<school>.+?)_(?P<date>\d{8}-\d{6}-\d+)_"
)


def make_key_fn(rule):
    if rule == "kv":
        def fn(uid, row):
            m = KV_RE.match(uid)
            if not m:
                return None
            d = m.groupdict()
            return f"{d['ini']}_{d['grade']}-{d['sec']}-{d['roll']}_{d['school']}"
        return fn
    if rule == "whole":
        return lambda uid, row: uid
    if rule.startswith("field:"):
        col = rule.split(":", 1)[1]
        return lambda uid, row: (row.get(col) or None)
    if rule.startswith("regex:"):
        pat = re.compile(rule.split(":", 1)[1])
        def fn(uid, row):
            m = pat.search(uid)
            return m.group(1) if m and m.groups() else (m.group(0) if m else None)
        return fn
    if rule.startswith("prefix:"):
        n = int(rule.split(":", 1)[1])
        return lambda uid, row: "_".join(uid.split("_")[:n]) or None
    sys.exit(f"[FATAL] unknown rule: {rule}")


def norm_key(k):
    """Aggressive normalisation for near-match detection only."""
    k = unicodedata.normalize("NFKD", k).lower()
    return re.sub(r"[^a-z0-9]", "", k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="training manifest / id list")
    ap.add_argument("--train_id_field", default="utt_id")
    ap.add_argument("--train_rule", default="kv")
    ap.add_argument("--test", required=True, help="test manifest / id list (e.g. MPS)")
    ap.add_argument("--test_id_field", default="utt_id")
    ap.add_argument("--test_rule", default="whole")
    ap.add_argument("--fuzzy_threshold", type=float, default=0.92,
                    help="difflib ratio above which two normalised keys are "
                         "flagged as a possible same speaker (0 disables)")
    ap.add_argument("--out_train", default="",
                    help="write a copy of --train with overlapping speakers removed")
    ap.add_argument("--report_only", action="store_true")
    ap.add_argument("--out_dir", default="")
    args = ap.parse_args()

    train_rows = load_ids(args.train, args.train_id_field)
    test_rows = load_ids(args.test, args.test_id_field)
    tr_key = make_key_fn(args.train_rule)
    te_key = make_key_fn(args.test_rule)

    train_keys = collections.defaultdict(list)
    tr_unparsed = 0
    for uid, row in train_rows:
        k = tr_key(uid, row)
        if k is None:
            tr_unparsed += 1
        else:
            train_keys[k].append(uid)

    test_keys = collections.defaultdict(list)
    te_unparsed = 0
    for uid, row in test_rows:
        k = te_key(uid, row)
        if k is None:
            te_unparsed += 1
        else:
            test_keys[k].append(uid)

    out = []
    out.append(f"train : {args.train}  ({len(train_rows)} rows, rule={args.train_rule})")
    out.append(f"test  : {args.test}  ({len(test_rows)} rows, rule={args.test_rule})")
    out.append(
        f"speaker keys: train {len(train_keys)} (unparsed {tr_unparsed}), "
        f"test {len(test_keys)} (unparsed {te_unparsed})"
    )
    out.append("")
    out.append("sample train keys: " + ", ".join(sorted(train_keys)[:5]))
    out.append("sample test  keys: " + ", ".join(sorted(test_keys)[:5]))
    out.append("")

    # Refuse to certify a clean result produced by a rule that parsed nothing.
    degenerate = len(train_keys) < 2 or len(test_keys) < 2
    if degenerate:
        out.append(
            "!! One side produced fewer than 2 distinct speaker keys. The "
            "extraction rule is almost certainly wrong for that corpus — fix "
            "--train_rule/--test_rule before trusting any overlap verdict."
        )

    exact = sorted(set(train_keys) & set(test_keys))
    out.append(f"EXACT speaker-key overlaps: {len(exact)}")
    for k in exact[:50]:
        out.append(f"  {k}   train utts={len(train_keys[k])} test utts={len(test_keys[k])}")
    if len(exact) > 50:
        out.append(f"  ... and {len(exact)-50} more")

    norm_train = collections.defaultdict(list)
    for k in train_keys:
        norm_train[norm_key(k)].append(k)
    norm_test = collections.defaultdict(list)
    for k in test_keys:
        norm_test[norm_key(k)].append(k)

    norm_hits = sorted(set(norm_train) & set(norm_test))
    norm_only = [n for n in norm_hits
                 if not (set(norm_train[n]) & set(norm_test[n]))]
    out.append("")
    out.append(
        f"NORMALISED (case/punct-insensitive) overlaps not already exact: {len(norm_only)}"
    )
    for n in norm_only[:25]:
        out.append(f"  {norm_train[n]}  <->  {norm_test[n]}")

    fuzzy = []
    if args.fuzzy_threshold > 0 and not degenerate:
        test_norm_list = sorted(norm_test)
        for tn in sorted(norm_train):
            for cand in difflib.get_close_matches(
                tn, test_norm_list, n=2, cutoff=args.fuzzy_threshold
            ):
                if cand != tn:
                    fuzzy.append((norm_train[tn], norm_test[cand],
                                  difflib.SequenceMatcher(None, tn, cand).ratio()))
    out.append("")
    out.append(f"FUZZY near-matches (ratio >= {args.fuzzy_threshold}): {len(fuzzy)}")
    for a, b, r in fuzzy[:25]:
        out.append(f"  {a}  <->  {b}   ratio={r:.3f}")
    if fuzzy:
        out.append(
            "  These are candidates, not verdicts — eyeball them. Two unrelated "
            "children can have similar ids."
        )

    clean = not exact and not norm_only and not degenerate
    out.append("")
    out.append(
        "VERDICT: no speaker overlap detected"
        if clean
        else "VERDICT: overlap (or an unusable extraction rule) — see above"
    )

    report = "\n".join(out)
    print(report)
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "speaker_overlap.txt"), "w",
                  encoding="utf-8") as fh:
            fh.write(report + "\n")

    if args.out_train and not args.report_only:
        bad = set(exact) | {k for n in norm_only for k in norm_train[n]}
        with open(args.train, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fields = reader.fieldnames
            kept, dropped = [], 0
            for row in reader:
                k = tr_key(row.get(args.train_id_field, ""), row)
                if k in bad:
                    dropped += 1
                else:
                    kept.append(row)
        with open(args.out_train, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(kept)
        print(f"\n[DONE] wrote {args.out_train}: kept {len(kept)}, dropped {dropped}")

    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())
