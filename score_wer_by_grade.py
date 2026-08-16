#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 5 — per-grade scoring, the part the whole exercise is actually for.

Reports, for every grade and for the corpus as a whole:
    WER, CER, and the substitution / deletion / insertion split
    a 95% bootstrap confidence interval on WER (resampling utterances)
    optionally a paired bootstrap p-value against a second system

Two things are specific to this corpus and neither is optional:

1. `--grade_key` picks what "grade" means.
     child       the grade of the child, parsed from the utterance id.
     story_level the grade level of the passage, from the story number
                 (EN-OL-RC-2xx = grade 3 ... 7xx = grade 8).
   These disagree: in the existing B24+B25 test set, 200 utterances are
   grade-5 children reading the grade-6 passage EN-OL-RC-500, and the Kaldi
   `Combined_data_B24_B25/grade_5/` and `grade_6/` directories each contain a
   copy of all 200 — summing the per-grade dirs gives 2004 utterances for a
   1804-utterance test set. Scoring from these manifests avoids the double
   count; say in the paper which key you used.

2. `--devanagari` picks whether code-switched tokens count. They are 2.8% of
   reference tokens and appear in 73% of test utterances, so the choice moves
   the number.
     keep        score them as-is. Requires a multilingual Whisper — a `.en`
                 checkpoint can never emit Devanagari and eats those as errors.
     drop        remove from BOTH sides for an English-only WER comparable
                 against an English-only acoustic model. Removing from the
                 reference only would charge the model an insertion for every
                 non-word it correctly transcribed, so both sides it is.
                 Caveat: this makes those positions free — emitting nothing
                 there scores the same as transcribing the non-word right.
     placeholder collapse every Devanagari token to <L1> on both sides. The
                 model must emit something non-English there but need not match
                 the transcriber's ad-hoc spelling. Best middle ground when the
                 L1 substitutions are part of what you are measuring.

Read the D/I columns, not just WER. Whisper's LM prior tends to substitute a
misread word back to the correct one, which lowers WER while erasing the miscue.
If the fine-tuned model's WER drops but its substitution count drops far faster
than deletions/insertions, check hypotheses by hand before believing it.

Usage:
    python score_wer_by_grade.py \
        --reference manifests/utt_reference_test.csv \
        --hyp       decode/whisper-small-kv-en_test/hyp_utterances.csv \
        --baseline_hyp decode/whisper-small-pretrained_test/hyp_utterances.csv \
        --out_dir   results/whisper-small-kv-en_test
"""

import argparse
import collections
import csv
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from text_normalisation import normalize  # noqa: E402

import jiwer  # noqa: E402


def read_csv(path, key, value_fields):
    out = {}
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[row[key]] = {f: row.get(f, "") for f in value_fields}
    return out


def word_counts(ref, hyp):
    """-> dict with hits/sub/del/ins/ref_len for one pair, jiwer-version safe."""
    if hasattr(jiwer, "process_words"):
        o = jiwer.process_words([ref], [hyp])
        return {
            "hits": o.hits,
            "sub": o.substitutions,
            "del": o.deletions,
            "ins": o.insertions,
            "ref_len": o.hits + o.substitutions + o.deletions,
        }
    m = jiwer.compute_measures(ref, hyp)
    return {
        "hits": m["hits"],
        "sub": m["substitutions"],
        "del": m["deletions"],
        "ins": m["insertions"],
        "ref_len": m["hits"] + m["substitutions"] + m["deletions"],
    }


def char_counts(ref, hyp):
    if hasattr(jiwer, "process_characters"):
        o = jiwer.process_characters([ref], [hyp])
        return (
            o.substitutions + o.deletions + o.insertions,
            o.hits + o.substitutions + o.deletions,
        )
    err = jiwer.cer(ref, hyp) * max(len(ref), 1)
    return err, max(len(ref), 1)


def aggregate(items):
    """items: list of per-utterance count dicts -> corpus-level rates."""
    tot = collections.Counter()
    for it in items:
        for k, v in it.items():
            if k != "utt_id":
                tot[k] += v
    n = max(tot["ref_len"], 1)
    return {
        "n_utts": len(items),
        "ref_words": tot["ref_len"],
        "WER": 100.0 * (tot["sub"] + tot["del"] + tot["ins"]) / n,
        "SUB": 100.0 * tot["sub"] / n,
        "DEL": 100.0 * tot["del"] / n,
        "INS": 100.0 * tot["ins"] / n,
        "CER": 100.0 * tot["cer_err"] / max(tot["cer_len"], 1),
    }


def bootstrap_ci(items, n_boot, seed):
    rng = random.Random(seed)
    n = len(items)
    if n < 2:
        return (float("nan"), float("nan"))
    wers = []
    for _ in range(n_boot):
        sample = [items[rng.randrange(n)] for _ in range(n)]
        s = sum(x["sub"] + x["del"] + x["ins"] for x in sample)
        r = max(sum(x["ref_len"] for x in sample), 1)
        wers.append(100.0 * s / r)
    wers.sort()
    return wers[int(0.025 * n_boot)], wers[int(0.975 * n_boot) - 1]


def paired_bootstrap(a_items, b_items, n_boot, seed):
    """P(system A is not better than B) under resampling of utterances."""
    by_id_b = {x["utt_id"]: x for x in b_items}
    pairs = [(x, by_id_b[x["utt_id"]]) for x in a_items if x["utt_id"] in by_id_b]
    if len(pairs) < 2:
        return float("nan"), len(pairs)
    rng = random.Random(seed)
    n = len(pairs)
    worse = 0
    for _ in range(n_boot):
        sa = sd = ra = rb = 0
        for _ in range(n):
            a, b = pairs[rng.randrange(n)]
            sa += a["sub"] + a["del"] + a["ins"]
            ra += a["ref_len"]
            sd += b["sub"] + b["del"] + b["ins"]
            rb += b["ref_len"]
        if (sa / max(ra, 1)) >= (sd / max(rb, 1)):
            worse += 1
    return worse / n_boot, n


def score_system(refs, hyp_path, norm_mode, devanagari, grade_key):
    hyps = read_csv(hyp_path, "utt_id", ["hyp", "grade", "story_level"])
    per_group = collections.defaultdict(list)
    all_items = []
    missing = 0
    empty_ref = 0

    for utt_id, ref_row in refs.items():
        if utt_id not in hyps:
            missing += 1
            continue
        ref = normalize(ref_row["reference"], norm_mode, devanagari)
        hyp = normalize(hyps[utt_id]["hyp"], norm_mode, devanagari)
        if not ref:
            empty_ref += 1
            continue
        wc = word_counts(ref, hyp)
        cer_err, cer_len = char_counts(ref, hyp)
        item = dict(wc)
        item["cer_err"] = cer_err
        item["cer_len"] = cer_len
        item["utt_id"] = utt_id
        group = ref_row.get(grade_key, "?") or "?"
        per_group[group].append(item)
        all_items.append(item)

    return per_group, all_items, missing, empty_ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, help="manifests/utt_reference_*.csv")
    ap.add_argument("--hyp", required=True, help="decode/*/hyp_utterances.csv")
    ap.add_argument("--baseline_hyp", default="", help="second system to compare against")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--norm", choices=["minimal", "whisper_en"], default="minimal")
    ap.add_argument(
        "--devanagari", choices=["keep", "drop", "placeholder"], default="keep"
    )
    ap.add_argument("--grade_key", choices=["grade", "story_level"], default="grade")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    refs = read_csv(
        args.reference, "utt_id", ["reference", "grade", "story_level", "round", "story"]
    )

    per_group, all_items, missing, empty_ref = score_system(
        refs, args.hyp, args.norm, args.devanagari, args.grade_key
    )

    base_group = base_all = None
    if args.baseline_hyp:
        base_group, base_all, _, _ = score_system(
            refs, args.baseline_hyp, args.norm, args.devanagari, args.grade_key
        )

    lines = []
    lines.append(f"reference   : {args.reference}")
    lines.append(f"hypothesis  : {args.hyp}")
    if args.baseline_hyp:
        lines.append(f"baseline    : {args.baseline_hyp}")
    lines.append(
        f"norm={args.norm}  devanagari={args.devanagari}  grade_key={args.grade_key}"
    )
    lines.append(
        f"utterances scored: {len(all_items)}   missing hypothesis: {missing}   "
        f"empty reference: {empty_ref}"
    )
    lines.append("")

    header = (
        f"{'grade':>7} {'utts':>6} {'refwrds':>8} {'WER%':>7} {'95% CI':>15} "
        f"{'SUB%':>6} {'DEL%':>6} {'INS%':>6} {'CER%':>7}"
    )
    if args.baseline_hyp:
        header += f" {'baseWER%':>9} {'ΔWER':>7} {'p':>6}"
    lines.append(header)
    lines.append("-" * len(header))

    rows_out = []
    for group in sorted(per_group, key=lambda g: (len(g), g)):
        items = per_group[group]
        a = aggregate(items)
        lo, hi = bootstrap_ci(items, args.n_boot, args.seed)
        line = (
            f"{group:>7} {a['n_utts']:>6} {a['ref_words']:>8} {a['WER']:>7.2f} "
            f"{f'[{lo:.2f},{hi:.2f}]':>15} {a['SUB']:>6.2f} {a['DEL']:>6.2f} "
            f"{a['INS']:>6.2f} {a['CER']:>7.2f}"
        )
        rec = {"grade": group, **a, "wer_ci_lo": lo, "wer_ci_hi": hi}
        if args.baseline_hyp:
            b_items = base_group.get(group, [])
            b = aggregate(b_items) if b_items else {"WER": float("nan")}
            p, _ = paired_bootstrap(items, b_items, args.n_boot, args.seed)
            line += f" {b['WER']:>9.2f} {a['WER']-b['WER']:>7.2f} {p:>6.3f}"
            rec["baseline_WER"] = b["WER"]
            rec["delta_WER"] = a["WER"] - b["WER"]
            rec["p_not_better"] = p
        lines.append(line)
        rows_out.append(rec)

    a = aggregate(all_items)
    lo, hi = bootstrap_ci(all_items, args.n_boot, args.seed)
    line = (
        f"{'ALL':>7} {a['n_utts']:>6} {a['ref_words']:>8} {a['WER']:>7.2f} "
        f"{f'[{lo:.2f},{hi:.2f}]':>15} {a['SUB']:>6.2f} {a['DEL']:>6.2f} "
        f"{a['INS']:>6.2f} {a['CER']:>7.2f}"
    )
    rec = {"grade": "ALL", **a, "wer_ci_lo": lo, "wer_ci_hi": hi}
    if args.baseline_hyp:
        b = aggregate(base_all)
        p, npair = paired_bootstrap(all_items, base_all, args.n_boot, args.seed)
        line += f" {b['WER']:>9.2f} {a['WER']-b['WER']:>7.2f} {p:>6.3f}"
        rec["baseline_WER"] = b["WER"]
        rec["delta_WER"] = a["WER"] - b["WER"]
        rec["p_not_better"] = p
    lines.append("-" * len(header))
    lines.append(line)
    rows_out.append(rec)

    lines.append("")
    lines.append(
        "p = paired-bootstrap probability that this system is NOT better than "
        "the baseline; < 0.05 means the improvement holds up under resampling."
    )

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(args.out_dir, "report.txt"), "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    with open(os.path.join(args.out_dir, "per_grade.json"), "w") as fh:
        json.dump(rows_out, fh, indent=2)
    with open(os.path.join(args.out_dir, "per_grade.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"\n[DONE] {args.out_dir}/report.txt, per_grade.csv, per_grade.json")


if __name__ == "__main__":
    main()
