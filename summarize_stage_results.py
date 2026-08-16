#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collect every `results/<model>/mps_<stage>_<devanagari>/per_grade.json` into one
comparison table.

The point of the table is the *contrast*, not any single number:

    stageW  pretrained -> WPP                (children only)
    stageA  pretrained -> IITM               (adult only)
    stageB  stageA -> WPP                    (sequential: adult then children)
    stageC  pretrained -> WPP+IITM pooled    (both at once)

stageB vs stageW is what tells you whether the adult stage helped at all.
stageB vs stageC is sequential vs pooled. Without stageW, a good stageB number
is unattributable — it could be entirely the WPP stage doing the work.
"""

import argparse
import glob
import json
import os
import re

STAGE_ORDER = ["stageW", "stageA", "stageB", "stageC"]
STAGE_DESC = {
    "stageW": "pretrained -> WPP (children only) [control]",
    "stageA": "pretrained -> IITM (adult only)",
    "stageB": "stageA -> WPP (sequential)",
    "stageC": "pretrained -> WPP+IITM pooled (single run)",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--devanagari", default="drop")
    args = ap.parse_args()

    found = {}
    pattern = os.path.join(args.results_root, f"mps_*_{args.devanagari}", "per_grade.json")
    for path in sorted(glob.glob(pattern)):
        m = re.search(r"mps_(\w+?)_" + re.escape(args.devanagari) + r"$",
                      os.path.basename(os.path.dirname(path)))
        if not m:
            continue
        found[m.group(1)] = json.load(open(path))

    if not found:
        print(f"[WARN] no results matched {pattern}")
        return

    lines = [
        "# MPS test-set results",
        "",
        f"Scoring: `--grade_key grade --devanagari {args.devanagari} --norm minimal`.",
        "MPS is grades 3-5; WPP training data is grades 6-10, so every number",
        "below is an out-of-grade-range generalisation result.",
        "",
        "`Δ` is versus the untouched pretrained checkpoint (negative = better).",
        "`p` is the paired-bootstrap probability the system is NOT better than",
        "pretrained; < 0.05 means the gain survives resampling.",
        "",
    ]

    grades = sorted(
        {r["grade"] for rows in found.values() for r in rows},
        key=lambda g: (g == "ALL", g),
    )

    header = "| stage | " + " | ".join(f"G{g}" if g != "ALL" else "ALL" for g in grades) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(grades) + 1))
    for stage in STAGE_ORDER + [s for s in found if s not in STAGE_ORDER]:
        if stage not in found:
            continue
        by_grade = {r["grade"]: r for r in found[stage]}
        cells = []
        for g in grades:
            r = by_grade.get(g)
            cells.append(f"{r['WER']:.1f}" if r else "—")
        lines.append(f"| {stage} | " + " | ".join(cells) + " |")

    lines += ["", "## Detail (ALL utterances)", "",
              "| stage | WER% | 95% CI | SUB | DEL | INS | CER% | Δ vs pretrained | p |",
              "|---|---|---|---|---|---|---|---|---|"]
    for stage in STAGE_ORDER + [s for s in found if s not in STAGE_ORDER]:
        if stage not in found:
            continue
        r = next((x for x in found[stage] if x["grade"] == "ALL"), None)
        if not r:
            continue
        d = r.get("delta_WER")
        p = r.get("p_not_better")
        delta_cell = f"{d:+.2f}" if d is not None else "—"
        p_cell = f"{p:.3f}" if p is not None else "—"
        lines.append(
            f"| {stage} | {r['WER']:.2f} "
            f"| [{r['wer_ci_lo']:.2f}, {r['wer_ci_hi']:.2f}] "
            f"| {r['SUB']:.2f} | {r['DEL']:.2f} | {r['INS']:.2f} | {r['CER']:.2f} "
            f"| {delta_cell} | {p_cell} |"
        )

    lines += ["", "## What each stage is", ""]
    for stage in STAGE_ORDER:
        if stage in found:
            lines.append(f"- **{stage}** — {STAGE_DESC[stage]}")

    lines += [
        "",
        "## Reading the table",
        "",
        "- **stageB vs stageW** — did the adult IITM stage help, or was it all WPP?",
        "  If stageB is not clearly better than stageW, the adult pass bought nothing.",
        "- **stageB vs stageC** — sequential vs pooled at equal data.",
        "- **DEL vs SUB** — a large drop in SUB with flat DEL/INS can mean the model",
        "  learned to repair miscues rather than transcribe them (README §8).",
        "  Read hypotheses before claiming a reading-assessment win.",
        "- **grade columns** — MPS grade 3 readers are the most disfluent and the",
        "  furthest from WPP's grade 6-10 training data; expect the worst WER there.",
    ]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[DONE] {args.out}")


if __name__ == "__main__":
    main()
