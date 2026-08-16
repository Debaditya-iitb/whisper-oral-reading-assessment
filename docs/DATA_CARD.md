# `whisper_finetune/data/` — the train / dev / test sets

Rebuild everything with `bash ../03_build_chunked_datasets.sh`. Nothing here is hand-edited.

```
data/
├── train/   wpp/  iitm/  combined/
├── dev/     wpp/  iitm/  combined/
└── test/    mps/
```

## What's in every leaf

Each leaf holds the **same rows in two views**:

| file | level | what it is |
|---|---|---|
| `wav.scp` | chunk | `<recording> <absolute path to wav>` |
| `segments` | chunk | `<chunk_id> <recording> <start> <end>` — the ≤28 s Whisper units |
| `text` | chunk | `<chunk_id> <transcript for that chunk>` (absent for `test/mps`, see below) |
| `utt2spk` `spk2utt` `utt2dur` | chunk | standard Kaldi |
| `chunks.csv` | chunk | **what the Python scripts actually read** — same rows plus `wav_path`, `grade`, `corpus`, `child_key` |
| `reference/` | utterance | the unit WER is reported on: `wav.scp text utt2spk spk2utt utt2dur utt2grade` + `utt_reference.csv` |

`wav.scp` + `segments` is the standard Kaldi idiom for "a span of a longer
recording", which is exactly what a Whisper chunk is. **No audio was copied** —
every path points at the original corpus in place.

`test/mps/` has no chunk-level `text` on purpose: MPS chunks are cut from audio
energy only (its transcripts carry no timestamps), so the reference lives at
utterance level in `test/mps/reference/text`. Decoding produces one hypothesis
per chunk, `decode_whisper_chunks.py` re-joins them per utterance, and scoring compares
against the whole utterance.

## Contents

| set | chunks | hours | mean | max | speakers | composition |
|---|---|---|---|---|---|---|
| `train/wpp` | 10,849 | 57.2 | 19.0 s | 28.00 s | 1,088 | children, grades 6–10 |
| `train/iitm` | 19,789 | 119.6 | 21.8 s | 28.00 s | 4,952 | adult, long-form **read** |
| `train/combined` | 20,009 | 112.5 | 20.2 s | 28.00 s | 3,341 | wpp 10,849 + iitm 9,160 (capped 60 h) |
| `dev/wpp` | 921 | 4.9 | 19.0 s | 28.00 s | 94 | |
| `dev/iitm` | 1,765 | 10.7 | 21.8 s | 28.00 s | 431 | |
| `dev/combined` | 1,699 | 9.6 | 20.3 s | 28.00 s | 290 | |
| `test/mps` | 3,484 | 19.1 | 19.7 s | 28.00 s | 1,110 | children, grades 3/4/5 |

Mean chunk length is 19–22 s everywhere including the test set — deliberate, so
the model never sees a length distribution at training time that it won't see at
test time.

## Guarantees, all checked at build time

* **No chunk exceeds 28 s.** Verified across all 58,516 chunks. 28 not 30 gives
  headroom against float rounding; Whisper truncates silently at 30 s and this
  ensures nothing ever reaches that.
* **Speaker-disjoint splits.** Train and dev share no speaker in any corpus.
  MPS speakers cannot overlap the training sets: different corpora, different
  years, and disjoint grade bands.
* **Sorted by key**, key sets agree across `text` / `utt2spk` / `segments`, and
  every `segments` recording exists in `wav.scp`.
* **No time overlaps** between chunks of the same recording.
* **Absolute paths**, resolved at build time — none of the stale
  `/path/to/user/…` or `/path/to/shared/…` entries from the
  upstream Kaldi dirs survive.

## IITM: what was selected and why

IITM-English is 184.7 h of three different speaking styles. Only **long-form
read** is used:

| category | recs | hours | mean rec | fillers/1k words | kept |
|---|---|---|---|---|---|
| `*_long_*` long-form read | 5,383 | 127.6 | 85 s | 0.0 | **yes** |
| `NNN_eng_*` short read | 31,776 | 41.9 | 4.7 s | 0.0 | no |
| `IE_vy_*` interviews | 62 | 14.7 | 851 s | 12.7 | no |
| `*_text` podcasts | 2 | 0.6 | 1061 s | 40.6 | no |

* **Conversational dropped** (15.3 h) — spontaneous speech is the wrong speaking
  style for a read-aloud assessment task. The filler-word rate confirms the
  split independently: the read categories contain literally zero fillers.
* **Short reads dropped** (41.9 h) — 4.7 s recordings can't be packed toward the
  30 s window, so they cost the most compute per hour of audio and would train
  the model on fragments far shorter than anything it sees at test time.
* **Segments packed to ≤28 s.** IITM's native Kaldi segments average 5.9 s;
  packing consecutive segments raises the mean to 21.8 s and cuts steps/epoch
  from 3,245 to 620 — a ~5× speedup on the same data. Gaps between segments are
  silence the transcribers skipped, so crossing them is safe.

Change any of this with `IITM_KEEP=long_read,short_read bash ../03_build_chunked_datasets.sh`
or `--iitm_no_merge`.

One caveat carried over from the source: IITM's own train/dev split is
**segment-level**, so all 2,414 of its dev recordings also appear in its train
list. That split is discarded here and redone speaker-disjointly by recording.
