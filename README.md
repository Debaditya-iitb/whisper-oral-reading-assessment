# Whisper fine-tuning for children's oral-reading assessment

Fine-tuning OpenAI Whisper (`small`, 244 M and `medium`, 769 M) on Indian
children's read-aloud English, evaluated on the **MPS** test set with a
per-grade, bootstrap-tested WER breakdown.

The task is **not** "get the lowest WER". The references are *verbatim miscue
transcripts* — they record what the child actually said, misreadings included.
A model that silently repairs a misread word into the word the child was
*supposed* to read scores a better WER while being **worse** at the actual job.
Every design decision below follows from that, and the evaluation is built to
expose it.

**Headline result** (MPS, 1,600 utterances, `devanagari=drop`, vs the untouched
pretrained checkpoint):

| model | pretrained | best fine-tune | Δ WER | which stage |
|---|---|---|---|---|
| `whisper-medium` | 12.52 % | **9.79 %** | −2.73 | stage W (children only) |
| `whisper-small` | 13.62 % | **12.26 %** | −1.36 | stage W (children only) |

Both gains survive a paired bootstrap (p < 0.001). The two more elaborate
recipes — adult-speech pretraining first (stage B) and pooled adult + child
data (stage C) — **did not beat the simple children-only fine-tune** at either
model size. Every number is in
[`results/mps_results_all_systems.csv`](results/mps_results_all_systems.csv).

---

## Contents

- [What is and isn't here](#what-is-and-isnt-here)
- [Quick start](#quick-start)
- [Files](#files)
- [The four-stage experiment](#the-four-stage-experiment)
- [The data](#the-data)
- [The 30-second problem](#the-30-second-problem)
- [The test set: MPS](#the-test-set-mps)
- [Vocabulary, tokens, and the phone-set question](#vocabulary-tokens-and-the-phone-set-question)
- [Scoring](#scoring)
- [Hyperparameters, and why not the alternatives](#hyperparameters-and-why-not-the-alternatives)
- [Results](#results)
- [Reading the results honestly](#reading-the-results-honestly)
- [Reproducing](#reproducing)
- [Limitations](#limitations)

---

## What is and isn't here

**Here:** the whole pipeline, environment pins, the data card, the long-form
design notes, and one results CSV backing every number in this file.

**Not here:** audio, transcripts, model weights, arrow datasets. None of it is
redistributable — WPP/KV is children's personal data; IITM and MPS have their
own licences. `.gitignore` enumerates what is withheld. Nothing depends on
hidden state: point the scripts at the corpora and everything regenerates.

---

## Quick start

```bash
bash 01_setup_environment.sh                        # venv/conda + pinned deps
bash 02_download_pretrained_models.sh whisper-medium

AUDIO_ROOT=/path/to/audio bash 03_build_chunked_datasets.sh
MODEL_TAG=whisper-medium bash 04_build_arrow_datasets.sh

MODEL_TAG=whisper-medium bash 05_finetune_all_stages.sh
MODEL_TAG=whisper-medium bash 06_evaluate_on_mps_testset.sh
```

Or `bash run_full_pipeline.sh` for all six in order. Every step is re-runnable
and skips work already done, so this is also the right thing to run after a
crash.

---

## Files

Numbered scripts are the pipeline, in run order. Everything else is named for
what it does.

### Pipeline

| file | task |
|---|---|
| `01_setup_environment.sh` | build the Python env from `requirements.txt`; installs CUDA torch **first** so nothing later resolves a CPU wheel |
| `02_download_pretrained_models.sh` | fetch pretrained Whisper checkpoints into `pretrained/`; everything downstream points at a **local** dir so the GPU node needs no internet |
| `03_build_chunked_datasets.sh` | end-to-end data build: manifests → chunk tables → Kaldi dirs → verification |
| `04_build_arrow_datasets.sh` | chunk tables → HuggingFace arrow datasets (one set per model size) |
| `05_finetune_all_stages.sh` | run stages W, A, B, C with their per-stage LR / epoch / warm-up schedule |
| `06_evaluate_on_mps_testset.sh` | decode pretrained + all four stages, score under all three Devanagari conventions, write the summary |
| `run_full_pipeline.sh` | all of the above in order |

### Configuration

| file | task |
|---|---|
| `config_paths.sh` / `config_paths.py` | **the only place machine-specific paths live.** Everything imports from here, so moving the project is a one-file edit |
| `config_experiment.sh` | model tag, stage → output-dir mapping, GPU-aware batch sizing |

### Data preparation

| file | task |
|---|---|
| `build_chunk_manifests.py` | the core data logic: read time-aligned label tracks, cut **audio and transcript together** into ≤ 28 s chunks |
| `prepare_mps_test_set.py` | MPS-specific: cut **audio only** at low-energy frames (its transcripts carry no timestamps), leaving the reference whole |
| `build_arrow_datasets.py` | chunk tables → arrow datasets; reads only `[start, end]` per wav, never copies or cuts audio on disk |
| `copy_audio_to_local_disk.py` | move a corpus to local disk and rewrite manifests to match |
| `make_audio_transfer_list.py` | list exactly which files a transfer needs |
| `export_kaldi_dirs.py` | re-express the same splits in Kaldi layout for cross-checking |

### Training, decoding, scoring

| file | task |
|---|---|
| `finetune_whisper_seq2seq.py` | the training recipe: `Seq2SeqTrainer`, `predict_with_generate=True`, early stopping on dev WER |
| `decode_whisper_chunks.py` | decode a chunk manifest, collapse n-gram loops, **re-join chunks into one hypothesis per utterance** |
| `score_wer_by_grade.py` | WER/CER/SUB/DEL/INS per grade, 95 % CI, paired bootstrap vs the baseline |
| `text_normalisation.py` | the shared normaliser — minimal by design (see [Scoring](#scoring)) |
| `summarize_stage_results.py` | collect every stage's scores into one table |

### Checks and cluster

| file | task |
|---|---|
| `check_paths_resolve.py` | every path resolves, before any GPU time is spent |
| `check_speaker_overlap.py` | no speaker crosses a split |
| `slurm_prepare_data.sh`, `slurm_finetune.sh`, `slurm_evaluate.sh`, `slurm_submit_all_stages.sh` | Slurm wrappers for the same steps |
| `tool_fetch_pretrained_checkpoint.sh` | single-checkpoint fetch helper |

### Documentation

| file | contents |
|---|---|
| [`docs/DESIGN_NOTES.md`](docs/DESIGN_NOTES.md) | the long-form notebook: alternatives considered, failure modes, corpus-by-corpus notes. This README is the summary; that file is the reasoning |
| [`docs/DATA_CARD.md`](docs/DATA_CARD.md) | what is in every split, the build-time guarantees, and why only IITM long-form read was used |
| [`docs/SERVER_SETUP.md`](docs/SERVER_SETUP.md), [`docs/PORTING.md`](docs/PORTING.md) | moving the project between machines |

---

## The four-stage experiment

A 2×2 on *what data* and *in what order*, with a real control:

| stage | init from | trains on | question it answers |
|---|---|---|---|
| **W** | pretrained | WPP (children) | **the control.** How far does child data alone get you? |
| **A** | pretrained | IITM (adult read) | does adult read-aloud speech help at all? |
| **B** | **stage A** | WPP (children) | does an adult warm-up *then* children beat children alone? |
| **C** | pretrained | WPP + IITM pooled | sequential vs pooled at equal data |

B depends on A; W and C depend on nothing. The default order `W A B C` runs the
cheapest, most informative stage first, so a mistake surfaces in 15 minutes
rather than after the adult stage.

---

## The data

| set | chunks | hours | speakers | mean chunk | composition |
|---|---|---|---|---|---|
| `train/wpp` | 10,849 | 57.2 | 1,088 | 19.0 s | children, **grades 6–10** |
| `train/iitm` | 19,789 | 119.6 | 4,952 | 21.8 s | adults, long-form **read** |
| `train/combined` | 20,009 | 112.5 | 3,341 | 20.2 s | wpp + iitm capped at 60 h |
| `dev/wpp` | 921 | 4.9 | 94 | 19.0 s | speaker-disjoint from train |
| `dev/iitm` | 1,765 | 10.7 | 431 | 21.8 s | |
| `dev/combined` | 1,699 | 9.6 | 290 | 20.3 s | |
| `test/mps` | 3,484 | 19.1 | 1,110 | 19.7 s | children, **grades 3/4/5** |

WPP grade distribution (chunks): G6 2,224 · G7 2,606 · G8 2,541 · G9 3,328 ·
G10 150.

Two properties enforced at build time:

- **Splits are speaker-disjoint.** No speaker crosses train/dev in any corpus.
- **The chunk-length distribution matches at train and test time** (19–22 s
  mean everywhere), so the model never meets a length regime in evaluation it
  did not meet in training.

Only IITM's **long-form read** category is used (127.6 h of its 184.7 h).
Conversational speech is the wrong speaking style for a read-aloud task, and
its 4.7 s short reads cannot be packed toward the 30 s window. The filler-word
rate confirms the split independently: the read categories contain literally
zero fillers, the interviews 12.7 per 1,000 words. Full reasoning in
[`docs/DATA_CARD.md`](docs/DATA_CARD.md).

### The 30-second problem

Whisper's encoder has a **fixed** 3,000-frame / 30-second receptive field, and
`WhisperFeatureExtractor` pads *or truncates* to exactly that — silently.

MPS utterances average 42.9 s. Feeding one in whole means the encoder sees
seconds 0–30, the label covers all 42.9 s, and **~30 % of every target has no
acoustic evidence behind it**. Cross-entropy still forces the decoder to emit
those tokens, and the only place it can get them is its own language model.
That is a direct instruction to hallucinate, and it fails with no error.

`build_chunk_manifests.py` avoids it by cutting **audio and transcript together
in one operation**. The label tracks are already time-aligned per breath group,
so a cut at a line boundary cuts waveform and text at the same instant, by
construction — no forced aligner, no separate text segmentation to reconcile.
Lines accumulate while `last.end − first.start ≤ 28 s`, and **a line is never
split**. 28 s, not 30 s, for headroom against float rounding. Verified across
all 58,516 chunks: zero over 28 s, zero dropped.

`decode_whisper_chunks.py` re-joins chunk hypotheses in start-time order, so
**the evaluation unit is the whole utterance**. Chunking is an internal detail
of the encoder window, not a change to what WER is computed on.

---

## The test set: MPS

[`DAP-Lab/mps_dataset`](https://github.com/DAP-Lab/mps_dataset) (Interspeech
2024) — children's read-aloud English from Maharashtra and Goa government
schools, collected summer 2023.

A strong external test set here because it shares the training corpora's lab
and annotation pipeline: `manualTranscript` is **verbatim miscue transcription**
with the same convention, the same tag set (SIL/BR/ON/FP/IR/MB/WH/HS — all
already in `text_normalisation.LABEL_TAGS`), and the same Devanagari convention
for intelligible-but-invalid English words.

### Analytics (measured from the built test set)

| | grade 3 | grade 4 | grade 5 | **all** |
|---|---|---|---|---|
| utterances | 499 | 504 | 597 | **1,600** |
| speakers | 346 | 351 | 413 | **1,110** |
| stories | 1 | 1 | 1 | **3** |
| audio (h) | 5.87 | 6.33 | 6.88 | **19.08** |
| reference words | 32,196 | 35,899 | 45,422 | **113,517** |
| words / utterance | 64.5 | 71.2 | 76.1 | **71.0** |
| seconds / utterance | 42.4 | 45.2 | 41.5 | **42.9** |
| Devanagari tokens | 1,872 | 1,706 | 1,532 | **5,110** |
| Devanagari token rate | 5.81 % | 4.75 % | 3.37 % | **4.50 %** |
| utterances containing ≥ 1 | 88.6 % | 88.5 % | 90.8 % | **89.4 %** |

- **Gender**: 871 male / 729 female utterances.
- **Chunks**: 3,484, mean 19.7 s, median 21.0 s, max 28.0 s, **zero** over 30 s;
  2.18 chunks/utterance, at most 3.
- **Utterance duration**: mean 42.9 s, max 61.6 s — *every* utterance exceeds
  the encoder window, so chunking is mandatory even for inference-only use.
- **Word types**: 2,960 over 113,517 running words. Only **three stories** (one
  per grade), so the lexicon is small and heavily repeated: `the` (8,315),
  `a` (3,501), `of` (3,409), `and` (2,784), `to` (2,692), `in` (2,661),
  `he` (2,559), `they` (2,524), `are` (2,217), `sheep` (2,152).
- **Devanagari types**: 2,088 distinct spellings for 5,110 tokens — a
  type/token ratio of 0.41 against 0.008 on the English side, because these are
  transcribers' ad-hoc spellings of non-words. Commonest: `क्रिएचर्स` (207),
  `बश` (173), `किप्ट` (90), `शेपड` (76), `क्रिएचर` (75), `सीप` (70),
  `लिज़ार्ड` (68), `लिजाड` (66), `युजली` (66), `सिप` (59) — attempts at
  *creatures*, *bush*, *kept*, *shepherd*, *sheep*, *lizard*, *usually*.

The grade-3 → grade-5 gradient in Devanagari rate (5.81 % → 3.37 %) is the
readability signal itself: younger readers fall back to L1 phonology more often.

### Two structural facts

**Speaker overlap is impossible, and unverifiable by id.** MPS speaker ids are
anonymised five-character hashes with no school or name fields — ethics
clearance was granted on that basis, so `check_speaker_overlap.py` *cannot*
confirm disjointness by id, and a "0 overlaps" result from it would be
meaningless. The argument that holds is structural: MPS was collected in 2023
from grades 3–5, WPP in 2020–21 from grades 6–10. A WPP child would have been
in grade 8–13 by 2023. Disjoint grade bands, two to three years apart.

**Train and test grades do not overlap — the main threat to the result.** WPP
is grades 6–10, MPS grades 3–5, so every number here is **out-of-grade-range
generalisation**. MPS describes its readers as L2 learners with "non-existent
(or very limited)" English vocabulary — weaker than WPP's older cohort. The KV
corpus covers grades 3–8, exactly the MPS range; adding it is the obvious next
ablation.

---

## Vocabulary, tokens, and the phone-set question

### There is no phone set here, and that is not an oversight

Whisper is an encoder–**decoder** emitting **orthographic BPE tokens**. It
cannot produce a phone sequence, so this pipeline has no phone inventory, no
lexicon, no `vocab.json` to build, no PER — and no per-frame posteriors, hence
no PRC-style confidence score.

That is a real capability loss versus a CTC system, and it is why this repo
*complements* rather than replaces phone-level wav2vec2/Kaldi work. The closest
Whisper analogue to PRC is mean token log-probability — a **different
quantity**, differently calibrated. Do not substitute one into a threshold
tuned for the other.

### The vocabulary that is actually used

Whisper's **fixed multilingual BPE vocabulary**: 50,258 base tokens, 51,865
including specials. Nothing is resized, extended or retrained; the same
tokenizer serves `small` and `medium`. Never use a `.en` checkpoint — the
English-only vocabularies cannot represent Devanagari at all.

Measured on the MPS references:

| | types | mean BPE tokens / word | max |
|---|---|---|---|
| English words | 872 | **1.08** | 4 |
| Devanagari words | 2,088 | **7.14** | 19 |

A Devanagari miscue costs **6.6× more tokens** than an English word, worst case
19 tokens for one word (`मॅग्निफाइगिंग`, an attempt at *magnifying*). Those
tokens are individually near-unpredictable, so the model pays a large
cross-entropy cost to learn them and rarely emits them at inference — which is
the mechanism behind the `keep`/`placeholder` result below.

Reference lengths sit comfortably inside the generation limit: mean 92 tokens
per utterance, median 90, max 154, against `GENERATION_MAX_LENGTH=225`. Nothing
is truncated, and chunks — what is actually generated — are shorter still.

---

## Scoring

`score_wer_by_grade.py` reports, per grade and overall: WER with a 95 %
bootstrap CI, SUB/DEL/INS, CER, and a **paired bootstrap** p-value against the
baseline. `p` is the probability the system is *not* better than pretrained;
`p < 0.05` means the gain survives resampling.

**Normalisation is deliberately minimal**: NFKC, lowercase, strip punctuation,
collapse whitespace. Nothing lexical. Whisper's own `EnglishTextNormalizer`
expands contractions, maps *gonna → going to*, drops fillers and rewrites
number words — every one of those can destroy a miscue. It is available as
`--norm whisper_en` only so a comparable-to-published number can be reported
alongside, never as the primary metric.

**Three Devanagari conventions, all reported:**

| mode | what it does | when it's right |
|---|---|---|
| `keep` | score the code-switched token as written | strictest; grades the transcriber's ad-hoc spelling as much as the model |
| `drop` | delete Devanagari tokens from **both** sides | English-only WER, comparable against an English-only model. Those positions become free — but a model that "repairs" the non-word into the target English word still pays, as an insertion. **The headline number.** |
| `placeholder` | replace every Devanagari token with `<L1>` on both sides | usually the most meaningful for reading assessment: the model must emit *something* non-English there, without being graded on spelling |

Whatever is chosen for scoring, Devanagari tokens stay in the **training**
targets. Stripping them leaves audio with no corresponding text, which trains
the decoder to skip real speech.

---

## Hyperparameters, and why not the alternatives

| setting | value | why this, not the alternative |
|---|---|---|
| `LEARNING_RATE` | **1e-5** | Whisper fine-tunes run ~10× below CTC LRs. At 3e-5+ (what the wav2vec2 recipe uses) the decoder forgets and emits fluent hallucination — catastrophic here, because fluent output *is* the failure mode. Change this one last. |
| effective batch | **32** everywhere | matches the wav2vec2 recipe so the comparison is architecture, not batch size. Held constant across GPU types by trading per-device batch against accumulation. |
| `NUM_TRAIN_EPOCHS` | 10 (W, B), 8 (C), 6 (A) | a ceiling, not a target — early stopping decides. Every run stopped between epoch 5 and 7. |
| `EARLY_STOPPING_PATIENCE` | 3, on **dev WER** | not dev loss: on seq2seq the two decouple badly — loss keeps falling while WER flattens or worsens. |
| `WARMUP_RATIO` | 0.05 (0.02 for stage B) | a ratio, not fixed steps, so it stays correct when the pool changes size. Stage B starts from an adapted checkpoint, so a full-strength warm-up washes stage A back out. |
| stage B LR | **7.5e-6** | a second full-strength pass at 1e-5 erases what stage A bought — that is what makes B a genuine transfer test rather than "stage W with extra steps". |
| `WEIGHT_DECAY` | 0.0 | Whisper's own fine-tuning default; the regularisation here is early stopping, and adding decay confounds the stage comparison. |
| `LABEL_SMOOTHING` | **0.0** | smoothing rewrites the loss toward "plausible" text — exactly backwards on verbatim miscue targets. |
| `DROPOUT` | 0.0 | Whisper's default. Raise to 0.1 only if dev WER diverges from train loss; it did not. |
| `FREEZE_ENCODER` | **False** | the encoder is what adapts to Indian-accented child speech. Freezing is cheaper and keeps the front-end intact but adapts far less — only worth it if VRAM forces it. |
| `GRADIENT_CHECKPOINTING` | True | ~40 % VRAM for ~20 % slower steps; what lets `medium` run at this batch size on one card. |
| precision | **bf16** | H100/A100. On pre-Ampere cards the script detects and switches to fp16. |
| `EVAL_NUM_BEAMS` | **1 (greedy)** | beam search leans harder on the internal LM and is measurably more likely to repair a misread word. Greedy also matches decode-time settings, so dev WER predicts test WER. |
| `condition_on_prev_tokens` | **False** | chunks are scored independently; conditioning lets one bad chunk drag a whole utterance into a hallucination loop. |
| `temperature` | 0.0 at decode | no sampling, no temperature fallback — fallback silently retries with a different strategy and makes results irreproducible. |
| `forced_decoder_ids` | cleared; language/task set on `generation_config` | leaving the checkpoint's own ids in place silently overrides `LANGUAGE`. A classic Whisper bug that yields a model quietly decoding as the wrong language. |
| `GENERATION_MAX_LENGTH` | 225 | Whisper's default, and 1.5× the longest reference (154 tokens). |

### Batch sizing is GPU-aware; effective batch is not

`config_experiment.sh` picks per-device batch × accumulation from the detected
GPU so the **effective batch stays 32** everywhere:

| GPU class | `whisper-small` | `whisper-medium` | `large-v3` |
|---|---|---|---|
| H100 / H200 | 32 × 1 | 16 × 2 | 8 × 4 |
| A100-80G / L40S | 16 × 2 | 8 × 4 | 4 × 8 |
| ≤ 48 GB | 8 × 4 | 4 × 8 | 2 × 16 |

Under `torchrun` the per-device batch is divided by the GPU count, so the global
effective batch is still 32. Only step count and wall time change — never the
recipe.

### Model size

`small` and `medium` are both fully fine-tuned. `large-v3` is not: it does not
fit for full fine-tuning on one 80 GB card, it uses **128 mel bins** where
small/medium use 80 (so arrow datasets are built per model tag and are *not*
interchangeable), and `large-v3-turbo`'s distilled 4-layer decoder makes the
auto-correction problem *worse* — the wrong direction for miscue work.

---

## Results

All numbers: MPS, 1,600 utterances, 108,408 scored reference words
(`devanagari=drop`; 113,540 under `keep`/`placeholder`), `norm=minimal`, greedy
decoding. Machine-readable:
[`results/mps_results_all_systems.csv`](results/mps_results_all_systems.csv)
— 120 rows, `model × system × devanagari_mode × grade`.

### whisper-medium — pretrained baseline 12.52 %

| stage | WER % | 95 % CI | SUB | DEL | INS | CER % | Δ | p |
|---|---|---|---|---|---|---|---|---|
| **W** children only | **9.79** | [9.53, 10.06] | 5.10 | 1.59 | 3.10 | 7.45 | **−2.73** | 0.000 |
| A adult only | 14.17 | [13.84, 14.53] | 8.33 | 1.38 | 4.46 | 10.31 | +1.65 | 1.000 |
| B adult → children | 10.15 | [9.89, 10.39] | 5.34 | 1.70 | 3.11 | 7.75 | −2.37 | 0.000 |
| C pooled | 11.42 | [11.14, 11.71] | 6.05 | 1.93 | 3.43 | 8.60 | −1.10 | 0.000 |

Per grade (baseline G3 14.98 / G4 12.89 / G5 10.53):

| stage | G3 | G4 | G5 |
|---|---|---|---|
| **W** | **8.55** (−6.42) | **11.17** (−1.72) | **9.56** (−0.97) |
| A | 15.16 (+0.18, p=0.79) | 15.15 (+2.26) | 12.73 (+2.20) |
| B | 9.10 (−5.87) | 11.31 (−1.59) | 9.96 (−0.57) |
| C | 10.69 (−4.28) | 13.09 (+0.20, p=0.78) | 10.62 (+0.09, p=0.69) |

Training, 2 × H100 under torchrun, effective batch 32:

| stage | best epoch | dev WER | dev set | train time |
|---|---|---|---|---|
| W | 7.0 | 8.96 % | wpp_dev | 43 min |
| A | 6.0 | 12.12 % | **iitm_dev** | 68 min |
| B | 7.0 | 8.98 % | wpp_dev | 33 min |
| C | 5.0 | 8.88 % | wpp_dev | 37 min |

Stage A's dev WER is on the adult dev set and is **not comparable** to the
other three.

### whisper-small — pretrained baseline 13.62 %

| stage | WER % | 95 % CI | SUB | DEL | INS | CER % | Δ | p |
|---|---|---|---|---|---|---|---|---|
| **W** | **12.26** | [11.97, 12.57] | 6.07 | 3.18 | 3.01 | 9.69 | **−1.36** | 0.000 |
| A | 18.19 | [17.79, 18.60] | 11.93 | 1.78 | 4.47 | 12.07 | +4.57 | 1.000 |
| B | 12.74 | [12.44, 13.03] | 6.16 | 3.55 | 3.03 | 10.31 | −0.88 | 0.000 |
| C | 12.86 | [12.53, 13.17] | 6.37 | 3.12 | 3.37 | 10.19 | −0.76 | 0.002 |

Per grade (baseline G3 17.80 / G4 13.61 / G5 10.73) — **the most important
table in this file**:

| stage | G3 | G4 | G5 |
|---|---|---|---|
| W | 10.83 (−6.96) | 14.93 (**+1.32**, p=0.99) | 11.16 (**+0.43**, p=0.99) |
| A | 20.79 (+2.99) | 21.17 (+7.56) | 14.06 (+3.33) |
| B | 10.83 (−6.96) | 15.93 (**+2.32**, p=1.00) | 11.56 (**+0.83**, p=1.00) |
| C | **10.50** (−7.30) | 15.17 (**+1.56**, p=1.00) | 12.69 (**+1.96**, p=1.00) |

Every `whisper-small` fine-tune improves grade 3 by 6–7 points and makes grades
4 and 5 *worse*. The aggregate −1.36 is real, but grade 3 carries the entire
result.

Training, 2 × H100, effective batch 32:

| stage | best epoch | dev WER | dev set | train time |
|---|---|---|---|---|
| W | 6.0 | 9.46 % | wpp_dev | 14 min |
| A | 6.0 | 16.29 % | **iitm_dev** | 35 min |
| B | 5.0 | 9.49 % | wpp_dev | 11 min |
| C | 6.0 | 9.50 % | wpp_dev | 19 min |

### All three Devanagari conventions (overall WER %)

| | medium | | | small | | |
|---|---|---|---|---|---|---|
| stage | drop | keep | placeholder | drop | keep | placeholder |
| pretrained | 12.52 | 13.28 | 13.28 | 13.62 | 14.23 | 14.23 |
| **W** | **9.79** | 11.20 | 10.10 | **12.26** | 14.15 *(p=0.38)* | 12.60 |
| A | 14.17 | 14.69 | 14.69 | 18.19 | 18.55 | 18.55 |
| B | 10.15 | 11.60 | 10.48 | 12.74 | 14.57 *(p=0.92)* | 13.12 |
| C | 11.42 | 12.73 | 11.71 | 12.86 | 14.56 *(p=0.91)* | 13.21 |

Under the strictest convention (`keep`), **no `whisper-small` fine-tune is
significantly better than pretrained**.

### medium vs small

| | small (244 M) | medium (769 M) |
|---|---|---|
| pretrained WER | 13.62 | 12.52 |
| best fine-tuned WER | 12.26 | **9.79** |
| gain | −1.36 | **−2.73** |
| holds under `keep`? | **no** (p = 0.38) | yes (−2.08, p < 0.001) |
| improves all three grades? | **no** (G4, G5 regress) | yes |
| deletion rate, stage W | 3.18 | 1.59 |
| train time, stage W | 14 min | 43 min |
| decode, 19.1 h audio | 115 s (RTF 0.0017) | 138 s (RTF 0.0020) |

Doubling wall-clock cost buys roughly twice the WER gain and — more importantly
— the difference between a result that holds under every scoring convention and
one that does not. **Use `medium`.** `small` is for pipeline smoke tests.

### Decode health

RTF ≈ 0.002 (19.1 h in 115–176 s on one H100). N-gram loops, the classic
Whisper failure, are essentially absent: 0–5 looped chunks of 3,484 for medium,
1–7 for small, all collapsed by `decode_whisper_chunks.py`. Zero missing
hypotheses, zero empty references, in every run.

---

## Reading the results honestly

**1. Adult read-aloud speech does not transfer to children.** Stage A is worse
than not fine-tuning at all: +1.65 WER (medium), +4.57 (small), p = 1.000 both.
119.6 h of clean adult read speech actively damages performance on 8–11-year-old
L2 readers. Speaking style and speaker age dominate data volume.

**2. The adult warm-up buys nothing.** Stage B is *worse* than the
children-only control at both sizes (10.15 vs 9.79; 12.74 vs 12.26), and
pooling (C) is worse still. The recipe this experiment was designed to test
does not pay off. The simplest recipe wins.

**3. The aggregate hides a grade split for `small`.** G3 improves 6–7 points;
G4 and G5 regress at p ≈ 1.0. Since the model never saw grades 3–5 in training,
what fine-tuning teaches is the *disfluency and L1-interference patterns of
weak readers* — helping most where readers are weakest, hurting where they are
already near the pretrained model's comfort zone. Medium is better behaved but
shows the same gradient (−6.42 / −1.72 / −0.97).

**4. Watch SUB/DEL/INS, not just WER.** This is the miscue-repair check. For
medium stage W, substitutions fall while insertions hold at 3.10 — consistent
with genuinely better transcription rather than fluent repair. For `small`,
deletions are twice medium's (3.18 vs 1.59), and in grade 4 stage B the
deletion rate reaches 6.38 %, which is most of why that cell regresses. **A WER
gain driven by falling SUB with rising INS would be the repair failure mode** —
read hypotheses before claiming a reading-assessment win.

**5. `keep` vs `placeholder` is diagnostic on its own.** For pretrained and for
stage A the two are *identical* (13.28/13.28 and 14.69/14.69 for medium). Not a
coincidence: neither model ever emits Devanagari — pretrained because it decodes
English, stage A because IITM contains 0 % Devanagari — so both conventions
penalise them equally. Every WPP-trained model does better under `placeholder`
than `keep`, meaning it *has* learned to emit something non-English at those
positions, just not the transcriber's exact spelling. For reading assessment
that detection is the useful behaviour, and `placeholder` is the fairest
headline metric.

**6. Everything is out-of-grade-range generalisation.** Train 6–10, test 3–5.
The obvious next experiment is adding grade-matched KV data (grades 3–8) and
rerunning stage W — that ablation separates grade-range artefact from model
limitation.

---

## Reproducing

You need the three corpora; none ship with this repo.

| corpus | how to get it |
|---|---|
| **MPS** (test) | public: `git clone https://github.com/DAP-Lab/mps_dataset` (~1.9 GB) |
| **IITM-English** | IIT Madras release; institutional licence |
| **WPP / KV** | DAP-Lab internal; children's personal data, not redistributable |

```bash
bash 01_setup_environment.sh
bash 02_download_pretrained_models.sh whisper-small whisper-medium

AUDIO_ROOT=/path/to/audio bash 03_build_chunked_datasets.sh
python prepare_mps_test_set.py --mps_root /path/to/mps_dataset
python check_paths_resolve.py        # every path resolves before any GPU time
python check_speaker_overlap.py      # no speaker crosses a split

MODEL_TAG=whisper-medium bash 04_build_arrow_datasets.sh
MODEL_TAG=whisper-medium bash 05_finetune_all_stages.sh
MODEL_TAG=whisper-medium bash 06_evaluate_on_mps_testset.sh
```

`config_paths.sh` / `config_paths.py` is the only file to edit for a new
machine — corpus roots also read from `WPP_ROOT`, `IITM_ROOT`, `MPS_ROOT`,
`KV_ROOT` if exported. Every stage writes `run_config.json` next to its
checkpoint recording exactly what produced it.

Cost to reproduce both models end to end: about **4.3 hours wall clock on
2 × H100** for training (medium 3.0 h, small 1.3 h, four stages each), plus data
preparation and ~12 minutes of decoding.

---

## Limitations

- **Out-of-grade-range evaluation.** Train 6–10, test 3–5, no overlap. The
  grade-matched KV data exists and is unused.
- **Three stories.** One passage per grade, so the test lexicon is small and
  repeated. WER here measures acoustic and disfluency modelling, not vocabulary
  coverage.
- **No phone-level output.** No PER, no frame posteriors, no PRC.
- **Devanagari spellings are transcriber-dependent** (2,088 types for 5,110
  tokens), so `keep` grades transcription convention as much as model quality.
- **Single seed.** One run at seed 42; the bootstrap CIs cover test-set
  sampling, not training-run variance.
- **The miscue-repair question is not settled.** The SUB/DEL/INS breakdown is
  consistent with genuine improvement, but confirming it needs a miscue-level
  comparison against MPS's word-level `textAlignment` labels (Cor/Sub/Ins/Del),
  which ship with the dataset and are not yet used here.
