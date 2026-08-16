#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3 — Whisper seq2seq fine-tuning recipe for the KV oral-reading corpus.

Deliberate differences from `retrain_prc/finetune_w2v2_recipe.py`, because
Whisper is an encoder-decoder LM, not a CTC head:

* `Seq2SeqTrainer` + `predict_with_generate=True`. CTC scores by argmax over
  frames; Whisper has to actually run autoregressive generation to produce a
  hypothesis, so dev evaluation is ~10x slower than the wav2vec2 one. Keep the
  dev set small (the manifest builder gives you ~2.9 h / 532 chunks).
* Learning rate 1e-5 (not 3e-5). Whisper fine-tunes are run an order of
  magnitude below the CTC LR; at 3e-5+ the decoder starts to forget and you get
  hallucinated fluent text.
* No SpecAugment-style masking knobs. Whisper's config has none of
  `mask_time_prob` / `layerdrop` in the wav2vec2 sense; regularisation here is
  dropout + early stopping + (optionally) freezing the encoder.
* `metric_for_best_model="eval_wer"`. There is no phone vocabulary and no PER —
  Whisper emits orthographic words.

THE ONE THING TO WATCH ON THIS TASK
-----------------------------------
Whisper has a very strong internal language model. On disfluent child reading it
will happily "repair" what the child actually said back into fluent English —
which is precisely the signal you are trying to measure. A model that silently
corrects miscues can post a *better* WER while being *worse* for reading
assessment. That is why `decode_whisper_chunks.py` reports the insertion/deletion/
substitution breakdown and a deletion-rate check, and why greedy decoding
(num_beams=1) is the default there. Always look at that breakdown, not just WER.
"""

import argparse
import inspect
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import numpy as np
import torch
from datasets import load_from_disk
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from text_normalisation import normalize  # noqa: E402
import config_paths as paths  # noqa: E402

# ==============================================================
# USER CONFIG — edit these per fine-tuning round.
# Keep DEV_DATASET_DIR fixed across rounds so eval_wer stays comparable.
# ==============================================================

# Defaults are relative to this folder (config_paths.py), so they follow it anywhere.
# slurm_finetune.sh overrides all four via CLI.
TRAIN_DATASET_DIR = os.path.join(paths.HF_DATASETS, "whisper-medium", "wpp_train")
DEV_DATASET_DIR   = os.path.join(paths.HF_DATASETS, "whisper-medium", "wpp_dev")
BASE_MODEL_DIR    = os.path.join(paths.PRETRAINED, "whisper-medium")
OUTPUT_DIR        = os.path.join(paths.MODELS, "whisper-medium", "stageW_wpp_only")

LANGUAGE = "english"
TASK     = "transcribe"
SEED     = 42

# --- batch / schedule ---
# small  on 1x A100-80G: per_device 16, accum 2   -> effective 32
# medium on 1x A100-80G: per_device  8, accum 4   -> effective 32
# large-v3 full FT does not fit on one 80G card; use FREEZE_ENCODER=True or LoRA.
PER_DEVICE_TRAIN_BATCH  = 16
GRAD_ACCUM_STEPS        = 2
PER_DEVICE_EVAL_BATCH   = 8
NUM_TRAIN_EPOCHS        = 10       # early stopping normally cuts this short
LEARNING_RATE           = 1e-5
WARMUP_RATIO            = 0.05
LR_SCHEDULER            = "linear"
WEIGHT_DECAY            = 0.0
MAX_GRAD_NORM           = 1.0
EARLY_STOPPING_PATIENCE = 3
LABEL_SMOOTHING         = 0.0      # >0 rewrites the loss; leave off for miscues

# --- regularisation / capacity ---
DROPOUT           = 0.0    # Whisper default; raise to 0.1 only if dev WER diverges
FREEZE_ENCODER    = False  # True = train decoder only: much cheaper, keeps the
                           # acoustic front-end intact, but adapts less to
                           # Indian-accented child speech. Try False first.
GRADIENT_CHECKPOINTING = True
USE_BF16          = True   # A100/H100/L40 -> bf16. Older cards: False + USE_FP16 True.
USE_FP16          = False
# "sdpa" uses PyTorch scaled_dot_product_attention (FlashAttention-2 kernels on
# H100). "eager" is the safe fallback; set ATTN_IMPL=eager if you hit a version
# that mis-handles Whisper under sdpa.
ATTN_IMPL         = os.environ.get("ATTN_IMPL", "sdpa")
ALLOW_TF32        = True   # ~free speedup for fp32 matmuls on Ampere and later

# --- generation during eval ---
GENERATION_MAX_LENGTH = 225
EVAL_NUM_BEAMS        = 1

# --- scoring during training (must match score_wer_by_grade.py to be comparable) ---
NORM_MODE  = "minimal"
DEVANAGARI = "keep"

# ==============================================================


# ==============================================================
# OPTIONAL CLI OVERRIDES
# The CONFIG block above remains the default and the documentation of intent —
# edit it for a one-off run, exactly like finetune_w2v2_recipe.py. These flags
# exist so the multi-stage experiment driver can run four fine-tunes in a row
# without four hand-edits of this file (and four chances to forget one).
# ==============================================================

_ap = argparse.ArgumentParser(add_help=True)
_ap.add_argument("--train_dataset")
_ap.add_argument("--dev_dataset")
_ap.add_argument("--base_model")
_ap.add_argument("--output_dir")
_ap.add_argument("--learning_rate", type=float)
_ap.add_argument("--epochs", type=int)
_ap.add_argument("--batch_size", type=int)
_ap.add_argument("--grad_accum", type=int)
_ap.add_argument("--warmup_ratio", type=float)
_ap.add_argument("--patience", type=int)
_ap.add_argument("--freeze_encoder", choices=["yes", "no"])
_ap.add_argument("--devanagari", choices=["keep", "drop", "placeholder"])
_ap.add_argument("--stage_note", default="", help="free text recorded in run_config.json")
_args = _ap.parse_args()

TRAIN_DATASET_DIR = _args.train_dataset or TRAIN_DATASET_DIR
DEV_DATASET_DIR = _args.dev_dataset or DEV_DATASET_DIR
BASE_MODEL_DIR = _args.base_model or BASE_MODEL_DIR
OUTPUT_DIR = _args.output_dir or OUTPUT_DIR
LEARNING_RATE = _args.learning_rate or LEARNING_RATE
NUM_TRAIN_EPOCHS = _args.epochs or NUM_TRAIN_EPOCHS
PER_DEVICE_TRAIN_BATCH = _args.batch_size or PER_DEVICE_TRAIN_BATCH
GRAD_ACCUM_STEPS = _args.grad_accum or GRAD_ACCUM_STEPS
WARMUP_RATIO = _args.warmup_ratio if _args.warmup_ratio is not None else WARMUP_RATIO
EARLY_STOPPING_PATIENCE = _args.patience or EARLY_STOPPING_PATIENCE
if _args.freeze_encoder:
    FREEZE_ENCODER = _args.freeze_encoder == "yes"
if _args.devanagari:
    DEVANAGARI = _args.devanagari

print(f"[CONFIG] base   : {BASE_MODEL_DIR}")
print(f"[CONFIG] train  : {TRAIN_DATASET_DIR}")
print(f"[CONFIG] dev    : {DEV_DATASET_DIR}")
print(f"[CONFIG] output : {OUTPUT_DIR}")
print(f"[CONFIG] lr={LEARNING_RATE} epochs={NUM_TRAIN_EPOCHS} "
      f"batch={PER_DEVICE_TRAIN_BATCH}x{GRAD_ACCUM_STEPS} "
      f"freeze_encoder={FREEZE_ENCODER}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ==============================================================
# STALE DISTRIBUTED ENVIRONMENT GUARD
# accelerate decides you are running DDP from a single variable:
#     accelerate/state.py:  elif int(os.environ.get("LOCAL_RANK", -1)) != -1
#                                and not cpu and torch.cuda.is_available():
#                               torch.distributed.init_process_group(...)
# So a leftover LOCAL_RANK (common inside NGC/Docker images, or left behind by
# a previous torchrun) makes it call init_process_group with the env:// method,
# which then dies on the missing WORLD_SIZE:
#     ValueError: environment variable WORLD_SIZE expected, but not set
# A real torchrun launch sets the whole set, so only clear a PARTIAL one.
# ==============================================================

_DIST_VARS = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
              "GROUP_RANK", "MASTER_ADDR", "MASTER_PORT", "TORCHELASTIC_RUN_ID")
_real_launch = "TORCHELASTIC_RUN_ID" in os.environ or all(
    v in os.environ for v in ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
)
_stale = {v: os.environ[v] for v in _DIST_VARS if v in os.environ}
if _stale and not _real_launch:
    for v in _DIST_VARS:
        os.environ.pop(v, None)
    print(f"[WARN] cleared a partial distributed environment {_stale} — it was "
          f"not a torchrun launch, and accelerate would have tried to "
          f"initialise torch.distributed and failed on the missing WORLD_SIZE. "
          f"For real multi-GPU use: NUM_GPUS=n bash 05_finetune_all_stages.sh")

# ==============================================================
# GPU GUARD
# HuggingFace Trainer moves the model and every batch to CUDA on its own — there
# is no device code to write. What it does NOT do is complain when there is no
# GPU: `Seq2SeqTrainingArguments(bf16=True)` is accepted on a CPU-only box and
# training silently proceeds on CPU, orders of magnitude slower, with no error.
# That is almost always a CPU-only torch wheel. Fail loudly instead.
# ==============================================================

if not torch.cuda.is_available():
    if os.environ.get("ALLOW_CPU") != "1":
        sys.exit(
            "[FATAL] no CUDA device visible — training would silently run on CPU.\n"
            "  nvidia-smi working?        -> driver / allocation problem\n"
            "  CPU-only torch wheel?      -> pip install torch --index-url "
            "https://download.pytorch.org/whl/cu121\n"
            "  CUDA_VISIBLE_DEVICES unset to ''?\n"
            "  Deliberate CPU run (tiny smoke test): ALLOW_CPU=1"
        )
    print("[WARN] ALLOW_CPU=1 — running on CPU. Expect this to be ~100x slower.")
    USE_BF16 = USE_FP16 = False
else:
    _n = torch.cuda.device_count()
    _p = torch.cuda.get_device_properties(0)
    print(f"[GPU] {_n} x {_p.name}  {_p.total_memory/2**30:.0f} GiB  "
          f"sm_{_p.major}{_p.minor}")
    if USE_BF16 and not torch.cuda.is_bf16_supported():
        print("[WARN] bf16 unsupported on this GPU — switching to fp16")
        USE_BF16, USE_FP16 = False, True
    if _n > 1:
        # Trainer sets train_batch_size = per_device * n_gpu when it is not
        # running under torchrun, so more visible GPUs silently multiplies the
        # effective batch and changes the recipe.
        print(
            f"[WARN] {_n} GPUs visible. Without torchrun, Trainer uses "
            f"DataParallel and the effective batch becomes "
            f"{PER_DEVICE_TRAIN_BATCH * _n * GRAD_ACCUM_STEPS}, not "
            f"{PER_DEVICE_TRAIN_BATCH * GRAD_ACCUM_STEPS}. "
            f"Pin one GPU with CUDA_VISIBLE_DEVICES=0, or launch under torchrun."
        )

set_seed(SEED)

if ALLOW_TF32:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ==============================================================
# DATA
# ==============================================================

train_dataset = load_from_disk(TRAIN_DATASET_DIR)
dev_dataset = load_from_disk(DEV_DATASET_DIR)
print(f"Train chunks: {len(train_dataset)}")
print(f"Dev chunks (full, fixed): {len(dev_dataset)}")

processor = WhisperProcessor.from_pretrained(BASE_MODEL_DIR, language=LANGUAGE, task=TASK)
feature_extractor: WhisperFeatureExtractor = processor.feature_extractor
tokenizer: WhisperTokenizer = processor.tokenizer

ON_THE_FLY = "audio" in train_dataset.column_names


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int
    on_the_fly: bool = False

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
        if self.on_the_fly:
            waves = [np.asarray(f["audio"], dtype=np.float32) / 32767.0 for f in features]
            batch = self.processor.feature_extractor(
                waves, sampling_rate=16000, return_tensors="pt"
            )
        else:
            input_features = [{"input_features": f["input_features"]} for f in features]
            batch = self.processor.feature_extractor.pad(
                input_features, return_tensors="pt"
            )

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # The tokenizer prepends <|startoftranscript|>; the model adds it again
        # when it shifts labels right to build decoder_input_ids, so strip it
        # here or every target is off by one token.
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


# ==============================================================
# MODEL
# ==============================================================

_model_kwargs = dict(
    dropout=DROPOUT,
    attention_dropout=DROPOUT,
    activation_dropout=DROPOUT,
)
try:
    model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL_DIR, attn_implementation=ATTN_IMPL, **_model_kwargs
    )
    print(f"[INFO] attention implementation: {ATTN_IMPL}")
except (ValueError, TypeError, ImportError) as _e:
    print(f"[WARN] attn_implementation={ATTN_IMPL} unavailable ({_e}); using eager")
    model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL_DIR, **_model_kwargs
    )

# Whisper decides what to decode from these. Leaving the pretrained
# forced_decoder_ids in place pins the checkpoint's own language/task prefix and
# silently overrides LANGUAGE above; clearing them and setting language/task on
# the generation config is the supported way.
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.generation_config.forced_decoder_ids = None
model.generation_config.suppress_tokens = []
model.generation_config.language = LANGUAGE
model.generation_config.task = TASK
# Chunks are scored independently, so never let generation condition on the
# previous chunk's text — that is the main source of runaway hallucination.
if hasattr(model.generation_config, "condition_on_prev_tokens"):
    model.generation_config.condition_on_prev_tokens = False

if FREEZE_ENCODER:
    model.freeze_encoder()
    print("[INFO] encoder frozen — training decoder only")

if GRADIENT_CHECKPOINTING:
    model.gradient_checkpointing_enable()
model.config.use_cache = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"[INFO] trainable params: {trainable/1e6:.1f}M / {total/1e6:.1f}M")


# ==============================================================
# METRIC
# ==============================================================

import jiwer  # noqa: E402


def compute_wer(pred):
    pred_ids = pred.predictions
    label_ids = np.where(pred.label_ids != -100, pred.label_ids, tokenizer.pad_token_id)

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    pairs = [
        (normalize(r, NORM_MODE, DEVANAGARI), normalize(h, NORM_MODE, DEVANAGARI))
        for r, h in zip(label_str, pred_str)
    ]
    pairs = [(r, h) for r, h in pairs if r]  # jiwer errors on an empty reference
    if not pairs:
        return {"wer": 1.0, "cer": 1.0}
    refs = [r for r, _ in pairs]
    hyps = [h for _, h in pairs]
    return {
        "wer": jiwer.wer(refs, hyps),
        "cer": jiwer.cer(refs, hyps),
    }


# ==============================================================
# TRAINING ARGS
#   transformers renamed `evaluation_strategy` -> `eval_strategy` in 4.46 and
#   `tokenizer=` -> `processing_class=` in the Trainer. Detect rather than pin,
#   so this script survives whichever version ends up in the `whisper` env.
# ==============================================================

_ta_params = set(inspect.signature(Seq2SeqTrainingArguments.__init__).parameters)
_eval_strategy_kw = "eval_strategy" if "eval_strategy" in _ta_params else "evaluation_strategy"

args_kwargs = dict(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    learning_rate=LEARNING_RATE,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type=LR_SCHEDULER,
    weight_decay=WEIGHT_DECAY,
    max_grad_norm=MAX_GRAD_NORM,
    label_smoothing_factor=LABEL_SMOOTHING,
    bf16=USE_BF16,
    fp16=USE_FP16,
    gradient_checkpointing=GRADIENT_CHECKPOINTING,
    save_strategy="epoch",
    save_total_limit=2,
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    predict_with_generate=True,
    generation_max_length=GENERATION_MAX_LENGTH,
    generation_num_beams=EVAL_NUM_BEAMS,
    dataloader_num_workers=4,
    remove_unused_columns=False,
    report_to="none",
    seed=SEED,
)
args_kwargs[_eval_strategy_kw] = "epoch"
training_args = Seq2SeqTrainingArguments(**args_kwargs)

# ==============================================================
# PROVENANCE
# ==============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
run_config = {
    "train_dataset": TRAIN_DATASET_DIR,
    "dev_dataset": DEV_DATASET_DIR,
    "base_model": BASE_MODEL_DIR,
    "language": LANGUAGE,
    "task": TASK,
    "seed": SEED,
    "stage_note": _args.stage_note,
    "train_size": len(train_dataset),
    "dev_size": len(dev_dataset),
    "freeze_encoder": FREEZE_ENCODER,
    "dropout": DROPOUT,
    "on_the_fly_features": ON_THE_FLY,
    "scoring": {"norm_mode": NORM_MODE, "devanagari": DEVANAGARI},
    "training_args": training_args.to_dict(),
}
with open(os.path.join(OUTPUT_DIR, "run_config.json"), "w") as f:
    json.dump(run_config, f, indent=2, default=str)

# ==============================================================
# TRAIN
# ==============================================================

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
    on_the_fly=ON_THE_FLY,
)

trainer_kwargs = dict(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    data_collator=data_collator,
    compute_metrics=compute_wer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
)
_tr_params = set(inspect.signature(Seq2SeqTrainer.__init__).parameters)
if "processing_class" in _tr_params:
    trainer_kwargs["processing_class"] = processor.feature_extractor
else:
    trainer_kwargs["tokenizer"] = processor.feature_extractor

trainer = Seq2SeqTrainer(**trainer_kwargs)

print(
    f"[INFO] device={training_args.device}  n_gpu={training_args.n_gpu}  "
    f"per-step batch={training_args.train_batch_size}  "
    f"effective batch={training_args.train_batch_size * GRAD_ACCUM_STEPS}"
)
print("[INFO] starting training ...")
trainer.train()

print("[INFO] final evaluation on the full fixed dev set:")
eval_results = trainer.evaluate()
print(eval_results)

# Generation parameters must live on generation_config, not on model.config —
# transformers 4.38 warns about the duplicates and later versions escalate it.
# Copy them across, then drop the stragglers from config before saving.
for _k in ("suppress_tokens", "begin_suppress_tokens", "forced_decoder_ids"):
    if hasattr(model.generation_config, _k) is False and _k in model.config.__dict__:
        setattr(model.generation_config, _k, model.config.__dict__[_k])
    model.config.__dict__.pop(_k, None)

trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
model.generation_config.save_pretrained(OUTPUT_DIR)

with open(os.path.join(OUTPUT_DIR, "dev_metrics.json"), "w") as f:
    json.dump(eval_results, f, indent=2, default=str)

print(f"[DONE] best model -> {OUTPUT_DIR}  dev WER = {eval_results.get('eval_wer')}")
