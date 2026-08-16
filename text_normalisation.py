#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared text normalisation for Whisper training/scoring on the KV oral-reading data.

Why a custom normaliser instead of Whisper's own `EnglishTextNormalizer`:
the references here are *verbatim miscue transcripts* — they record what the
child actually said, including misreadings, substitutions and code-switched
Devanagari. `EnglishTextNormalizer` "repairs" text (expands contractions,
maps "gonna"->"going to", strips filler words, rewrites number words) which is
exactly the kind of edit that destroys a miscue. Use `minimal` for anything
you intend to draw reading-accuracy conclusions from; `whisper_en` is provided
only so you can report the "standard ASR benchmark" number alongside it.

Tags used by the KV label tracks (removed from every reference):
    ON  ambient/other noise      BR  breath          SIL silence
    MB  mumble/background        FP  filled pause    IR  irrelevant
    HS  hesitation               WH  whisper
"""

import re
import unicodedata

LABEL_TAGS = {"ON", "BR", "SIL", "MB", "FP", "IR", "HS", "WH"}

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
# keep letters (any script), digits, apostrophes; everything else -> space
_KEEP = re.compile(r"[^\w'ऀ-ॿ]+", flags=re.UNICODE)
_MULTISPACE = re.compile(r"\s+")


def strip_tags(text: str) -> str:
    """Drop the uppercase annotation tags from a raw label-track line."""
    return " ".join(t for t in text.split() if t not in LABEL_TAGS)


def is_tag_only(text: str) -> bool:
    toks = text.split()
    return len(toks) == 0 or all(t in LABEL_TAGS for t in toks)


def has_devanagari(text: str) -> bool:
    return bool(_DEVANAGARI.search(text))


def normalize(text: str, mode: str = "minimal", devanagari: str = "keep") -> str:
    """
    mode:
        minimal    lowercase, NFKC, punctuation stripped, whitespace collapsed.
                   No lexical rewriting at all -> safe for miscue analysis.
        whisper_en Whisper's EnglishTextNormalizer (needs transformers).
                   Comparable to published Whisper WERs, but rewrites words.
    devanagari:
        keep        score code-switched tokens as-is (they are real miscues).
        drop        delete Devanagari tokens from BOTH ref and hyp, so you get
                    an English-only WER comparable to an English-only AM.
                    Note this makes those positions FREE: a model that emits
                    nothing there scores identically to one that transcribed
                    the non-word correctly. It does still penalise a model that
                    "repairs" the non-word into the English word the child was
                    supposed to read — that becomes an insertion.
        placeholder replace every Devanagari token with <L1> on BOTH sides.
                    The model must emit *something* non-English at that
                    position, but is not required to spell the non-word
                    exactly. Usually the right choice for reading assessment:
                    it measures detection of the L1 substitution without
                    scoring the transcriber's ad-hoc spelling of it.

    NOTE ON TRAINING: whatever you choose here for scoring, keep the Devanagari
    tokens in the *training* targets. Stripping them leaves audio with no
    corresponding text, which trains the decoder to skip real speech — the same
    failure mode as feeding it a truncated 30 s window (see README §3).
    """
    if mode == "whisper_en":
        from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

        global _EN_NORM
        try:
            _EN_NORM
        except NameError:
            _EN_NORM = EnglishTextNormalizer({})
        text = _EN_NORM(text)
    else:
        text = unicodedata.normalize("NFKC", text)
        text = text.lower()
        text = _KEEP.sub(" ", text)

    if devanagari == "drop":
        text = " ".join(t for t in text.split() if not _DEVANAGARI.search(t))
    elif devanagari == "placeholder":
        text = " ".join(
            "<L1>" if _DEVANAGARI.search(t) else t for t in text.split()
        )

    return _MULTISPACE.sub(" ", text).strip()
