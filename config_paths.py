#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single source of truth for every path outside this folder.

Why this exists
---------------
Copy `whisper_finetune/` to another machine and nothing should need editing
except, at most, four environment variables. Two rules make that work:

1. **This folder locates itself.** `WHISPER_HOME` is derived from
   `__file__`, never hardcoded, so scripts, data, models and results move
   together as one unit.
2. **Audio lives outside, so it is referenced by named root, not by absolute
   path.** Manifests store `${WPP_ROOT}/WPP_VV_Mumbai_2020/audios/x.wav`
   rather than `/path/to/workspace/.../x.wav`. Readers call `resolve()`.

To move to a new server: copy the folder, put the corpora somewhere, and either
export the variables or edit the defaults below. `python check_paths_resolve.py` tells
you exactly which ones are wrong.

    export WPP_ROOT=/data/corpora/wpp-2020
    export IITM_ROOT=/data/corpora/IITM_English_all_new
    export MPS_ROOT=/data/corpora/mps_dataset
    export KV_ROOT=/data/corpora/KVS_6Grades_EN_MT
"""

import os
import re

WHISPER_HOME = os.path.dirname(os.path.abspath(__file__))

DATA = os.path.join(WHISPER_HOME, "data")
# Audio brought inside the folder by copy_audio_to_local_disk.py. Once that has run, the
# manifests reference ${AUDIO_ROOT} and the project needs NO environment
# variables at all — it is fully self-contained.
AUDIO_ROOT = os.path.join(DATA, "audio")
MANIFESTS = os.path.join(WHISPER_HOME, "manifests_kaldi")
PRETRAINED = os.path.join(WHISPER_HOME, "pretrained")
HF_DATASETS = os.path.join(WHISPER_HOME, "hf_whisper")
MODELS = os.path.join(WHISPER_HOME, "models")
DECODE = os.path.join(WHISPER_HOME, "decode")
RESULTS = os.path.join(WHISPER_HOME, "results")

# Corpus roots. Environment variable wins; otherwise the value this workspace
# was built on. These are the ONLY machine-specific strings in the project.
_DEFAULTS = {
    "WPP_ROOT": (
        "/path/to/corpora/wpp-2020"
    ),
    "IITM_ROOT": "/path/to/corpora/IITM_English_all_new",
    "MPS_ROOT": "/path/to/corpora/mps_dataset",
    "KV_ROOT": "/path/to/corpora/KVS_6Grades_EN_MT",
}

ROOTS = {k: os.environ.get(k, v) for k, v in _DEFAULTS.items()}
# Always resolvable, never machine-specific: it lives inside this folder.
ROOTS["AUDIO_ROOT"] = AUDIO_ROOT

_VAR = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def resolve(path: str) -> str:
    """`${WPP_ROOT}/a/b.wav` -> `/data/corpora/wpp-2020/a/b.wav`.

    Absolute paths with no placeholder pass through unchanged, so manifests
    written before this indirection existed still load.
    """
    if not path:
        return path

    def sub(m):
        name = m.group(1)
        if name in ROOTS:
            return ROOTS[name]
        if name in os.environ:
            return os.environ[name]
        raise KeyError(
            f"unknown path root ${{{name}}} in manifest. Known roots: "
            f"{sorted(ROOTS)}. Set it as an environment variable or add it to "
            f"config_paths.py."
        )

    return _VAR.sub(sub, path)


def templatize(path: str) -> str:
    """Inverse of `resolve`: rewrite an absolute path under a known root back to
    `${ROOT}/...`. Longest root first, so nested roots can't mis-match."""
    for name, root in sorted(ROOTS.items(), key=lambda kv: -len(kv[1])):
        root = root.rstrip("/")
        if root and (path == root or path.startswith(root + "/")):
            return "${" + name + "}" + path[len(root):]
    return path


def missing_roots():
    """-> [(name, path)] for roots that do not exist on this machine."""
    return [(k, v) for k, v in sorted(ROOTS.items()) if not os.path.isdir(v)]
