"""Task 13 -- LLM narration of a frozen Recommendation, and a pre-registered
evaluation of that narration's faithfulness TO the frozen artifacts.

The Task 12 verdict is frozen ground truth.
If narration fails faithfulness, that is a Task 13 finding about the narration
layer and re-litigates nothing upstream. attributions_main.csv, the
Recommendation interface, and all Task 9-12 outputs are read-only.

This module holds the REGISTERED constants and the report
plumbing shared by the three stage drivers.  Nothing here computes an
evidentiary number; the constants are pins and the helpers are formatting.

Conventions in force: ASCII-safe output, overwrite guards on
failure records, full-precision checksums, staged hard gates.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]
OUT_DIR = REPO_ROOT / "outputs" / "task13"
NARRATION_DIR = OUT_DIR / "narrations"

# Read-only frozen inputs.
SMOKE_RECOMMENDATIONS = REPO_ROOT / "outputs" / "task10" / "smoke_recommendations.csv"
SMOKE_WEIGHTS_CANDIDATES = (
    REPO_ROOT / "outputs" / "task10" / "smoke_weights.csv",
    REPO_ROOT / "outputs" / "task9" / "smoke_weights.csv",
)
ATTRIBUTIONS_MAIN = REPO_ROOT / "outputs" / "task12" / "attributions_main.csv"

# --------------------------------------------------------------------------- #
# Registered knob table.
# --------------------------------------------------------------------------- #
MODEL = "qwen3.5:9b"
ENDPOINT = "http://localhost:11434/api/generate"
VERSION_ENDPOINT = "http://localhost:11434/api/version"
SHOW_ENDPOINT = "http://localhost:11434/api/show"
TAGS_ENDPOINT = "http://localhost:11434/api/tags"

# The frozen options block.
OPTIONS: dict[str, object] = {
    "temperature": 0,
    "top_k": 1,
    "seed": 42,
    "num_predict": 512,
    "num_ctx": 8192,
    "num_gpu": 0,        # CPU forced
}

PENALTY_KNOB_NAMES = ("presence_penalty", "frequency_penalty", "repeat_penalty")
PENALTY_KNOBS: dict[str, object] = {
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "repeat_penalty": 1.0,
}

PENALTY_KNOBS_DEV_START = {
    "presence_penalty": "1.5 (model Modelfile)",
    "frequency_penalty": "UNSPECIFIED in Modelfile; server built-in default",
    "repeat_penalty": "UNSPECIFIED in Modelfile; server built-in default",
}

KEEP_ALIVE = "60m"
STREAM = False
THINK = False            # invariant I2, re-asserted per call

# Invariant I4 budget: prompt_eval_count <= num_ctx - num_predict.
I4_PROMPT_BUDGET = int(OPTIONS["num_ctx"]) - int(OPTIONS["num_predict"])

# Gate B2 static prompt-size bound (characters).
B2_PROMPT_CHAR_BOUND = 12_000

# Determinism standard: k repeat generations per cell, byte-identical response.
K_REPEATS = 3

# Registered probe prompt for the Stage A determinism measurement.  Fixed, and
# deliberately NOT the production template.
PROBE_PROMPT = "Describe in exactly three sentences what a diversified ETF portfolio is."

# Cell ordering.  Dates are READ from the frozen smoke table (never retyped);
PROFILE_ORDER: tuple[str, ...] = ("conservative", "balanced", "growth")
ANCHOR_CELL: tuple[str, str] = ("2024-12-02", "balanced")   # development cell

EFFICIENCY_TOL = 1e-12

DISPLAY_WEIGHT_DP = 1
DISPLAY_VOL_SIGFIG = 2
DISPLAY_ATTR_DP = 1


def ascii_escape(text: str) -> tuple[str, int]:
    out: list[str] = []
    n = 0
    for ch in text:
        if ord(ch) > 127:
            out.append(f"\\u{ord(ch):04x}")
            n += 1
        else:
            out.append(ch)
    return "".join(out), n


def md_table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    head = "| " + " | ".join(str(h) for h in header) + " |"
    sep = "|" + "|".join(["---"] * len(header)) + "|"
    body = ["| " + " | ".join(str(v) for v in r) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def env_header(title: str, pin: dict | None = None) -> list[str]:
    """Report header: interpreter pin, package versions, Ollama pin, options.

    Every stage report header records the python/package
    versions PLUS the Ollama server version and model digest, the options
    block, and the standard-in-force line (appended by the caller).
    """
    import numpy as np
    import pandas as pd
    import pyarrow
    import scipy

    lines = [
        f"# {title}",
        "",
        "Environment pin",
        "",
        f"- interpreter: `{sys.executable}`",
        f"- python {sys.version.split()[0]}, numpy {np.__version__}, "
        f"pandas {pd.__version__}, scipy {scipy.__version__}, "
        f"pyarrow {pyarrow.__version__}",

    ]
    if pin:
        lines += [
            f"- Ollama server version: `{pin['server_version']}`",
            f"- model tag: `{pin['model_tag']}`, digest "
            f"`{pin['model_digest']}`",
            f"- parameter size `{pin['parameter_size']}`, quantization "
            f"`{pin['quantization_level']}`, family `{pin['family']}`, "
            f"architecture `{pin['architecture']}`",
        ]
    lines += [
        "",
        "Frozen options block, sent verbatim on every call:",
        "",
        "```",
        f"model      = {MODEL!r}",
        f"endpoint   = {ENDPOINT!r}",
        f"think      = {THINK!r}      # invariant I2",
        f"stream     = {STREAM!r}",
        f"keep_alive = {KEEP_ALIVE!r}",
        f"options    = {OPTIONS!r}",
        f"penalties  = {PENALTY_KNOBS!r}"
        "```",
        "",
    ]
    return lines


def options_fingerprint() -> tuple[str, str]:
    """(canonical JSON of everything sent per call, its SHA-256).

    The registered artifact is the OPTIONS BLOCK, not just the file that holds
    it, so the hash covers the frozen options, the penalty
    knobs, and the three top-level per-call fields together.
    """
    import hashlib
    import json

    payload = {
        "model": MODEL,
        "endpoint": ENDPOINT,
        "think": THINK,
        "stream": STREAM,
        "keep_alive": KEEP_ALIVE,
        "options": {**OPTIONS, **PENALTY_KNOBS},
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return blob, hashlib.sha256(blob.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_report(name: str, lines: Sequence[str]) -> str:
    """Write the report.

    The file lands first so a violation never costs the evidence; the check
    then halts.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target = OUT_DIR / name
    if target.exists():
        old = target.read_text(encoding="utf-8", errors="replace")
        if "STAGE A HALTED" in old or "HALT." in old:
            raise RuntimeError(
                f"overwrite guard: {name} is a failure "
                "record and must not be regenerated in place "
                "-- STOP"
            )
    report = "\n".join(lines)
    target.write_text(report, encoding="utf-8")
    bad = sorted({c for c in report if ord(c) > 127})

    return report


def guarded_write(path: Path, text: str) -> None:
    """
    Any file whose existing content contains a FAIL/HALT marker is preserved;
    the new content is written beside it with a numbered suffix instead.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = path.read_text(encoding="utf-8", errors="replace")
        if "FAIL" in old or "HALT" in old:
            n = 1
            while (alt := path.with_suffix(f".{n}{path.suffix}")).exists():
                n += 1
            alt.write_text(text, encoding="utf-8")
            raise RuntimeError(
                f"overwrite guard: {path.name} holds a failure record and was "
                f"not replaced; new content written to {alt.name} -- STOP"
            )
    path.write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Static import audit -- evidence for the "no shap / no torch / no requests"
# --------------------------------------------------------------------------- #
FORBIDDEN_IMPORTS = ("shap", "torch", "requests", "ollama")


def package_imports() -> list[tuple[str, str, str]]:
    """(file, statement, top-level target) for every import in ``src/task13``."""
    import ast

    rows: list[tuple[str, str, str]] = []
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    rows.append((path.name, f"import {alias.name}",
                                 alias.name.split(".")[0]))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = ", ".join(a.name for a in node.names)
                dots = "." * node.level
                target = ("src." + module) if node.level >= 2 else (
                    f"src.task13.{module}" if node.level == 1 else module.split(".")[0]
                )
                rows.append((path.name, f"from {dots}{module} import {names}", target))
    return rows
