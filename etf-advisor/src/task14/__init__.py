"""Task 14 -- risk-preference elicitor, two-arm selection.

Boundary: *The Task 12 and Task 13 verdicts are frozen
ground truth, and all Task 9-13 outputs are read-only. The elicitor creates no
new profile, ladder, or budget freedom: its output is exactly one registered
profile string from ``PROFILES`` (src.task9.allocators) or a clarify signal.
``recommend()`` validation is the backstop, never the primary check. A weak
elicitor is a Task 14 finding about the elicitation layer and re-litigates
nothing upstream.*

This module holds the REGISTERED constants and the report
plumbing shared by the three stage drivers.  Nothing here computes an
evidentiary number; the constants are pins and the helpers are formatting.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

# --------------------------------------------------------------------------- #
# Frozen inheritances -- IMPORTED, never retyped 
# --------------------------------------------------------------------------- #
from ..task13 import (
    ENDPOINT,
    KEEP_ALIVE,
    MODEL,
    OPTIONS as T13_OPTIONS,
    PENALTY_KNOBS,
    PROBE_PROMPT,
    K_REPEATS,
    B2_PROMPT_CHAR_BOUND as PROMPT_CHAR_BOUND,
    STREAM,
    THINK,
    VERSION_ENDPOINT,
)
from ..task13 import client as _client
from ..task9.allocators import PROFILES

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]
OUT_DIR = REPO_ROOT / "outputs" / "task14"
TRANSCRIPT_DIR = OUT_DIR / "transcripts"

# The registered persona fixture, hashed at Stage A.
PERSONA_PATH = PACKAGE_DIR / "personas.json"


# The Task 13 record the model pin is carried from (read-only).
T13_STAGE_A_REPORT = REPO_ROOT / "outputs" / "task13" / "stage_a_report.md"
T13_STAGE_C_REPORT = REPO_ROOT / "outputs" / "task13" / "stage_c_report.md"
# Task 13's own repeat-1 transcripts of the SAME registered probe prompt.
# Read-only; used for the recorded cross-task observation, which feeds no gate.
#   probe_rep1    -- Task 13 STAGE A, under the SHIPPED penalties
#                    (presence_penalty 1.5).
#   b3_probe_rep1 -- Task 13 GATE B3 re-probe, under the frozen
#                    penalties {0.0, 0.0, 1.0}; this is the penalties-constant
#                    comparator for Task 14, which carries those same values.
T13_PROBE_REP1 = REPO_ROOT / "outputs" / "task13" / "narrations" / "probe_rep1.txt"
T13_B3_PROBE_REP1 = (REPO_ROOT / "outputs" / "task13" / "narrations"
                     / "b3_probe_rep1.txt")

# --------------------------------------------------------------------------- #
# Registered knob table
# --------------------------------------------------------------------------- #
# Pin, knob table.  These two literals are
# the ONLY retyped pin values in the package, and Stage A ties both of them to
# the Task 13 record on disk AND to the live host before using them.
PIN_SERVER_VERSION = "0.32.5"
PIN_MODEL_DIGEST = (
    "6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7"
)

# Options variant: the Task 13 frozen block with exactly one registered delta.
NUM_PREDICT = 128
OPTIONS: dict[str, object] = {**T13_OPTIONS, "num_predict": NUM_PREDICT}

# Invariant I4 budget for this task: num_ctx - num_predict = 8192 - 128 = 8064.
I4_PROMPT_BUDGET = int(OPTIONS["num_ctx"]) - int(OPTIONS["num_predict"])

# --------------------------------------------------------------------------- #
# Registered question set and extraction domains
# --------------------------------------------------------------------------- #
Q_KEYS: tuple[str, ...] = ("Q1", "Q2", "Q3", "Q4", "Q5")

QUESTIONS: dict[str, str] = {
    "Q1": "When do you expect to need most of this money?",
    "Q2": "Suppose your portfolio lost 20% of its value within a year. "
          "What would you most likely do?",
    "Q3": "What matters more to you: protecting what you have, or growing it "
          "as much as possible over time?",
    "Q4": "How stable is your income, and do you have an emergency fund "
          "covering several months of expenses?",
    "Q5": "Have you invested in markets before, and how comfortable are you "
          "with market fluctuations?",
}

# SCORED domain values, in the order the register lists them.  The list ORDER is
# load-bearing: the rubric's points are the index.
DOMAINS: dict[str, tuple[str, ...]] = {
    "Q1": ("short", "medium", "long"),
    "Q2": ("sell", "hold", "buy_more"),
    "Q3": ("preserve", "balanced", "grow"),
    "Q4": ("low", "medium", "high"),
    "Q5": ("none", "some", "experienced"),
}

UNCLEAR = "unclear"

# What Arm A's extraction parser will accept per question: the scored domain
# plus the first-class ``unclear``.
EXTRACTION_DOMAINS: dict[str, tuple[str, ...]] = {
    q: DOMAINS[q] + (UNCLEAR,) for q in Q_KEYS
}

# The output.  PROFILES is imported from src.task9.allocators; 
# the ordering conservative < balanced < growth used
# by the rubric's ``min`` is the PROFILES tuple order itself.
CLARIFY = "clarify"
PROFILE_ORDER: tuple[str, ...] = tuple(PROFILES)
ARM_B_VOCAB: tuple[str, ...] = PROFILE_ORDER + (CLARIFY,)

# --------------------------------------------------------------------------- #
# Import audit surface
# --------------------------------------------------------------------------- #
FORBIDDEN_IMPORTS = ("shap", "torch", "requests", "ollama")
# Non-stdlib targets this package is authorized to import, and nothing else.
ALLOWED_NON_STDLIB = (
    "src.task14",            # intra-package
    "src.task13",            # frozen constants, carried never retyped
    "src.task13.client",     # frozen transport
    "src.task9.allocators",  # PROFILES only
)


# --------------------------------------------------------------------------- #
# Hashing / ASCII-safety / formatting helpers
# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ascii_escape(text: str) -> tuple[str, int]:
    """(ascii-safe rendering, count of non-ASCII characters replaced).

    Registered fixtures and model output both may contain non-ASCII characters
    (the register itself uses U+2014 inside several persona answers).  Files on
    disk always keep the raw text; this helper is used only where such text is
    quoted INTO a report, so the ASCII-safe rule is satisfied
    without discarding or silently normalising evidence.
    """
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


def normalize_ws(text: str) -> str:
    """Collapse every run of whitespace to a single space.

    The register is hard-wrapped at ~72 columns, so a registered fixture string
    can straddle a line break.  Every cross-check in this package compares
    whitespace-normalised needle against whitespace-normalised register text;
    the normalisation is symmetric and touches whitespace only.
    """
    return " ".join(text.split())


def options_fingerprint() -> tuple[str, str]:
    """(canonical JSON of everything sent per call, its SHA-256).

    Structurally identical to ``src.task13.options_fingerprint`` -- same keys,
    same canonical-JSON settings -- so the two hashes are directly comparable
    and the ONLY thing that can move the digest is the registered
    ``num_predict`` delta.  Stage A prints both and the Task 13 recorded value.
    """
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


def env_header(title: str, pin: dict | None = None) -> list[str]:
    """Report header: interpreter pin, package versions, Ollama pin, options 
    and options-variant hash.
    """
    import importlib.metadata as im

    blob, opt_sha = options_fingerprint()
    lines = [
        f"# {title}",
        "",
        "Environment pin (registered convention): everything evidentiary runs "
        "under the repo venv; the system interpreter is never used.",
        "",
        f"- interpreter: `{sys.executable}`",
        f"- python {sys.version.split()[0]}, numpy {im.version('numpy')}, "
        f"pandas {im.version('pandas')}, scipy {im.version('scipy')}, "
        f"pyarrow {im.version('pyarrow')}",
    ]
    if pin:
        lines += [
            f"- Ollama server version: `{pin['server_version']}`",
            f"- model tag: `{pin['model_tag']}`, digest `{pin['model_digest']}`",
            f"- parameter size `{pin['parameter_size']}`, quantization "
            f"`{pin['quantization_level']}`, family `{pin['family']}`, "
            f"architecture `{pin['architecture']}`",
        ]
    lines += [
        f"- options-variant hash: `{opt_sha}`",
        "",
        "Registered options variant -- the Task 13 frozen "
        "block with exactly one delta, `num_predict` 512 -> 128; sent verbatim "
        "on every call:",
        "",
        "```",
        f"model      = {MODEL!r}",
        f"endpoint   = {ENDPOINT!r}",
        f"think      = {THINK!r}      # invariant I2",
        f"stream     = {STREAM!r}",
        f"keep_alive = {KEEP_ALIVE!r}",
        f"options    = {OPTIONS!r}",
        f"penalties  = {PENALTY_KNOBS!r}",
        f"I4 budget  = {I4_PROMPT_BUDGET}   # num_ctx - num_predict",
        "```",
        "",
    ]
    return lines


def write_report(name: str, lines: Sequence[str]) -> str:
    """Write the report, then enforce the ASCII-safe rule.

    The file lands first so a violation never costs the evidence; the check
    then halts loudly.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target = OUT_DIR / name
    if target.exists():
        old = target.read_text(encoding="utf-8", errors="replace")
        if "HAND-AUGMENTED" in old or "HALTED" in old or "HALT." in old:
            raise RuntimeError(
                f"overwrite guard: {name} is a hand-augmented report or a "
                "failure record and must not be regenerated in place "
                " -- STOP"
            )
    report = "\n".join(lines)
    target.write_text(report, encoding="utf-8")
    bad = sorted({c for c in report if ord(c) > 127})
    if bad:
        raise RuntimeError(
            f"{name}: non-ASCII characters in report output "
            f"{[(c, hex(ord(c))) for c in bad]} (requires "
            "ASCII-safe output) -- STOP"
        )
    return report


def guarded_write(path: Path, text: str) -> None:
    """Overwrite guard: never silently replace a failure record."""
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
# Static import audit
# --------------------------------------------------------------------------- #
def package_imports() -> list[tuple[str, str, str]]:
    """(file, statement, top-level target) for every import in ``src/task14``."""
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
                    f"src.task14.{module}" if node.level == 1
                    else module.split(".")[0]
                )
                rows.append((path.name, f"from {dots}{module} import {names}",
                             target))
    return rows


def classify_import(target: str) -> str:
    """``stdlib`` / ``frozen-allowed`` / ``FORBIDDEN`` / ``UNAUTHORIZED``."""
    if target in FORBIDDEN_IMPORTS:
        return "FORBIDDEN"
    if target in sys.stdlib_module_names:
        return "stdlib"
    if target in ALLOWED_NON_STDLIB or target.startswith("src.task14"):
        return "frozen-allowed"
    return "UNAUTHORIZED"


# --------------------------------------------------------------------------- #
# Transport -- frozen client.py primitives + the single registered delta
# --------------------------------------------------------------------------- #
_IN_FLIGHT = False
_CALL_SEQ = 0


def effective_options() -> dict:
    """Exactly what goes on the wire: the frozen Task 13 effective options 
    with the one registered Task 14 delta applied on top.
    """
    return {**_client.effective_options(), "num_predict": NUM_PREDICT}


def generate(prompt: str, *, purpose: str,
             timeout: float = 1800.0) -> _client.GenerationResult:
    """One generation under the registered options variant.  Strictly sequential.

    I1/I2/I3 raise (they invalidate the evidence); I4 is recorded as data
    against the task-14 budget, matching the frozen Task 13 posture.
    """
    global _IN_FLIGHT, _CALL_SEQ
    if _IN_FLIGHT:
        raise _client.SequentialityError(
            "task14.generate re-entered while a request was in flight; the "
            "registered client is strictly sequential -- STOP"
        )
    _IN_FLIGHT = True
    _CALL_SEQ += 1
    seq = _CALL_SEQ
    try:
        payload = {
            "model": MODEL,
            "prompt": prompt,
            "stream": STREAM,
            "think": THINK,
            "keep_alive": KEEP_ALIVE,
            "options": effective_options(),
        }
        t0 = time.perf_counter()
        try:
            raw = _client._post_json(ENDPOINT, payload, timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise _client.InvariantError(
                f"generation seq={seq} purpose={purpose!r}: HTTP {exc.code} "
                f"from {ENDPOINT}: {detail} -- STOP"
            ) from exc
        wall = time.perf_counter() - t0
    finally:
        _IN_FLIGHT = False

    res = _client.GenerationResult(
        seq=seq,
        purpose=purpose,
        prompt_chars=len(prompt),
        response=raw.get("response", ""),
        raw=raw,
        wall_seconds=wall,
        total_duration_ns=int(raw.get("total_duration", -1)),
        load_duration_ns=int(raw.get("load_duration", -1)),
        prompt_eval_duration_ns=int(raw.get("prompt_eval_duration", -1)),
        eval_count=int(raw.get("eval_count", -1)),
        eval_duration_ns=int(raw.get("eval_duration", -1)),
    )

    # I1 -- pinned model
    res.i1_model_reported = str(raw.get("model", "ABSENT"))
    res.i1_model_ok = res.i1_model_reported == MODEL
    if not res.i1_model_ok:
        raise _client.InvariantError(
            f"I1 violated on seq={seq}: response model "
            f"{res.i1_model_reported!r} != pinned {MODEL!r} -- STOP"
        )

    # I2 -- non-thinking, checked by the FROZEN Task 13 implementation
    res.i2_no_thinking_ok, res.i2_thinking_evidence = _client._check_i2(raw)
    if not res.i2_no_thinking_ok:
        raise _client.InvariantError(
            f"I2 violated on seq={seq}: thinking content present "
            f"({res.i2_thinking_evidence}) despite think={THINK!r} -- STOP"
        )

    # I3 -- completion; done_reason is data, not a criterion
    res.i3_done = bool(raw.get("done", False))
    res.i3_done_reason = str(raw.get("done_reason", "ABSENT"))
    if not res.i3_done:
        raise _client.InvariantError(
            f"I3 violated on seq={seq}: done={raw.get('done')!r} on a "
            f"stream=False request -- STOP"
        )

    # I4 -- no silent truncation, against the TASK 14 budget.  Recorded.
    res.i4_prompt_eval_count = int(raw.get("prompt_eval_count", -1))
    res.i4_ok = 0 <= res.i4_prompt_eval_count <= I4_PROMPT_BUDGET

    return res


def call_count() -> int:
    return _CALL_SEQ


LOG_COLUMNS = [
    "seq", "purpose", "prompt_chars", "response_chars",
    "i1_model_ok", "i1_model_reported", "i2_no_thinking_ok",
    "i2_thinking_evidence", "i3_done", "i3_done_reason",
    "i4_prompt_eval_count", "i4_budget", "i4_ok",
    "wall_seconds", "total_duration_ns", "load_duration_ns",
    "prompt_eval_duration_ns", "eval_count", "eval_duration_ns",
    "eval_tokens_per_second",
]


def log_row(res: _client.GenerationResult) -> dict[str, object]:
    """Task 13's log row with the task-14 I4 budget in the ``i4_budget`` column."""
    row = dict(res.log_row())
    row["i4_budget"] = I4_PROMPT_BUDGET
    return row


def read_live_pin(timeout: float = 60.0) -> dict[str, Any]:
    """Pin captured from the live host by the frozen Task 13 code."""
    return _client.capture_pin(timeout)


def server_version(timeout: float = 15.0) -> str:
    return _client.server_version(timeout)
