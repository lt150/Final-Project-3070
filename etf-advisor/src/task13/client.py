"""Strictly sequential Ollama client for Task 13.

Transport is stdlib ``urllib.request`` only: no ``ollama``
package, no ``requests``.  Every generation goes through :func:`generate`,
which sends the frozen options block verbatim and asserts / logs the four
registered per-call invariants:

- **I1** response ``model`` field equals the pinned model name.
- **I2** no ``thinking`` field and no thinking content in the payload.
- **I3** ``done`` is true; ``done_reason`` is recorded verbatim.  A
  ``done_reason != "stop"`` is a *recorded observation*, not a failure --
  evaluation proceeds on the emitted text.
- **I4** ``prompt_eval_count <= num_ctx - num_predict`` (= 7680), the
  no-silent-truncation check, verified from the real response and logged.

I1, I2 and the ``done`` half of I3 invalidate the evidence if they break, so
they raise.  The ``done_reason`` value and the I4 comparison are recorded as
data on the result object and surface in the per-call log.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from . import (
    ENDPOINT,
    KEEP_ALIVE,
    I4_PROMPT_BUDGET,
    MODEL,
    OPTIONS,
    PENALTY_KNOBS,
    PENALTY_KNOB_NAMES,
    SHOW_ENDPOINT,
    STREAM,
    TAGS_ENDPOINT,
    THINK,
    VERSION_ENDPOINT,
)

# Sequentiality is a registered property of the client, so it is enforced
# structurally rather than merely intended.
_IN_FLIGHT = False
_CALL_SEQ = 0


class SequentialityError(RuntimeError):
    pass


class InvariantError(RuntimeError):
    pass


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_json(url: str, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# --------------------------------------------------------------------------- #
# Pin capture (Stage A step 1) -- read from the live system, never retyped
# --------------------------------------------------------------------------- #
def server_version(timeout: float = 15.0) -> str:
    return str(_get_json(VERSION_ENDPOINT, timeout)["version"])


def capture_pin(timeout: float = 60.0) -> dict[str, Any]:
    """Server version + model tag/digest/details, imported from the live host.

    Every field lands in the report verbatim.  A missing field is left as the
    literal string ``"ABSENT"`` so Gate A1 can see the hole rather than have it
    papered over by a default.
    """
    version = server_version(timeout)

    tags = _get_json(TAGS_ENDPOINT, timeout)
    matches = [m for m in tags.get("models", []) if m.get("name") == MODEL]
    if len(matches) != 1:
        raise InvariantError(
            f"pin capture: /api/tags lists {len(matches)} entries named {MODEL!r} "
            f"(expected exactly 1); available = "
            f"{[m.get('name') for m in tags.get('models', [])]} -- STOP"
        )
    tag_entry = matches[0]

    show = _post_json(SHOW_ENDPOINT, {"model": MODEL}, timeout)
    details = show.get("details", {}) or {}
    info = show.get("model_info", {}) or {}

    def g(d: dict, k: str) -> Any:
        v = d.get(k)
        return "ABSENT" if v in (None, "", [], {}) else v

    return {
        "server_version": version,
        "model_tag": MODEL,
        "model_digest": g(tag_entry, "digest"),
        "model_size_bytes": g(tag_entry, "size"),
        "parameter_size": g(details, "parameter_size"),
        "quantization_level": g(details, "quantization_level"),
        "format": g(details, "format"),
        "family": g(details, "family"),
        "families": g(details, "families"),
        "parent_model": details.get("parent_model", "") or "(none)",
        "architecture": g(info, "general.architecture"),
        "parameter_count": g(info, "general.parameter_count"),
        "file_type": g(info, "general.file_type"),
        "native_context_length": g(info, f"{info.get('general.architecture', 'x')}.context_length"),
        "template": g(show, "template"),
        "model_default_parameters": g(show, "parameters"),
        "capabilities": g(show, "capabilities"),
        "requires": g(show, "requires"),
        "renderer_parser": _renderer_parser(show.get("modelfile", "")),
    }


def _renderer_parser(modelfile: str) -> str:
    """The RENDERER / PARSER directives, as reported by ``/api/show``.

    They matter for reproducibility: they say whether the server applies a
    native chat renderer to ``/api/generate`` prompts and a native parser that
    splits thinking from the response, rather than a Go template.
    """
    picked = [
        line.strip()
        for line in modelfile.splitlines()
        if line.strip().startswith(("RENDERER", "PARSER", "TEMPLATE"))
    ]
    return "; ".join(picked) if picked else "ABSENT"


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
@dataclass
class GenerationResult:
    seq: int
    purpose: str
    prompt_chars: int
    response: str
    raw: dict = field(repr=False)
    # invariant outcomes
    i1_model_ok: bool = False
    i1_model_reported: str = ""
    i2_no_thinking_ok: bool = False
    i2_thinking_evidence: str = ""
    i3_done: bool = False
    i3_done_reason: str = ""
    i4_prompt_eval_count: int = -1
    i4_ok: bool = False
    # timing / counters
    wall_seconds: float = 0.0
    total_duration_ns: int = -1
    load_duration_ns: int = -1
    prompt_eval_duration_ns: int = -1
    eval_count: int = -1
    eval_duration_ns: int = -1

    @property
    def eval_tokens_per_second(self) -> float:
        if self.eval_count > 0 and self.eval_duration_ns > 0:
            return self.eval_count / (self.eval_duration_ns / 1e9)
        return float("nan")

    def log_row(self) -> dict[str, object]:
        return {
            "seq": self.seq,
            "purpose": self.purpose,
            "prompt_chars": self.prompt_chars,
            "response_chars": len(self.response),
            "i1_model_ok": self.i1_model_ok,
            "i1_model_reported": self.i1_model_reported,
            "i2_no_thinking_ok": self.i2_no_thinking_ok,
            "i2_thinking_evidence": self.i2_thinking_evidence,
            "i3_done": self.i3_done,
            "i3_done_reason": self.i3_done_reason,
            "i4_prompt_eval_count": self.i4_prompt_eval_count,
            "i4_budget": I4_PROMPT_BUDGET,
            "i4_ok": self.i4_ok,
            "wall_seconds": f"{self.wall_seconds:.3f}",
            "total_duration_ns": self.total_duration_ns,
            "load_duration_ns": self.load_duration_ns,
            "prompt_eval_duration_ns": self.prompt_eval_duration_ns,
            "eval_count": self.eval_count,
            "eval_duration_ns": self.eval_duration_ns,
            "eval_tokens_per_second": f"{self.eval_tokens_per_second:.4f}",
        }


LOG_COLUMNS = [
    "seq", "purpose", "prompt_chars", "response_chars",
    "i1_model_ok", "i1_model_reported", "i2_no_thinking_ok",
    "i2_thinking_evidence", "i3_done", "i3_done_reason",
    "i4_prompt_eval_count", "i4_budget", "i4_ok",
    "wall_seconds", "total_duration_ns", "load_duration_ns",
    "prompt_eval_duration_ns", "eval_count", "eval_duration_ns",
    "eval_tokens_per_second",
]


def _check_i2(payload: dict) -> tuple[bool, str]:
    """I2: no ``thinking`` field, no thinking content anywhere in the payload.

    Ollama surfaces reasoning traces on a top-level ``thinking`` key when a
    model is asked to think.  With ``think: false`` the key should be absent or
    empty.  The check also scans for the raw ``<think>`` marker in the response
    in case a build ever leaks the trace into the text.
    """
    evidence: list[str] = []
    if "thinking" in payload:
        value = payload.get("thinking")
        if value not in (None, ""):
            evidence.append(f"thinking field present, {len(str(value))} chars")
        else:
            evidence.append("thinking key present but empty")
    text = payload.get("response", "") or ""
    for marker in ("<think>", "</think>", "<thinking>"):
        if marker in text:
            evidence.append(f"marker {marker!r} in response")
    hard = [e for e in evidence if "empty" not in e]
    return (not hard), ("; ".join(evidence) if evidence else "none")


def effective_options(dev_overrides: dict | None = None) -> dict:
    """The options actually sent: frozen block + frozen penalty knobs + dev.

    ``dev_overrides`` is permitted ONLY during anchor-cell template development 
    and may touch only the three penalty knobs; anything
    else raises, so the block cannot be edited by the back door.
    """
    opts = dict(OPTIONS)
    opts.update(PENALTY_KNOBS)
    if dev_overrides:
        illegal = set(dev_overrides) - set(PENALTY_KNOB_NAMES)
        if illegal:
            raise InvariantError(
                f"dev overrides may touch only {PENALTY_KNOB_NAMES};  "
                f"got {sorted(illegal)} -- STOP"
            )
        opts.update(dev_overrides)
    return opts


def generate(prompt: str, *, purpose: str, timeout: float = 1800.0,
             dev_overrides: dict | None = None) -> GenerationResult:
    """One generation under the frozen options block.  Strictly sequential."""
    global _IN_FLIGHT, _CALL_SEQ
    if _IN_FLIGHT:
        raise SequentialityError(
            "client.generate re-entered while a request was in flight; the "
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
            "options": effective_options(dev_overrides),
        }
        t0 = time.perf_counter()
        try:
            raw = _post_json(ENDPOINT, payload, timeout)
        except urllib.error.HTTPError as exc:  # surface the server's own message
            detail = exc.read().decode("utf-8", errors="replace")
            raise InvariantError(
                f"generation seq={seq} purpose={purpose!r}: HTTP {exc.code} from "
                f"{ENDPOINT}: {detail} -- STOP"
            ) from exc
        wall = time.perf_counter() - t0
    finally:
        _IN_FLIGHT = False

    response = raw.get("response", "")
    res = GenerationResult(
        seq=seq,
        purpose=purpose,
        prompt_chars=len(prompt),
        response=response,
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
        raise InvariantError(
            f"I1 violated on seq={seq}: response model {res.i1_model_reported!r} "
            f"!= pinned {MODEL!r} -- STOP"
        )

    # I2 -- non-thinking
    res.i2_no_thinking_ok, res.i2_thinking_evidence = _check_i2(raw)
    if not res.i2_no_thinking_ok:
        raise InvariantError(
            f"I2 violated on seq={seq}: thinking content present "
            f"({res.i2_thinking_evidence}) despite think={THINK!r} -- STOP"
        )

    # I3 -- completion; done_reason is data, not a criterion
    res.i3_done = bool(raw.get("done", False))
    res.i3_done_reason = str(raw.get("done_reason", "ABSENT"))
    if not res.i3_done:
        raise InvariantError(
            f"I3 violated on seq={seq}: done={raw.get('done')!r} on a "
            f"stream=False request -- STOP"
        )

    # I4 -- no silent truncation.  Recorded, not raised: a breach is a logged
    # deviation the stage report must carry.
    res.i4_prompt_eval_count = int(raw.get("prompt_eval_count", -1))
    res.i4_ok = 0 <= res.i4_prompt_eval_count <= I4_PROMPT_BUDGET

    return res


def call_count() -> int:
    return _CALL_SEQ
