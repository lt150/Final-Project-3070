"""Deterministic, rule-based claim extraction from narration text.

No model is involved here and no numeric is computed here: this module turns a
narration string into (i) a sentence classification table and (ii) a list of
canonical claim tuples, and it reports every numeric token it could NOT consume.
Adjudication against the frozen artifacts happens in :mod:`adjudicate`.

Design commitment: **under-extraction is punished, not
rewarded.** Every numeric token in the evaluated narration must be consumed by
exactly one extracted claim or be a cell-date token; anything else is an
UNACCOUNTED NUMERIC and takes the T7 hard-fail path. An extractor that silently
declines to parse an awkward sentence therefore fails the run rather than
flattering it.

Claim tuple:
``(type, subject, basis/object, value, stated_precision, char_span)``.

Precision notation: ``pct.N`` percent with N decimals, ``pp.N`` percentage
points with N decimals, ``raw.N`` stored units with N decimals, ``-`` for
non-numeric claims.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# Registered lexicons.  Fixed before any narration exists; every entry is a
# knob the Gate B1 fixtures pin.
# --------------------------------------------------------------------------- #
RANKING_MAX = ("largest", "biggest", "greatest", "highest", "top", "most")
RANKING_MIN = ("smallest", "lowest", "least")


DIRECTION_POSITIVE = (
    "overweights", "overweight", "overweighted",
    "increases", "increase", "increased",
    "raises", "raise", "raised",
    "adds", "add", "added",
)
DIRECTION_NEGATIVE = (
    "underweights", "underweight", "underweighted",
    "reduces", "reduce", "reduced",
    "decreases", "decrease", "decreased",
    "lowers", "lower", "lowered",
    "trims", "trim", "trimmed",
    "cuts", "cut",
)
# A directional claim is only read where the sentence actually sets itself
# against the classical benchmark, or uses an over/under-weight verb whose
# meaning is inherently relative to it.
CLASSICAL_TOKENS = ("classical", "benchmark")

ATTRIBUTION_TOKENS = ("attribution", "attributions", "contribution",
                      "contributions", "contributor", "contributes",
                      "contributed", "decompose", "decomposes")
VOLATILITY_TOKENS = ("volatility", "volatilities", "vol")
FORECAST_TOKENS = ("forecast", "forecasts", "forecasted", "forecasting",
                   "volatility", "volatilities", "vol", "predicted", "predicts",
                   "predict", "expected", "expects", "expect")
PROFILE_TOKENS = ("conservative", "balanced", "growth", "profile", "profiles")
COVARIANCE_TOKENS = ("covariance", "sigma", "correlation")

# T8 adjudication: a FLAGGED sentence violates iff it attributes
# a forecast value or forecast difference to the profile choice.  The registered
# operationalisation is a causal connective binding the two.
CAUSAL_TOKENS = (
    "because", "since", "due to", "owing to", "as a result", "therefore",
    "thus", "hence", "depends on", "depend on", "dependent on", "driven by",
    "reflects", "reflect", "given your", "given the", "based on your",
    "results from", "leads to", "causes", "cause", "makes", "so that",
)

BOILERPLATE_PATTERNS = (
    "not financial advice", "not investment advice", "generated automatically",
    "automatically generated", "for informational purposes",
    "does not constitute advice", "no advice is given",
)

VERBATIM_LICENSED = (
    "the attributions decompose the advisor-versus-classical difference",
)

# Small registered lexicon of qualitative quantity words.  Recorded as SOFT
# OBSERVATIONS.
QUALITATIVE_LEXICON = (
    "about a third", "a third", "roughly half", "about half", "half",
    "double", "twice", "a quarter", "two-thirds", "three-quarters",
    "most of", "the majority", "a majority", "nearly all", "almost all",
)

# Words that follow "N% of/in/to ..." but name no entity; binding falls through
# to the nearest ticker rather than reading them as a fabricated subject.
NON_ENTITY_WORDS = (
    "portfolio", "universe", "total", "assets", "capital", "allocation",
    "allocations", "weight", "weights", "book", "sleeve", "book.", "it", "its",
    "the", "them", "this", "that", "these", "those", "each", "which",
)

ABBREVIATIONS = ("u.s.", "e.g.", "i.e.", "etc.", "vs.", "inc.", "approx.",
                 "no.", "fig.", "st.", "dr.", "mr.", "ms.")

DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
NUM_RE = re.compile(r"[+-]?\d+(?:,\d{3})*(?:\.\d+)?")
PCT_SUFFIX = re.compile(r"\s*(?:%|percent\b)", re.I)
PP_SUFFIX = re.compile(r"\s*(?:percentage[ -]points?\b|pp\b|ppt\b)", re.I)
PROFILE_FRAME = re.compile(r"\bfor the (conservative|balanced|growth) profile\b", re.I)
CONNECTOR = re.compile(r"\s*(?:to|in|of|for|into)\s+(?:the\s+)?([A-Za-z][A-Za-z0-9._-]*)")


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Claim:
    ctype: str
    subject: str
    basis: str
    value: str
    precision: str
    span: tuple[int, int]
    sentence_index: int

    def tuple4(self) -> tuple[str, str, str, str, str]:
        """The comparable core used by the Gate B1 fixtures (order-insensitive)."""
        return (self.ctype, self.subject, self.basis, self.value, self.precision)


@dataclass
class SentenceRecord:
    index: int
    text: str
    start: int
    end: int
    klass: str                       # claim-bearing | no-claim
    subclass: str                    # boilerplate | registered-verbatim | other | -
    claim_ids: list[int] = field(default_factory=list)
    t8_flagged: bool = False
    t8_violation: bool = False
    t8_evidence: str = "-"
    qualitative: list[str] = field(default_factory=list)


@dataclass
class Extraction:
    text: str
    sentences: list[SentenceRecord]
    claims: list[Claim]
    unaccounted: list[tuple[int, int, str, int]]   # (start, end, token, sentence_index)

    @property
    def t8_violations(self) -> list[SentenceRecord]:
        return [s for s in self.sentences if s.t8_violation]

    @property
    def t8_flagged(self) -> list[SentenceRecord]:
        return [s for s in self.sentences if s.t8_flagged]


@dataclass
class NumericToken:
    start: int
    end: int          # end of the number PLUS any unit suffix
    num_end: int      # end of the digits alone
    text: str         # digits as written, commas stripped
    unit: str         # pct | pp | raw
    decimals: int
    consumed: bool = False

    @property
    def precision(self) -> str:
        return f"{self.unit}.{self.decimals}"


# --------------------------------------------------------------------------- #
# Segmentation and tokenisation
# --------------------------------------------------------------------------- #
def split_sentences(text: str) -> list[tuple[int, int, str]]:
    """Segment on ``[.!?]`` followed by whitespace or end of string.

    A decimal point never ends a sentence because it is followed by a digit;
    a short registered abbreviation list guards the remaining common cases.
    """
    spans: list[tuple[int, int, str]] = []
    start = 0
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in ".!?":
            j = i + 1
            if j >= n or text[j].isspace():
                head = text[start:j]
                low = head.lower()
                if not any(low.endswith(a) for a in ABBREVIATIONS):
                    body = head.strip()
                    if body:
                        off = head.index(body[0]) if body else 0
                        spans.append((start + off, start + off + len(body), body))
                    start = j
        i += 1
    tail = text[start:]
    if tail.strip():
        body = tail.strip()
        off = tail.index(body[0])
        spans.append((start + off, start + off + len(body), body))
    return spans


def numeric_tokens(sentence: str, offset: int, date_spans: list[tuple[int, int]]) -> list[NumericToken]:
    """Every numeric token in a sentence, with its unit and stated precision.

    Date tokens are masked out first so ``2024-12-02`` is one token rather than
    three numbers.
    """
    out: list[NumericToken] = []
    for m in NUM_RE.finditer(sentence):
        a, b = m.start() + offset, m.end() + offset
        if any(ds <= a and b <= de for ds, de in date_spans):
            continue
        raw = m.group(0)
        rest = sentence[m.end():]
        unit, end = "raw", b
        pp = PP_SUFFIX.match(rest)
        pc = PCT_SUFFIX.match(rest)
        if pp:
            unit, end = "pp", b + pp.end()
        elif pc:
            unit, end = "pct", b + pc.end()
        digits = raw.replace(",", "")
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        out.append(NumericToken(a, end, b, digits, unit, decimals))
    return out


def _ticker_spans(sentence: str, offset: int, tickers: tuple[str, ...]) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for t in tickers:
        for m in re.finditer(rf"\b{re.escape(t)}\b", sentence):
            spans.append((m.start() + offset, m.end() + offset, t))
    return sorted(spans)


def _nearest_after(spans: list[tuple[int, int, str]], pos: int) -> str | None:
    for s, _e, t in spans:
        if s >= pos:
            return t
    return None


def _nearest_before(spans: list[tuple[int, int, str]], pos: int) -> str | None:
    best = None
    for _s, e, t in spans:
        if e <= pos:
            best = t
    return best


def _has(low: str, tokens: tuple[str, ...]) -> bool:
    return any(re.search(rf"\b{re.escape(tok)}\b", low) for tok in tokens)


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def extract(text: str, tickers: tuple[str, ...]) -> Extraction:
    sentences: list[SentenceRecord] = []
    claims: list[Claim] = []
    unaccounted: list[tuple[int, int, str, int]] = []

    date_spans = [(m.start(), m.end()) for m in DATE_RE.finditer(text)]

    for idx, (s0, s1, stext) in enumerate(split_sentences(text)):
        low = stext.lower()
        rec = SentenceRecord(index=idx, text=stext, start=s0, end=s1,
                             klass="no-claim", subclass="other")
        local_dates = [(a, b) for a, b in date_spans if s0 <= a and b <= s1]
        nums = numeric_tokens(stext, s0, local_dates)
        tspans = _ticker_spans(stext, s0, tickers)
        my_claims: list[Claim] = []

        # ---- T8 screen: profile token AND forecast token in the same sentence
        if _has(low, PROFILE_TOKENS) and _has(low, FORECAST_TOKENS):
            rec.t8_flagged = True
            causal = [c for c in CAUSAL_TOKENS if c in low]
            profile_pos = min(
                (m.start() for tok in PROFILE_TOKENS
                 for m in re.finditer(rf"\b{tok}\b", low)), default=10**9)
            forecast_pos = min(
                (m.start() for tok in FORECAST_TOKENS
                 for m in re.finditer(rf"\b{tok}\b", low)), default=10**9)
            rec.t8_violation = bool(causal) and profile_pos < forecast_pos
            rec.t8_evidence = (
                f"causal={causal or 'none'}; profile@{profile_pos}; "
                f"forecast@{forecast_pos}"
            )

        # ---- soft observations (recorded, never adjudicated)
        rec.qualitative = [q for q in QUALITATIVE_LEXICON
                           if re.search(rf"\b{re.escape(q)}\b", low)]

        # ---- registered-verbatim short circuit: licensed, and no-claim
        norm = re.sub(r"[^a-z ]", "", low).strip()
        if any(norm == re.sub(r"[^a-z ]", "", v).strip() for v in VERBATIM_LICENSED):
            rec.subclass = "registered-verbatim"
            sentences.append(rec)
            continue

        attribution_ctx = _has(low, ATTRIBUTION_TOKENS)
        volatility_ctx = _has(low, VOLATILITY_TOKENS)

        # ---- T6 provenance -------------------------------------------------- #
        for m in PROFILE_FRAME.finditer(stext):
            my_claims.append(Claim("T6", "profile", "-", m.group(1).lower(), "-",
                                   (s0 + m.start(), s0 + m.end()), idx))
        for a, b in local_dates:
            my_claims.append(Claim("T6", "date", "-", text[a:b], "-", (a, b), idx))
        if _has(low, COVARIANCE_TOKENS):
            my_claims.append(Claim("T6", "covariance_source", "-", stext, "-",
                                   (s0, s1), idx))

        # ---- T3 attribution numerics ---------------------------------------- #
        out_t = _output_ticker(stext, s0, tickers)
        in_t = _input_ticker(stext, s0, tickers)
        if attribution_ctx:
            for tok in nums:
                if tok.consumed or tok.unit not in ("pp", "pct"):
                    continue
                subject = in_t or _nearest_after(tspans, tok.end) or \
                    _nearest_before(tspans, tok.start)
                basis = out_t or "UNRESOLVED_OUTPUT"
                if subject is None:
                    continue
                tok.consumed = True
                my_claims.append(Claim("T3", subject, basis, tok.text,
                                       tok.precision, (tok.start, tok.end), idx))

        # ---- T2 forecast-volatility numerics -------------------------------- #
        if volatility_ctx:
            for tok in nums:
                if tok.consumed:
                    continue
                subject = _nearest_after(tspans, tok.end) or \
                    _nearest_before(tspans, tok.start)
                if subject is None:
                    continue
                tok.consumed = True
                my_claims.append(Claim("T2", subject, "forecast_vol", tok.text,
                                       tok.precision, (tok.start, tok.end), idx))

        # ---- T1 weight numerics / T7 fabricated ----------------------------- #
        for tok in nums:
            if tok.consumed:
                continue
            entity = _bind_entity(text, tok, tspans, tickers)
            if entity is None:
                continue
            tok.consumed = True
            if entity in tickers:
                my_claims.append(Claim("T1", entity, "advisor_weight", tok.text,
                                       tok.precision, (tok.start, tok.end), idx))
            else:
                my_claims.append(Claim("T7", entity, "unknown_ticker", tok.text,
                                       tok.precision, (tok.start, tok.end), idx))

        # ---- T4 directional -------------------------------------------------- #
        dir_tokens = [(m.start() + s0, w, +1)
                      for w in DIRECTION_POSITIVE
                      for m in re.finditer(rf"\b{w}\b", low)]
        dir_tokens += [(m.start() + s0, w, -1)
                       for w in DIRECTION_NEGATIVE
                       for m in re.finditer(rf"\b{w}\b", low)]
        if dir_tokens and (_has(low, CLASSICAL_TOKENS) or
                           any(w.startswith(("over", "under")) for _p, w, _s in dir_tokens)):
            basis = "attribution_sign" if attribution_ctx else "w_adv_minus_w_cls"
            for pos, word, sign in sorted(dir_tokens):
                subject = _nearest_after(tspans, pos + len(word))
                if subject is None:
                    continue
                my_claims.append(Claim(
                    "T4", subject, basis, "positive" if sign > 0 else "negative",
                    "-", (pos, pos + len(word)), idx))

        # ---- T5 ranking ------------------------------------------------------- #
        for word, direction in [(w, "max") for w in RANKING_MAX] + \
                               [(w, "min") for w in RANKING_MIN]:
            for m in re.finditer(rf"\b{word}\b", low):
                pos = m.start() + s0
                if attribution_ctx:
                    qset = f"attributions[output={out_t or 'UNRESOLVED_OUTPUT'}]"
                    subject = in_t or _nearest_after(tspans, pos)
                    basis = f"{word}_abs_contribution"
                elif volatility_ctx:
                    qset, basis = "forecast_vols", word
                    subject = _nearest_before(tspans, pos) or _nearest_after(tspans, pos)
                else:
                    qset, basis = "advisor_weights", word
                    subject = _nearest_before(tspans, pos) or _nearest_after(tspans, pos)
                if subject is None:
                    continue
                my_claims.append(Claim("T5", subject, basis, qset, "-",
                                       (pos, pos + len(word)), idx))

        # ---- unaccounted numerics -------------------------------------------- #
        for tok in nums:
            if not tok.consumed:
                unaccounted.append((tok.start, tok.end, text[tok.start:tok.end], idx))

        if my_claims:
            rec.klass, rec.subclass = "claim-bearing", "-"
        elif any(p in low for p in BOILERPLATE_PATTERNS):
            rec.subclass = "boilerplate"
        base = len(claims)
        rec.claim_ids = list(range(base, base + len(my_claims)))
        claims.extend(my_claims)
        sentences.append(rec)

    return Extraction(text=text, sentences=sentences, claims=claims,
                      unaccounted=unaccounted)


def _output_ticker(stext: str, offset: int, tickers: tuple[str, ...]) -> str | None:
    """The OUTPUT ticker of an attribution claim: ``to the <T> weight difference``."""
    pat = re.compile(
        rf"\b(?:to|on|in)\s+(?:the\s+)?({'|'.join(map(re.escape, tickers))})"
        r"(?:'s)?\s+(?:weight|allocation|position)?\s*(?:difference|gap|deviation)")
    m = pat.search(stext)
    if m:
        return m.group(1)
    m2 = re.search(
        rf"\b({'|'.join(map(re.escape, tickers))})(?:'s)?\s+"
        r"(?:weight|allocation|position)\s+(?:difference|gap|deviation)", stext)
    return m2.group(1) if m2 else None


def _input_ticker(stext: str, offset: int, tickers: tuple[str, ...]) -> str | None:
    """The INPUT ticker of an attribution claim: ``from the <T> forecast``."""
    m = re.search(
        rf"\bfrom\s+(?:the\s+)?({'|'.join(map(re.escape, tickers))})", stext)
    return m.group(1) if m else None


def _bind_entity(text: str, tok: NumericToken,
                 tspans: list[tuple[int, int, str]],
                 tickers: tuple[str, ...]) -> str | None:
    """Bind a numeric to the entity it quantifies.

    Registered order: the connector phrase immediately after the numeric
    (``to``/``in``/``of``/``for``/``into`` + word), then the nearest following
    ticker, then the nearest preceding ticker.  A connector word drawn from the
    registered non-entity list (``portfolio``, ``total``, ...) names no entity
    and falls through rather than being read as a fabricated subject.
    """
    m = CONNECTOR.match(text[tok.end:])
    if m:
        word = m.group(1).rstrip(".,;:")
        if word.lower() not in NON_ENTITY_WORDS:
            for t in tickers:
                if word.upper() == t:
                    return t
            return word
    return _nearest_after(tspans, tok.end) or _nearest_before(tspans, tok.start)
