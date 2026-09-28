# Task 14 Stage A -- machinery: pin, rubric, parsers, personas

Environment pin (registered convention): everything evidentiary runs under the repo venv; the system interpreter is never used.

- interpreter: `C:\Users\tianc\Desktop\Final Project\etf-advisor\.venv\Scripts\python.exe`
- python 3.13.0, numpy 2.4.6, pandas 3.0.3, scipy 1.17.1, pyarrow 25.0.0
- Ollama server version: `0.32.5`
- model tag: `qwen3.5:9b`, digest `6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7`
- parameter size `9.7B`, quantization `Q4_K_M`, family `qwen35`, architecture `qwen35`
- options-variant hash: `43dcb68ee2727dea55d2a0c69a2ba53d67f1a9a8ae56449bd7a06b0333ab68f9`

Registered options variant -- the Task 13 frozen block with exactly one delta, `num_predict` 512 -> 128; sent verbatim on every call:

```
model      = 'qwen3.5:9b'
endpoint   = 'http://localhost:11434/api/generate'
think      = False      # invariant I2
stream     = False
keep_alive = '60m'
options    = {'temperature': 0, 'top_k': 1, 'seed': 42, 'num_predict': 128, 'num_ctx': 8192, 'num_gpu': 0}
penalties  = {'presence_penalty': 0.0, 'frequency_penalty': 0.0, 'repeat_penalty': 1.0}
I4 budget  = 8064   # num_ctx - num_predict
```

Determinism standard IN FORCE: **repeat-exact** -- the k = 3 repeats of the registered probe were byte-identical, so the registered k = 3 byte-identical standard is carried forward unmodified.

## 0. Objective and boundary

This module builds BOTH elicitor arms (A: LLM extraction -> deterministic rubric; B: direct LLM mapping), evaluate them one-shot under the register's protocol, select per the registered rule, and ship the winner as `src/task14/elicit.py`.

> The Task 12 and Task 13 verdicts are frozen ground truth, and all Task 9-13 outputs are read-only. The elicitor creates no new profile, ladder, or budget freedom: its output is exactly one registered profile string from `PROFILES` (src.task9.allocators) or a clarify signal. `recommend()` validation is the backstop, never the primary check. A weak elicitor is a Task 14 finding about the elicitation layer and re-litigates nothing upstream.

## 1. Register intake

### Registered question set, read back from the code

| # | question (asked verbatim) | extraction domain |
|---|---|---|
| Q1 | When do you expect to need most of this money? | `short`, `medium`, `long`, `unclear` |
| Q2 | Suppose your portfolio lost 20% of its value within a year. What would you most likely do? | `sell`, `hold`, `buy_more`, `unclear` |
| Q3 | What matters more to you: protecting what you have, or growing it as much as possible over time? | `preserve`, `balanced`, `grow`, `unclear` |
| Q4 | How stable is your income, and do you have an emergency fund covering several months of expenses? | `low`, `medium`, `high`, `unclear` |
| Q5 | Have you invested in markets before, and how comfortable are you with market fluctuations? | `none`, `some`, `experienced`, `unclear` |

## 2. Pin re-verification and the options variant

The pin is carried from the Task 13 closing record. Each value is checked three ways: against the knob table, against the live host, and for verbatim presence in both Task 13 stage reports on disk.

| pin field | registered | live host | verbatim in Task 13 reports | drift |
|---|---|---|---|---|
| Ollama server version | `0.32.5` | `0.32.5` | yes | none |
| model tag | `qwen3.5:9b` | `qwen3.5:9b` | yes | none |
| model digest | `6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7` | `6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7` | yes | none |

Pin fields captured: 20; reported ABSENT: none.

### Options-block variant

| item | value |
|---|---|
| Task 13 frozen block, canonical JSON SHA-256 | `aaafb46f9870a225f2aec4691e228dc209eea446917a247851ed0fc3cfe8f4b5` |
| that hash appears verbatim in `outputs/task13/stage_c_report.md` | yes |
| Task 14 options-variant SHA-256 | `43dcb68ee2727dea55d2a0c69a2ba53d67f1a9a8ae56449bd7a06b0333ab68f9` |
| delta between the two canonical payloads | `{'num_predict': (512, 128)}` |
| delta is num_predict-only | yes |
| I4 budget for this task | `8064` = num_ctx 8192 - num_predict 128 |
| static prompt bound | `12000` chars |

The Task 14 fingerprint is computed by a function structurally identical to `src.task13.options_fingerprint` (same keys, same canonical-JSON settings), and that function reproduces the Task 13 hash recorded at Task 13 Gate B3. The only thing that can move the digest is the one registered knob.

## 3. Determinism probe

Registered probe prompt, reused verbatim from Task 13:

> Describe in exactly three sentences what a diversified ETF portfolio is.

One discarded warm-up call, then k = 3 warm generations under the registered options variant.

| seq | purpose | prompt_eval_count | I4 (<= 8064) | eval_count | done_reason | load_duration_s | wall_s | eval tok/s |
|---|---|---|---|---|---|---|---|---|
| 1 | stage_a:warmup_discarded | 24 | PASS | 81 | `stop` | 0.59 | 13.93 | 7.38 |
| 2 | stage_a:probe_rep1 | 24 | PASS | 81 | `stop` | 0.56 | 13.84 | 7.44 |
| 3 | stage_a:probe_rep2 | 24 | PASS | 81 | `stop` | 0.57 | 13.74 | 7.46 |
| 4 | stage_a:probe_rep3 | 24 | PASS | 81 | `stop` | 0.58 | 13.83 | 7.39 |

| invariant | exercised on | outcome |
|---|---|---|
| I1 model == pinned tag | 4 calls | held on all |
| I2 no thinking content | 4 calls | held on all |
| I3 done == true | 4 calls | held on all |
| I4 prompt_eval_count <= 8064 | 4 calls | held on all |

I2 evidence strings, per call: ['none', 'none', 'none', 'none'].

| repeat | response chars | sha256 (16) | == rep1 |
|---|---|---|---|
| 1 | 495 | 0da353ea288fc516 | yes |
| 2 | 495 | 0da353ea288fc516 | yes |
| 3 | 495 | 0da353ea288fc516 | yes |

**Byte-identical across k = 3: YES.** First divergence: -.

Determinism standard IN FORCE: **repeat-exact**. Under the registered posture this is a MEASUREMENT, not a pass/fail criterion; Gate A1 requires only that it ran and is recorded. Both arms will be evaluated under whichever standard is in force.

### Recorded observation -- the same probe under the Task 13 block

Task 13 ran this identical probe prompt under its frozen options block and kept the transcript. Comparing the two is free evidence about what the registered variant actually changes. This feeds NO gate and is a recorded observation only.

| item | Task 13 (num_predict 512) | Task 14 (num_predict 128) |
|---|---|---|
| probe prompt | identical (registered, reused verbatim) | identical |
| model tag + digest | identical | identical |
| server version | identical | identical |
| temperature / top_k / seed | 0 / 1 / 42 | 0 / 1 / 42 |
| response chars | 553 | 495 |
| response sha256 (16) | `e478f60d4193b77a` | `0da353ea288fc516` |
| byte-identical to the other task | **no** | **no** |

The two transcripts DIFFER, first at character offset 28 (common prefix `A diversified ETF portfolio `). Every other input is held constant -- same prompt, model digest, server version, temperature 0, top_k 1, seed 42, penalties -- so the registered `num_predict` 512 -> 128 delta is the only candidate cause, and greedy decoding is evidently not invariant to it on this server build. Consequences, recorded: (i) Task 13 transcripts are NOT reproducible under the Task 14 variant, which is why the options-variant hash is recorded in every header and why the variant is frozen for the whole task; (ii) the within-task standard is unaffected -- the k = 3 repeats under the variant were byte-identical; (iii) no Task 13 result is disturbed, since nothing here re-runs Task 13. This is an observation about the environment, not a finding about either arm, and it feeds no gate.

Runtime estimate for Stage C from the measured decode rate (7.43 eval tok/s, mean over 3 probe repeats): 84 generations x `num_predict` 128 = 24 min upper bound if every generation runs to the token cap, less where the model stops earlier.

### Probe transcript, repeat 1 (quoted ASCII-escaped; raw bytes on disk)

```
A diversified ETF portfolio is a collection of exchange-traded funds that collectively hold a wide variety of assets across different sectors, geographies, and investment styles. This structure allows investors to gain broad market exposure while significantly reducing the risk associated with any single company or industry. By spreading capital across multiple funds, the portfolio aims to smooth out volatility and provide more stable long-term returns compared to holding individual stocks.
```

Non-ASCII characters in the quoted transcript: 0 (`outputs/task14/transcripts/probe_rep1.txt` holds the raw text).

## 4. Rubric implemented and validated vs R1-R9

The point table is DERIVED from the registered domain listings rather than retyped (*each answer scores 0, 1, or 2 in the order its domain is listed*), so an ordering slip cannot hide in a second copy. Every registered intermediate is compared, not just the final profile: columns show `registered / computed`.

| id | answers (Q1..Q5) | score | score_profile | cap | expected profile | verdict |
|---|---|---|---|---|---|---|
| R1 | short, sell, preserve, low, none | 0 / 0 | conservative / conservative | conservative / conservative | conservative / conservative | PASS |
| R2 | long, buy_more, grow, high, experienced | 10 / 10 | growth / growth | growth / growth | growth / growth | PASS |
| R3 | medium, hold, preserve, medium, none | 3 / 3 | conservative / conservative | balanced / balanced | conservative / conservative | PASS |
| R4 | medium, hold, preserve, medium, some | 4 / 4 | balanced / balanced | balanced / balanced | balanced / balanced | PASS |
| R5 | long, hold, balanced, high, some | 7 / 7 | balanced / balanced | growth / growth | balanced / balanced | PASS |
| R6 | long, buy_more, balanced, high, some | 8 / 8 | growth / growth | growth / growth | growth / growth | PASS |
| R7 | short, buy_more, grow, high, experienced | 8 / 8 | growth / growth | conservative / conservative | conservative / conservative | PASS |
| R8 | long, buy_more, grow, low, experienced | 8 / 8 | growth / growth | balanced / balanced | balanced / balanced | PASS |
| R9 | medium, buy_more, grow, high, experienced | 9 / 9 | growth / growth | balanced / balanced | balanced / balanced | PASS |

R3/R4 exercise the 3/4 threshold; R5/R6 the 7/8 threshold; R7-R9 the caps.

Rubric fixtures exact: **9/9**. Threshold pairs exercised: R3/R4 (3->4) and R5/R6 (7->8); cap fixtures exercised: R7 (short horizon), R8 (low capacity), R9 (medium horizon).

## 5. Parsers implemented and validated vs PF1-PF8

Registered decisions on two points the register leaves open, both fixed here before any evaluation call exists:

- **D-A1 (outer whitespace, Arm A).** The register spells out per-line stripping but not the whole-output boundary, so a single trailing newline -- a transport artifact, not a model content choice -- would otherwise read as a sixth line and force MALFORMED. The parser strips the whole response first, then splits, then strips each line. Interior blank lines still yield more than five lines and stay MALFORMED. The registered Arm B rule opens with exactly this move (*strip surrounding whitespace*), so applying it to Arm A keeps the arms level at the boundary.
- **D-A2 (no case folding, Arm A).** The register lists `lowercase` as an Arm B normalization step and does NOT list it for Arm A. Implemented literally: `Q1: Medium` is MALFORMED on Arm A while `Balanced.` is VALID on Arm B. This is a registered asymmetry in strictness between the arms, carried as registered rather than harmonised.

| id | arm | raw model output | registered expectation | observed | parser reason | verdict |
|---|---|---|---|---|---|---|
| PF1 | A | `Q1: medium\nQ2: hold\nQ3: balanced\nQ4: medium\nQ5: some` | VALID extraction | VALID extraction | five valid lines | PASS |
| PF2 | A | `Q1: medium\nQ2: hold\nQ3: balanced\nQ4: medium` | MALFORMED | MALFORMED | line count 4 != 5 | PASS |
| PF3 | A | `Q1: about five years so medium` | MALFORMED | MALFORMED | line count 1 != 5 | PASS |
| PF4 | A | `Q1: unclear\nQ2: hold\nQ3: balanced\nQ4: medium\nQ5: some` | VALID -> clarify(Q1) | VALID -> clarify(Q1) | five valid lines | PASS |
| PF5 | B | `balanced` | VALID: balanced | VALID: balanced | exact match after registered normalization | PASS |
| PF6 | B | `Balanced.` | VALID: balanced (normalization) | VALID: balanced | exact match after registered normalization | PASS |
| PF7 | B | `I would say balanced` | MALFORMED | MALFORMED | normalized 'i would say balanced' not in ('conservative', 'balanced', 'growth', 'clarify') | PASS |
| PF8 | B | `balanced or growth` | MALFORMED | MALFORMED | normalized 'balanced or growth' not in ('conservative', 'balanced', 'growth', 'clarify') | PASS |

Parser fixtures exact: **8/8**. PF2 and PF4 are given in the register as prose (*four lines only (Q5 missing)*, *`Q1: unclear` + four valid lines*) and are constructed from PF1's registered lines, so the construction is itself register-sourced; the other six raws are registered literals and are transcribed verbatim.

### Supplementary parser checks (NOT part of the registered 8)

PF3 as registered is a one-line raw, so it is MALFORMED on the line-count mechanism and the domain-match mechanism at once. These cases isolate each mechanism so the parser cannot pass PF3 for the wrong reason. They add no criterion and change no fixture.

| id | arm | raw | expected behaviour | observed | verdict |
|---|---|---|---|---|---|
| S1 | A | `Q1: about five years so medium\nQ2: hold\nQ3: balanced\nQ4: medium\nQ5: some` | MALFORMED -- five lines, Q1 value outside the domain (isolates the domain-match mechanism from PF3's line count) | MALFORMED | PASS |
| S2 | A | `Q1: medium\nQ2: hold\nQ3: balanced\nQ4: medium\nQ5: some\n` | VALID -- one trailing newline is a transport artifact, absorbed by the D-A1 whole-output strip | VALID | PASS |
| S3 | A | `Q1: Medium\nQ2: hold\nQ3: balanced\nQ4: medium\nQ5: some` | MALFORMED -- Arm A does not case-fold (decision D-A2) | MALFORMED | PASS |
| S4 | B | `balanced .` | MALFORMED -- the registered Arm B order strips whitespace once, before the trailing '.', and does not re-strip | MALFORMED | PASS |

Supplementary checks: **4/4**.

## 6. Personas transcribed and cross-checked

| item | value |
|---|---|
| persona file | `src/task14/personas.json` |
| personas | 17 = 3 dev + 9 clear-cut + 5 ambiguous/edge |
| evaluation personas | 14 |
| counts match register Sec. 4 | yes |
| encoding | ASCII file; U+2014 held as JSON `\u2014` escapes, decoding to the register's characters verbatim |

| id | kind | expected extraction | expected profile / acceptable set | answer chars | non-ASCII | register needles found |
|---|---|---|---|---|---|---|
| D1 | dev | short/sell/preserve/low/none | conservative | 158 | 0 | 7/7 |
| D2 | dev | medium/hold/balanced/medium/some | balanced | 120 | 0 | 7/7 |
| D3 | dev | long/buy_more/grow/high/experienced | growth | 128 | 0 | 7/7 |
| P-C1 | clear_cut | short/sell/preserve/low/none | conservative | 244 | 0 | 7/7 |
| P-C2 | clear_cut | short/buy_more/grow/high/experienced | conservative | 145 | 0 | 7/7 |
| P-C3 | clear_cut | medium/sell/preserve/low/none | conservative | 169 | 1 | 7/7 |
| P-B1 | clear_cut | medium/hold/balanced/medium/some | balanced | 144 | 0 | 7/7 |
| P-B2 | clear_cut | long/hold/balanced/medium/some | balanced | 210 | 2 | 7/7 |
| P-B3 | clear_cut | medium/buy_more/preserve/high/some | balanced | 178 | 1 | 7/7 |
| P-G1 | clear_cut | long/buy_more/grow/high/experienced | growth | 209 | 2 | 7/7 |
| P-G2 | clear_cut | long/buy_more/grow/high/some | growth | 189 | 1 | 7/7 |
| P-G3 | clear_cut | long/hold/grow/high/experienced | growth | 161 | 1 | 7/7 |
| P-X1 | ambiguous | - | {conservative, balanced, clarify} | 182 | 0 | 8/8 |
| P-X2 | ambiguous | - | {balanced, growth, clarify} | 125 | 0 | 8/8 |
| P-X3 | ambiguous | - | {clarify} | 149 | 0 | 7/7 |
| P-X4 | ambiguous | - | {clarify} | 146 | 0 | 7/7 |
| P-X5 | ambiguous | - | {growth} | 159 | 2 | 7/7 |

### Register self-consistency of the clear-cut and dev expectations

The registered expected extraction, pushed through the fixture-validated rubric, must reproduce the registered expected profile. A mismatch here is the Stage C HALT-AND-ESCALATE combination surfacing at Stage A, where it is still cheap; it is checked now so that a Stage C occurrence can only be a genuine anomaly.

| persona | registered extraction | score | score_profile | cap | rubric profile | registered profile | verdict |
|---|---|---|---|---|---|---|---|
| D1 | short/sell/preserve/low/none | 0 | conservative | conservative | conservative | conservative | PASS |
| D2 | medium/hold/balanced/medium/some | 5 | balanced | balanced | balanced | balanced | PASS |
| D3 | long/buy_more/grow/high/experienced | 10 | growth | growth | growth | growth | PASS |
| P-C1 | short/sell/preserve/low/none | 0 | conservative | conservative | conservative | conservative | PASS |
| P-C2 | short/buy_more/grow/high/experienced | 8 | growth | conservative | conservative | conservative | PASS |
| P-C3 | medium/sell/preserve/low/none | 1 | conservative | balanced | conservative | conservative | PASS |
| P-B1 | medium/hold/balanced/medium/some | 5 | balanced | balanced | balanced | balanced | PASS |
| P-B2 | long/hold/balanced/medium/some | 6 | balanced | growth | balanced | balanced | PASS |
| P-B3 | medium/buy_more/preserve/high/some | 6 | balanced | balanced | balanced | balanced | PASS |
| P-G1 | long/buy_more/grow/high/experienced | 10 | growth | growth | growth | growth | PASS |
| P-G2 | long/buy_more/grow/high/some | 9 | growth | growth | growth | growth | PASS |
| P-G3 | long/hold/grow/high/experienced | 9 | growth | growth | growth | growth | PASS |

### Mechanical cross-check against the register text

The imported-never-retyped doctrine, enforced. Every fixture value in this package -- each question, each point in the mapping table, each threshold and cap line, each R and PF row, each rule sentence, and for every persona each free-text answer, each expected extraction value, each expected profile and each acceptable set -- is rebuilt from the code/JSON and required to appear verbatim in the register file. The register is hard-wrapped at ~72 columns, so a registered string can straddle a line break; both sides of every comparison are therefore whitespace-normalised (runs of whitespace collapsed to one space) and nothing else is touched.

| category | needles | found | missing |
|---|---|---|---|
| question set | 5 | 5 | 0 |
| rubric mapping | 13 | 13 | 0 |
| rubric fixtures R1-R9 | 9 | 9 | 0 |
| parser fixtures PF1-PF8 | 8 | 8 | 0 |
| registered rules | 13 | 13 | 0 |
| personas | 121 | 121 | 0 |

**Total needles checked: 169; mismatches: 0.** Full needle-by-needle results: `outputs/task14/register_crosscheck.csv`.

## 7. Static import audit over `src/task14`

Registered surface: stdlib, plus `src.task13` (frozen constants carried never-retyped, and `client.py` for all transport), plus `src.task9.allocators` for `PROFILES` only. `stdlib` is decided by `sys.stdlib_module_names`, not by a hand-kept list. Package versions in the header are read with `importlib.metadata` rather than by importing numpy/pandas/scipy/pyarrow, so this package's own import list stays stdlib-only beyond the three frozen targets.

| file | statement | target | class |
|---|---|---|---|
| __init__.py | `from __future__ import annotations` | `__future__` | stdlib |
| __init__.py | `import hashlib` | `hashlib` | stdlib |
| __init__.py | `import json` | `json` | stdlib |
| __init__.py | `import sys` | `sys` | stdlib |
| __init__.py | `import time` | `time` | stdlib |
| __init__.py | `import urllib.error` | `urllib` | stdlib |
| __init__.py | `from datetime import datetime, timezone` | `datetime` | stdlib |
| __init__.py | `from pathlib import Path` | `pathlib` | stdlib |
| __init__.py | `from typing import Any, Sequence` | `typing` | stdlib |
| __init__.py | `from ..task13 import ENDPOINT, KEEP_ALIVE, MODEL, OPTIONS, PENALTY_KNOBS, PENALTY_KNOBS_FROZEN_AT, PROBE_PROMPT, K_REPEATS, B2_PROMPT_CHAR_BOUND, STREAM, THINK, VERSION_ENDPOINT` | `src.task13` | frozen-allowed |
| __init__.py | `from ..task13 import client` | `src.task13` | frozen-allowed |
| __init__.py | `from ..task9.allocators import PROFILES` | `src.task9.allocators` | frozen-allowed |
| __init__.py | `import importlib.metadata` | `importlib` | stdlib |
| __init__.py | `import ast` | `ast` | stdlib |
| parse.py | `from __future__ import annotations` | `__future__` | stdlib |
| parse.py | `from dataclasses import dataclass` | `dataclasses` | stdlib |
| parse.py | `from . import ARM_B_VOCAB, EXTRACTION_DOMAINS, Q_KEYS` | `src.task14.` | frozen-allowed |
| parse.py | `from . import rubric` | `src.task14.` | frozen-allowed |
| rubric.py | `from __future__ import annotations` | `__future__` | stdlib |
| rubric.py | `from dataclasses import dataclass` | `dataclasses` | stdlib |
| rubric.py | `from . import CLARIFY, DOMAINS, PROFILE_ORDER, Q_KEYS, UNCLEAR` | `src.task14.` | frozen-allowed |
| stage_a.py | `from __future__ import annotations` | `__future__` | stdlib |
| stage_a.py | `import csv` | `csv` | stdlib |
| stage_a.py | `import json` | `json` | stdlib |
| stage_a.py | `import sys` | `sys` | stdlib |
| stage_a.py | `from . import ALLOWED_NON_STDLIB, ARM_B_VOCAB, DOMAINS, EXTRACTION_DOMAINS, FORBIDDEN_IMPORTS, I4_PROMPT_BUDGET, K_REPEATS, MODEL, NUM_PREDICT, OUT_DIR, PERSONA_PATH, PIN_MODEL_DIGEST, PIN_SERVER_VERSION, PROBE_PROMPT, PROFILE_ORDER, PROMPT_CHAR_BOUND, QUESTIONS, Q_KEYS, REGISTER_PATH, T13_PROBE_REP1, T13_STAGE_A_REPORT, T13_STAGE_C_REPORT, TRANSCRIPT_DIR, ascii_escape, classify_import, env_header, generate, log_row, md_table, normalize_ws, options_fingerprint, package_imports, read_live_pin, sha256_file, sha256_text, write_report, LOG_COLUMNS` | `src.task14.` | frozen-allowed |
| stage_a.py | `from . import parse` | `src.task14.` | frozen-allowed |
| stage_a.py | `from . import rubric` | `src.task14.` | frozen-allowed |
| stage_a.py | `from ..task13 import options_fingerprint` | `src.task13` | frozen-allowed |

Authorized non-stdlib targets: ['src.task14', 'src.task13', 'src.task13.client', 'src.task9.allocators']. Forbidden targets ['shap', 'torch', 'requests', 'ollama']: 0 present. Unauthorized targets: 0.

Addition to the file list, recorded: `src/task14/__init__.py`. The substantive modules are already listed; this is the package file that carries them and holds the registered constants and report plumbing, mirroring `src/task13/__init__.py`.

## 8. Gates

| gate | criterion | evidence | verdict |
|---|---|---|---|
| A1 | register hashed; pin verified with zero drift; variant hash recorded; determinism probe executed and standard in force recorded | pin drift 0; variant hash `43dcb68ee2727dea...`; 4 probe calls; standard recorded in Sec. 3 | PASS |
| A2 | rubric fixtures 9/9 exact, incl. both threshold pairs and all three cap fixtures | 9/9 exact, every registered intermediate (score, score_profile, cap, profile) compared | PASS |
| A3 | parser fixtures 8/8 exact; personas cross-check 0 mismatches; all hashes recorded; import audit clean | 8/8 parser fixtures; 169 register needles, 0 missing; persona counts ok; 0 structural issues; 0 disallowed imports | PASS |

## 9. Deliverables written by this stage

- `src/task14/__init__.py`, `rubric.py`, `parse.py`, `personas.json`, `stage_a.py`
- `outputs/task14/stage_a_report.md` (this file)
- `outputs/task14/register_crosscheck.csv` (169 needles; addition to the data list, recorded)
- `outputs/task14/stage_a_call_log.csv` (4 calls)
- `outputs/task14/determinism_log.csv` (probe repeats)
- `outputs/task14/transcripts/probe_rep{1,2,3}.txt`, `probe_warmup_discarded.txt`

## 10. Stop line

STAGE A COMPLETE -- gates A1/A2/A3 PASS. No evaluation persona has been sent to the model. 
