# Task 14 Stage A ADDENDUM -- re-registered fixture battery, corrected observation

Environment pin (registered convention): everything evidentiary runs under the repo venv; the system interpreter is never used.

- interpreter: `C:\Users\tianc\Desktop\Final Project\etf-advisor\.venv\Scripts\python.exe`
- python 3.13.0, numpy 2.4.6, pandas 3.0.3, scipy 1.17.1, pyarrow 25.0.0
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

Determinism standard IN FORCE: **repeat-exact** (measured at Stage A and unchanged; this addendum issues ZERO generations).

## 1. T14-A1 fixture battery (no LLM involved)

T14-A1 harmonises Arm A per-line parsing with Arm B's registered normalization. `parse.py` now routes BOTH arms through one `registered_normalization()` function -- strip surrounding whitespace, strip one trailing `.`, lowercase, in that order and no other -- so no strictness difference between the arms can survive in the parser layer.

### PF1-PF8 -- must still pass 8/8 unchanged

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

**PF fixtures after T14-A1: 8/8** (expectations unchanged by the amendment).

### Battery S1-S7

| id | status | arm | raw | expected | observed | note | verdict |
|---|---|---|---|---|---|---|---|
| S1 | supplementary | A | `Q1: about five years so medium\nQ2: hold\nQ3: balanced\nQ4: medium\nQ5: some` | MALFORMED | MALFORMED | five lines, Q1 value outside the domain after normalization; isolates the domain-match mechanism from PF3's line count. Extra words remain MALFORMED under T14-A1. | PASS |
| S2 | supplementary | A | `Q1: medium\nQ2: hold\nQ3: balanced\nQ4: medium\nQ5: some\n` | VALID medium/hold/balanced/medium/some | VALID medium/hold/balanced/medium/some | one trailing newline is a transport artifact, absorbed by the D-A1 whole-output strip (D-A1 stands, ratified as S14-E1). | PASS |
| S3 | T14-A1 | A | `Q1: Medium\nQ2: hold\nQ3: balanced\nQ4: medium\nQ5: some` | VALID medium/hold/balanced/medium/some | VALID medium/hold/balanced/medium/some | capitalised value. Expectation FLIPPED by T14-A1: the value token now receives the registered Arm B normalization, so it lowercases to 'medium'. The Stage A literal-reading result (MALFORMED under D-A2) is preserved in the Stage A report as the pre-amendment record. | PASS |
| S4 | supplementary | B | `balanced .` | MALFORMED | MALFORMED | the registered order strips whitespace once, then one trailing '.', and does not re-strip. | PASS |
| S5 | T14-A1 | A | `q1: medium\nQ2: hold\nQ3: balanced\nQ4: medium\nQ5: some` | VALID medium/hold/balanced/medium/some | VALID medium/hold/balanced/medium/some | lowercase label; the label matches 'Qk' case-insensitively under T14-A1. | PASS |
| S6 | T14-A1 | A | `Q1: medium.\nQ2: hold\nQ3: balanced\nQ4: medium\nQ5: some` | VALID medium/hold/balanced/medium/some | VALID medium/hold/balanced/medium/some | one trailing '.' on the value, stripped by the registered normalization. | PASS |
| S7 | T14-A1 | A | `Q1: medium .\nQ2: hold\nQ3: balanced\nQ4: medium\nQ5: some` | MALFORMED | MALFORMED | mirrors S4: strip whitespace once, strip one trailing '.', no re-strip, so the residual trailing space fails the exact domain match. | PASS |

**T14-A1 re-registered/additional fixtures: 4/4. Full S1-S7 battery: 7/7.**

S3's expectation is FLIPPED by the amendment. The Stage A literal-reading result for the same input is preserved unaltered in the Stage A report as the pre-amendment record.

### Rubric re-validation

R1-R9 re-run against the unchanged rubric: **9/9** exact, every registered intermediate compared. The amendment is confined to `parse.py`; `rubric.py` is byte-unchanged since Stage A ratification.

## 2. Corrected cross-task probe observation

### 2.1 Struck record -- the Stage A text, read from the preserved report

Reproduced verbatim below and STRUCK. It is preserved because the error was one of attribution, not of measurement: the byte comparison it reported was correct; the causal claim drawn from it was not.

```
STRUCK RECORD -- superseded by Sec. 2.3 below

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
```

### 2.2 Why it was wrong

The comparator was Task 13's STAGE A probe transcript. That transcript predates amendment and was generated under the model's SHIPPED penalties (1.5 (model Modelfile)), whereas Task 14 carries the frozen values {'presence_penalty': 0.0, 'frequency_penalty': 0.0, 'repeat_penalty': 1.0}. Penalties were therefore NOT held constant in that pairing, and the claim that `num_predict` was the only candidate cause does not follow. The penalties-constant comparator is Task 13's Gate B3 re-probe.

| transcript | penalties in force | num_predict |
|---|---|---|
| `outputs/task13/narrations/probe_rep1.txt` (T13 Stage A) | shipped defaults: presence_penalty 1.5 (model Modelfile); frequency/repeat unspecified | 512 |
| `outputs/task13/narrations/b3_probe_rep1.txt` (T13 Gate B3) | T13-A1 frozen {'presence_penalty': 0.0, 'frequency_penalty': 0.0, 'repeat_penalty': 1.0} | 512 |
| `outputs/task14/transcripts/probe_rep1.txt` (T14 Stage A) | T13-A1 frozen {'presence_penalty': 0.0, 'frequency_penalty': 0.0, 'repeat_penalty': 1.0}, carried | 128 |

### 2.3 Three-way byte comparison over the on-disk transcripts

| transcript | bytes | sha256 |
|---|---|---|
| `T13 Stage A probe rep1` | 553 | `e478f60d4193b77aae0b3489f60f4bec7e7da8bf407ea51b3e83d4e420162a13` |
| `T13 Gate B3 probe rep1` | 495 | `0da353ea288fc5165e3bf5151aeba1f3c90696b03eac1127f18e8e644e1cc41a` |
| `T14 Stage A probe rep1` | 495 | `0da353ea288fc5165e3bf5151aeba1f3c90696b03eac1127f18e8e644e1cc41a` |

| pair | byte-identical | first divergence |
|---|---|---|
| T13 Stage A vs T13 Gate B3 | **no** | byte 28 |
| T13 Stage A vs T14 Stage A | **no** | byte 28 |
| T13 Gate B3 vs T14 Stage A | yes | - |

Distinct transcripts among the three: **2**.

### 3.4 Branch obtained: **WITHDRAWN**

`T14 == T13-B3` byte-for-byte, so the registered first branch obtains and the num_predict-sensitivity claim is **WITHDRAWN**.

Recorded:

> Under the penalties-constant comparison, the Task 14 options variant reproduces the Task 13 frozen-config probe **byte-identically across tasks and across processes**: `b3_probe_rep1.txt` and `probe_rep1.txt` share SHA-256 `0da353ea288fc5165e3bf5151aeba1f3c90696b03eac1127f18e8e644e1cc41a` despite differing in `num_predict` (512 vs 128). The divergence measured at Stage A -- first at byte 28 against Task 13's Stage A transcript -- is the already-recorded T13-A1 penalty effect (shipped `presence_penalty` 1.5 versus the frozen 0.0), not a `num_predict` effect.

Two consequences follow, and both are strictly better than what the struck text asserted. First, greedy decoding on this build is evidently INVARIANT to `num_predict` for this prompt, so the registered variant costs nothing in comparability. Second, the Task 13 Gate B3 transcript is reproducible from Task 14's configuration, which is a cross-task reproducibility result the struck text had mistaken for a sensitivity finding.

The Task 15 rider registered alongside this correction is unaffected by the branch and stands as registered: the Task 13 closing-record inheritance-5 byte-match smoke check MUST run under the Task 13 frozen options block verbatim (`num_predict` 512), never the Task 14 variant.

## Stop line

STAGE A ADDENDUM COMPLETE -- T14-A1 implemented and validated, errata logged, observation corrected and its branch reported. Zero generations were issued. No evaluation persona has been sent to the model.
