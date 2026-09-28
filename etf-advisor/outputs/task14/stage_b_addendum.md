# Task 14 Stage B ADDENDUM -- amendment T14-A2, dev persona D4 (abstention probe)

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

Determinism standard IN FORCE: **repeat-exact** (measured at Stage A; options variant unchanged, so no re-probe is registered).

## 1. Why D4 exists

D1-D3 are clear-cut by construction, so Stage B never observed either arm emitting `unclear` (Arm A) or `clarify` (Arm B) -- the machinery for it was fixture-proven at Stage A (PF4), but neither arm's propensity to abstain had been exercised against the model, even though five evaluation personas turn on exactly that and two accept `clarify` only. Amendment T14-A2 adds D4 to the development sandbox to exercise the path before Stage C, under the frozen prompts.

### D4, as stored

| question | answer |
|---|---|
| Q1 | Soon -- although really it's for when I retire in thirty years. |
| Q2 | Sell it all, or maybe double down, depends on my mood. |
| Q3 | I want it completely safe and also to double every year. |
| Q4 | Depends on the month. |
| Q5 | Sort of. |

Registered expectations: outcome = clarify for BOTH arms. Arm A: Q1, Q2, Q3 must extract unclear (routing clarify(Q1)); Q4 and Q5 are unconstrained at dev (unclear or a domain value both acceptable).

## 2. D4 through both arms, k = 1, frozen prompts

| arm | iteration | prompt sha256 (16) | raw response | parse | extraction | outcome |
|---|---|---|---|---|---|---|
| A | 2 | `6b3cf1d87644d853` | `Q1: long\nQ2: unclear\nQ3: unclear\nQ4: unclear\nQ5: unclear` | VALID | `long/unclear/unclear/unclear/unclear` | **clarify** |
| B | 2 | `afbc556ba60635d1` | `clarify` | VALID | `-` | **clarify** |

### Registered expectation checks

| arm | registered expectation | expected | observed | verdict |
|---|---|---|---|---|
| Arm A | parse VALID | VALID | VALID | PASS |
| Arm A | Q1, Q2, Q3 extract `unclear` | unclear x3 | Q1=long, Q2=unclear, Q3=unclear | **FAIL** |
| Arm A | routing clarify(Q1) | clarify(Q1) | clarify(Q2) | **FAIL** |
| Arm A | Q4, Q5 unconstrained at dev | any | Q4=unclear, Q5=unclear | n/a |
| Arm A | outcome | clarify | clarify | PASS |
| Arm B | parse VALID | VALID | VALID | PASS |
| Arm B | outcome | clarify | clarify | PASS |

**At least one arm did not meet the registered expectations.** Per T14-A2 a bounded prompt-iteration window against D1-D4 ONLY now opens. This driver does not edit prompts; it stops here so the window is opened at planning level with the evidence above in hand.

## 3. Gate B1 discipline, extended to D1-D4 (cumulative audit)

| check | result |
|---|---|
| Stage B calls logged in total (all runs) | 10 |
| calls matching the registered purpose grammar | 10/10 |
| distinct personas called | D1, D2, D3, D4 |
| evaluation personas called | **0** |
| prompt files on disk scanned | 8 |
| evaluation-persona answer strings searched for | 70 |
| occurrences found in any sent prompt | **0** |
| worst rendered evaluation prompt (rendering only) | 2178 <= 12000 |

The audit is cumulative over `outputs/task14/stage_b_call_log.csv`, so it covers every Stage B call ever made, not just this run. The content scan re-runs over every prompt ever sent, as T14-A2 requires before Stage C.

## 4. Deliverables written by this stage

- `outputs/task14/stage_b_addendum.md` (this file)
- `outputs/task14/dev_iteration_log_arm_{a,b}.csv` (D4 rows appended)
- `outputs/task14/stage_b_call_log.csv` (cumulative; D4 calls appended)
- `outputs/task14/transcripts/dev/arm_{a,b}_D4_iter*.txt`

## Stop line

STAGE B ADDENDUM INCOMPLETE -- see above.
