# Task 14 -- T14-A2 bounded prompt-iteration record

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

## 0. Why the window opened

Under the Gate B2 frozen prompts, D4 was run through both arms (`outputs/task14/stage_b_addendum.md` record). Arm B met the registered expectations. Arm A did not: from an answer naming two irreconcilable horizons it extracted `long` at Q1 rather than `unclear`, so it routed clarify(Q2) instead of the registered clarify(Q1). Its outcome was still `clarify`, so the failure was in WHICH question it flagged, not in whether it abstained. Amendment T14-A2 opens a bounded prompt-iteration window against D1-D4 only.

## 1. Bound and scope, declared

| item | value |
|---|---|
| content available to the window | dev personas D1-D4 only (amendment T14-A2) |
| declared attempt bound | 2 revision attempts |
| arms revised | Arm A only -- Arm B met its expectations and its prompt is untouched |
| evaluation personas involved | none, at any point |
| dev set re-run each attempt | D1-D4 through BOTH arms |

Both arms are re-run every attempt for two reasons: a fix aimed at D4 must not silently cost D1-D3, and the unrevised arm is re-measured rather than assumed, which doubles as a determinism control on an unchanged prompt.

## 2. The Arm A revision

The revision is a general extraction rule and names nothing specific to D4. The prompt previously listed `self-contradictory` inside a catch-all clause about answers that support NO label; the model read an answer supporting TWO labels as a case for picking the better one. The revision separates those cases and states the multi-label rule explicitly:

> If the answer supports more than one of that question's labels -- for example it gives two different answers, or changes its mind -- then it supports none of them: output: unclear. Do not pick whichever seems more likely, and do not reconcile the two.

This is the definition of `unclear` (*a first-class output whenever the text does not support a registered value*) made operative for the two-label case. It carries no profile semantics and no mapping guidance, so Arm A's registered constraint still holds. Note that `balanced` on Q3 is a single label meaning both, so an answer like D2's "A bit of both" supports exactly one label and is not caught by this rule -- D2's result below is the check on that.

## 3. Attempt results -- D1-D4 through both arms

| iter | arm | persona | prompt sha256 (16) | parse | extraction | outcome | detail | verdict |
|---|---|---|---|---|---|---|---|---|
| 4 | A | D1 | `a5760514e9d33e63` | VALID | `short/sell/preserve/low/none` | **conservative** | conservative / extraction exact | PASS |
| 4 | A | D2 | `a5760514e9d33e63` | VALID | `medium/hold/balanced/medium/some` | **balanced** | balanced / extraction exact | PASS |
| 4 | A | D3 | `a5760514e9d33e63` | VALID | `long/buy_more/grow/high/experienced` | **growth** | growth / extraction exact | PASS |
| 4 | A | D4 | `a5760514e9d33e63` | VALID | `unclear/unclear/unclear/unclear/unclear` | **clarify** | Q1-Q3 unclear=True; clarify(Q1); clarify | PASS |
| 4 | B | D1 | `afbc556ba60635d1` | VALID | `-` | **conservative** | conservative | PASS |
| 4 | B | D2 | `afbc556ba60635d1` | VALID | `-` | **balanced** | balanced | PASS |
| 4 | B | D3 | `afbc556ba60635d1` | VALID | `-` | **growth** | growth | PASS |
| 4 | B | D4 | `afbc556ba60635d1` | VALID | `-` | **clarify** | clarify | PASS |

Dev set met on both arms: **True** (8/8 persona-arm cells).

## 4. Prompt freeze after the window

| prompt | current hash | hash present in the ratified `stage_b_report.md` | revised in this window |
|---|---|---|---|
| `src/task14/prompt_arm_a.txt` | `a5760514e9d33e63e06984a3a8de5e6a956dc3d2176082e51ed6ffb3c792523b` | no -- this is a new hash | yes -- multi-label rule added |
| `src/task14/prompt_arm_b.txt` | `afbc556ba60635d1f5768e4b1ba7abaebc2cc6fa05e25cceb634ac15b6f4a085` | yes -- unchanged since Gate B2 | no -- byte-unchanged |

Whether each prompt moved is established by looking its CURRENT hash up in the ratified Gate B2 report rather than by retyping the old value: Arm B's hash still resolves there, Arm A's does not. Arm A is re-frozen at the hash above and is the prompt Stage C will use. Any later edit to either prompt is a fresh amendment.

## 5. Gate B1 / B2 discipline, cumulative

| check | result |
|---|---|
| Stage B calls logged in total (all runs) | 28 |
| calls matching the registered purpose grammar | 28/28 |
| distinct personas called | D1, D2, D3, D4 |
| evaluation personas called | **0** |
| prompt files on disk scanned | 24 |
| evaluation-persona answer strings searched for | 70 |
| occurrences found in any sent prompt | **0** |
| worst rendered evaluation prompt (rendering only) | 2410 <= 12000 |

## 6. Deliverables written by this stage

- `outputs/task14/stage_b_t14a2_iteration.md` (this file)
- `outputs/task14/dev_iteration_log_arm_{a,b}.csv` (attempt rows appended)
- `outputs/task14/stage_b_call_log.csv` (cumulative)
- `outputs/task14/transcripts/dev/` (prompt + response per call)
- `src/task14/prompt_arm_a.txt` re-frozen; `prompt_arm_b.txt` untouched

## 7. Stop line

T14-A2 CONCLUDED -- the bounded window closed within its declared bound. Arm A re-frozen at a new hash, Arm B byte-unchanged, D1-D4 met on both arms. No evaluation persona has been sent to the model. 
