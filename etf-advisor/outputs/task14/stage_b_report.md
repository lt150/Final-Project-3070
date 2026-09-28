# Task 14 Stage B -- prompt development, both arms, dev personas only

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

Determinism standard: **repeat-exact** (measured at Stage A). No options change is available at this stage and the options variant is byte-identical to Stage A's, so no re-probe is registered and none was run.

## 0. Boundary

> The Task 12 and Task 13 verdicts are frozen ground truth, and all Task 9-13 outputs are read-only. The elicitor creates no new profile, ladder, or budget freedom: its output is exactly one registered profile string from `PROFILES` (src.task9.allocators) or a clarify signal. `recommend()` validation is the backstop, never the primary check. A weak elicitor is a Task 14 finding about the elicitation layer and re-litigates nothing upstream.

## 1. Prompt design and the registered asymmetry

| requirement | Arm A | Arm B |
|---|---|---|
| five registered questions verbatim | yes | yes |
| persona's answers substituted | yes | yes |
| instructs the registered output format | five lines `Qk: <value>` | exactly one word |
| instructs `unclear` where the text does not support a value | yes | n/a (Arm B has no extraction) |
| profile semantics / mapping guidance | **none** -- the rubric decides | **permitted and used** -- horizon guidance, capacity-over-willingness, when to clarify |

The architectural difference under test is WHERE the decision lives, not how well either prompt is written . Arm B is therefore given the profile semantics its architecture needs, including that `growth` is the schema maximum and out-of-schema requests are not honoured -- a property Arm A gets structurally for free, because its rubric cannot emit anything outside `PROFILES`.

Mechanical template checks, run before any generation: each registered question must appear verbatim in both templates; all five placeholders must be present and fully consumed at render time; Arm A's template must contain no profile name (`balanced` exempt -- it is a registered Q3 domain value) and no `clarify`; Arm B's template must offer all four registered vocabulary words.

**Template issues: 0.**

## 2. Development iterations (D1-D3 only)

### Arm A -- 1 iteration(s), 3 logged calls

| iter | persona | prompt sha256 (16) | parse | extraction | extraction == expected | rubric trace | outcome | expected | match |
|---|---|---|---|---|---|---|---|---|---|
| 1 | D1 | `6b3cf1d87644d853` | VALID | `short/sell/preserve/low/none` | True | `0+0+0+0+0=0; score_profile=conservative; cap=conservative; min -> conservative` | **conservative** | conservative | PASS |
| 1 | D2 | `6b3cf1d87644d853` | VALID | `medium/hold/balanced/medium/some` | True | `1+1+1+1+1=5; score_profile=balanced; cap=balanced; min -> balanced` | **balanced** | balanced | PASS |
| 1 | D3 | `6b3cf1d87644d853` | VALID | `long/buy_more/grow/high/experienced` | True | `2+2+2+2+2=10; score_profile=growth; cap=growth; min -> growth` | **growth** | growth | PASS |

Arm A, final iteration: **3/3** dev personas reproduce their registered expected profile.

### Arm B -- 1 iteration(s), 3 logged calls

| iter | persona | prompt sha256 (16) | parse | extraction | extraction == expected | rubric trace | outcome | expected | match |
|---|---|---|---|---|---|---|---|---|---|
| 1 | D1 | `afbc556ba60635d1` | VALID | `-` | - | `-` | **conservative** | conservative | PASS |
| 1 | D2 | `afbc556ba60635d1` | VALID | `-` | - | `-` | **balanced** | balanced | PASS |
| 1 | D3 | `afbc556ba60635d1` | VALID | `-` | - | `-` | **growth** | growth | PASS |

Arm B, final iteration: **3/3** dev personas reproduce their registered expected profile.

Dev personas are excluded from every evaluation count (register Sec. 4). Their only role is prompt development; the numbers above certify nothing about either arm's evaluation performance.

## 3. Prompt freeze

| artifact | SHA-256 | chars |
|---|---|---|
| `src/task14/prompt_arm_a.txt` | `6b3cf1d87644d853e5cb6d6b944b1f4fb1bf423e101cd3325500eead09f0a28d` | 1609 |
| `src/task14/prompt_arm_b.txt` | `afbc556ba60635d1f5768e4b1ba7abaebc2cc6fa05e25cceb634ac15b6f4a085` | 1964 |

Both prompts are FROZEN at these hashes. Any later edit is an amendment (T14-Ax) and requires halting first.

## 4. Gate B1 -- dev-only discipline

| check | result |
|---|---|
| Stage B calls this run | 7 |
| calls matching the registered purpose grammar | 7/7 |
| distinct personas called | D1, D2, D3 |
| evaluation personas called | **0** |
| prompt files on disk scanned | 6 |
| evaluation-persona answer strings searched for | 70 |
| occurrences found in any sent prompt | **0** |

The purpose-string audit is backed by a content scan: every prompt actually sent to the model is on disk, and all 70 evaluation-persona answer strings (14 personas x 5 questions) are searched for in every one of them. Zero occurrences is the evidence that no evaluation persona has reached the model.

## 5. Gate B2 -- static prompt bound, rendering only

All 14 evaluation personas rendered through BOTH templates -- rendering only, no generation -- and measured against the registered bound of 12000 characters. Rendering an evaluation persona is not a call and does not touch the model.

| arm | personas rendered | min chars | max chars | longest persona | bound | verdict |
|---|---|---|---|---|---|---|
| Arm A | 14 | 1704 | 1823 | P-C1 | <= 12000 | PASS |
| Arm B | 14 | 2059 | 2178 | P-C1 | <= 12000 | PASS |

Worst rendered prompt over both arms: **2178** characters (bound 12000). For reference, invariant I4 caps `prompt_eval_count` at 8064 tokens; the dev calls logged above record the measured token counts.

## 6. Gates

| gate | criterion | evidence | verdict |
|---|---|---|---|
| B1 | dev-only discipline held -- every Stage B call was D1-D3 or the registered probe; zero evaluation-persona calls | 7 calls this run, 0 off-grammar; personas called ['D1', 'D2', 'D3']; content scan 0 hits over 6 sent prompts | PASS |
| B2 | both prompts frozen with hashes recorded; both dev logs complete; static prompt bound checked for all 14 evaluation personas x both arms (rendering only) | A `6b3cf1d87644d853`, B `afbc556ba60635d1`; dev logs A 3 rows / B 3 rows; 28 renders, worst 2178 <= 12000 | PASS |

## 7. Deliverables written by this stage

- `src/task14/prompt_arm_a.txt`, `prompt_arm_b.txt` (frozen), `elicit_arms.py`, `stage_b.py`
- `outputs/task14/stage_b_report.md` (this file)
- `outputs/task14/dev_iteration_log_arm_a.csv` (3 rows), `dev_iteration_log_arm_b.csv` (3 rows)
- `outputs/task14/stage_b_call_log.csv` (cumulative per-call log)
- `outputs/task14/transcripts/dev/` (prompt + response per dev call)

## 8. Stop line

STAGE B COMPLETE -- gates B1/B2 PASS. Both prompts frozen. No evaluation persona has been sent to the model.
