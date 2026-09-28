# Task 14 Stage C -- one-shot two-arm evaluation

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

Determinism standard IN FORCE: **repeat-exact** -- every (persona, arm) cell was byte-identical across k = 3, so the registered standard held and the fallback was not activated.

Frozen prompts in force: Arm A `a5760514e9d33e63e06984a3a8de5e6a956dc3d2176082e51ed6ffb3c792523b`, Arm B `afbc556ba60635d1f5768e4b1ba7abaebc2cc6fa05e25cceb634ac15b6f4a085`. Persona file `e18db29bdfd0be91566bad1270355ba61ccce9098eda7b6f481b63425022224d`.

## 0. Boundary

> The Task 12 and Task 13 verdicts are frozen ground truth, and all Task 9-13 outputs are read-only. The elicitor creates no new profile, ladder, or budget freedom: its output is exactly one registered profile string from `PROFILES` (src.task9.allocators) or a clarify signal. `recommend()` validation is the backstop, never the primary check. A weak elicitor is a Task 14 finding about the elicitation layer and re-litigates nothing upstream.

## 1. Run integrity

| item | value |
|---|---|
| environment drift check | `/api/version` and digest re-read before the first generation: `0.32.5` / `6488c96fa5faab64...` vs the pin -- no drift |
| discarded warm-up | 1 (excluded from all counts) |
| evaluation generations | 84 / 84 |
| registered order honoured | yes -- Arm A over P-C1..P-G3 then P-X1..P-X5, k = 3 consecutive per persona, then Arm B in the same persona order |
| transcripts written immediately | yes, one file per generation, before the next call |
| I1 model == pinned tag | held on 85/85 |
| I2 no thinking content | held on 85/85 |
| I3 done == true | held on 85/85 |
| I3 done_reason values | `stop` x85 |
| I4 prompt_eval_count <= 8064 | held on 85/85; observed range 24-601 |
| eval_count range (cap 128) | 2-81 |
| per-call log | `outputs/task14/stage_c_call_log.csv`, 85 rows (warm-up included) |

## 2. Determinism finding, per (persona, arm)

| arm | persona | agreement | response chars | rep1 sha256 (16) | byte-identical | first divergence |
|---|---|---|---|---|---|---|
| A | P-C1 | === | 49; 49; 49 | 6d158381e23f11fd | YES | - |
| A | P-C2 | === | 56; 56; 56 | 0f669bb4c88729c4 | YES | - |
| A | P-C3 | === | 49; 49; 49 | 6d158381e23f11fd | YES | - |
| A | P-B1 | === | 50; 50; 50 | e3395b02ee2481aa | YES | - |
| A | P-B2 | === | 48; 48; 48 | 7d03e8134ec6fc12 | YES | - |
| A | P-B3 | === | 54; 54; 54 | 3d2e6b2b112d3838 | YES | - |
| A | P-G1 | === | 55; 55; 55 | c8a81210b7129c64 | YES | - |
| A | P-G2 | === | 48; 48; 48 | 202eb62602f5bd43 | YES | - |
| A | P-G3 | === | 51; 51; 51 | 4f749c57f9e79902 | YES | - |
| A | P-X1 | === | 55; 55; 55 | 0dd6dffda215ff57 | YES | - |
| A | P-X2 | === | 61; 61; 61 | 44cbb973c680d2d0 | YES | - |
| A | P-X3 | === | 59; 59; 59 | 7c839093316790dc | YES | - |
| A | P-X4 | === | 59; 59; 59 | 7c839093316790dc | YES | - |
| A | P-X5 | === | 55; 55; 55 | c8a81210b7129c64 | YES | - |
| B | P-C1 | === | 12; 12; 12 | c152d5bbb379c3f9 | YES | - |
| B | P-C2 | === | 7; 7; 7 | c010d5892545f57e | YES | - |
| B | P-C3 | === | 12; 12; 12 | c152d5bbb379c3f9 | YES | - |
| B | P-B1 | === | 8; 8; 8 | c0905088ba5b8067 | YES | - |
| B | P-B2 | === | 8; 8; 8 | c0905088ba5b8067 | YES | - |
| B | P-B3 | === | 8; 8; 8 | c0905088ba5b8067 | YES | - |
| B | P-G1 | === | 6; 6; 6 | 19c64195eb8f22c3 | YES | - |
| B | P-G2 | === | 6; 6; 6 | 19c64195eb8f22c3 | YES | - |
| B | P-G3 | === | 6; 6; 6 | 19c64195eb8f22c3 | YES | - |
| B | P-X1 | === | 8; 8; 8 | c0905088ba5b8067 | YES | - |
| B | P-X2 | === | 6; 6; 6 | 19c64195eb8f22c3 | YES | - |
| B | P-X3 | === | 7; 7; 7 | c010d5892545f57e | YES | - |
| B | P-X4 | === | 7; 7; 7 | c010d5892545f57e | YES | - |
| B | P-X5 | === | 7; 7; 7 | c010d5892545f57e | YES | - |

Cells byte-identical across k = 3: **28 / 28**. Divergent cells: **0**. The evaluated transcript is the first repeat per cell either way, and both arms are evaluated under whichever standard is in force.

## 3. Per-persona outcomes, both arms (one parse pass, one rubric pass, one category assignment)

| persona | kind | expected / acceptable | Arm A extraction | Arm A outcome | Arm A category | Arm B outcome | Arm B category |
|---|---|---|---|---|---|---|---|
| P-C1 | clear_cut | conservative | `medium/sell/preserve/low/none` | **conservative** | correct profile | **conservative** | correct profile |
| P-C2 | clear_cut | conservative | `short/buy_more/grow/high/experienced` | **conservative** | correct profile | **clarify** | unnecessary clarification |
| P-C3 | clear_cut | conservative | `medium/sell/preserve/low/none` | **conservative** | correct profile | **conservative** | correct profile |
| P-B1 | clear_cut | balanced | `medium/hold/balanced/high/some` | **balanced** | correct profile | **balanced** | correct profile |
| P-B2 | clear_cut | balanced | `long/hold/balanced/high/some` | **balanced** | correct profile | **balanced** | correct profile |
| P-B3 | clear_cut | balanced | `medium/buy_more/preserve/high/some` | **balanced** | correct profile | **balanced** | correct profile |
| P-G1 | clear_cut | growth | `long/buy_more/grow/high/experienced` | **growth** | correct profile | **growth** | correct profile |
| P-G2 | clear_cut | growth | `long/buy_more/grow/high/some` | **growth** | correct profile | **growth** | correct profile |
| P-G3 | clear_cut | growth | `long/hold/grow/high/experienced` | **growth** | correct profile | **growth** | correct profile |
| P-X1 | ambiguous | {conservative, balanced, clarify} | `medium/hold/balanced/medium/unclear` | **clarify** | conformant | **balanced** | conformant |
| P-X2 | ambiguous | {balanced, growth, clarify} | `medium/buy_more/balanced/high/experienced` | **balanced** | conformant | **growth** | conformant |
| P-X3 | ambiguous | {clarify} | `unclear/unclear/unclear/unclear/unclear` | **clarify** | conformant | **clarify** | conformant |
| P-X4 | ambiguous | {clarify} | `unclear/unclear/unclear/unclear/unclear` | **clarify** | conformant | **clarify** | conformant |
| P-X5 | ambiguous | {growth} | `long/buy_more/grow/high/experienced` | **growth** | conformant | **clarify** | nonconformant |

## 4. Arm A extraction accuracy (clear-cut personas, per question)

| persona | Q1 | Q2 | Q3 | Q4 | Q5 | matches |
|---|---|---|---|---|---|---|
| P-C1 | **medium** (exp short) | sell | preserve | low | none | 4/5 |
| P-C2 | short | buy_more | grow | high | experienced | 5/5 |
| P-C3 | medium | sell | preserve | low | none | 5/5 |
| P-B1 | medium | hold | balanced | **high** (exp medium) | some | 4/5 |
| P-B2 | long | hold | balanced | **high** (exp medium) | some | 4/5 |
| P-B3 | medium | buy_more | preserve | high | some | 5/5 |
| P-G1 | long | buy_more | grow | high | experienced | 5/5 |
| P-G2 | long | buy_more | grow | high | some | 5/5 |
| P-G3 | long | hold | grow | high | experienced | 5/5 |

| question | correct / 9 |
|---|---|
| Q1 | 8/9 |
| Q2 | 9/9 |
| Q3 | 9/9 |
| Q4 | 7/9 |
| Q5 | 9/9 |

Overall extraction accuracy over the 9 clear-cut personas: **42/45** question labels (93.3%). Extraction accuracy is reported separately from mapping accuracy and feeds no gate.

## 5. Arm A error attribution, and the halt-and-escalate check

No Arm A clear-cut persona scored below 1.0, so the attribution table is empty; `E2_error_attribution.md` carries the header regardless.

HALT-AND-ESCALATE check: an incorrect Arm A profile whose extraction is identical to the registered expectation is impossible unless the register is inconsistent. Occurrences: **0**.

Arm B has no intermediate by construction, so its errors are reported as opaque; the asymmetry is a finding of record.

## 6. Selection under the registered rule

| criterion | Arm A | Arm B |
|---|---|---|
| **primary**: graded clear-cut score (max 9.0) | **9.0** | **8.5** |
| -- correct profile (1.0 each) | 9 | 8 |
| -- unnecessary clarification (0.5 each) | 0 | 1 |
| -- incorrect profile (0.0) | 0 | 0 |
| -- malformed (0.0) | 0 | 0 |
| **secondary**: ambiguous-set conformance (max 5) | **5** | **4** |
| -- nonconformant | 0 | 1 |
| -- malformed | 0 | 0 |

Decision basis: **strict win on the primary criterion**.

### Registered outcome statement

> Arm A is selected under the registered rule: graded clear-cut score 9.0/9.0 vs 8.5/9.0 (correct 9 vs 8; unnecessary clarification 0 vs 1; incorrect 0 vs 0; malformed 0 vs 0); ambiguous-set conformance 5/5 vs 4/5; strict win on the primary criterion. The shipped elicitor emits exactly one registered profile string or a clarify signal; recommend() validation is the backstop.

### Registered disclosure

> The selected arm is certified on the registered personas; live free-text input at runtime is out-of-distribution relative to this evaluation.

## 7. The shipped elicitor

| check | observed | verdict |
|---|---|---|
| dataclass fields | `['arm', 'profile', 'clarify_question', 'malformed', 'extraction']` | PASS |
| selected arm readable | `A` | PASS |
| replay of A_P-C1_rep1 | `arm=A profile=conservative clarify_question=None malformed=False` | PASS |
| profile validated against PROFILES | `'conservative' in ('conservative', 'balanced', 'growth')` | PASS |

`src/task14/elicit.py` exposes `ElicitationResult` and `elicit()`, runs ONLY Arm A, and validates `profile` against `PROFILES` before returning. The losing arm's code remains in `elicit_arms.py` for the record and is not exposed. The selection record is `outputs/task14/selection.json`. Runtime handling of `malformed` (re-ask or fall back) is a Task 15 decision; `elicit()` only reports it. The interface check above uses a stored transcript replay, so it spends no generation and the 84-call count is unaffected.

## 8. Exhibits

- **E1** `outputs/task14/E1_outcome_categories.md` / `.csv` -- arm x category counts for both persona classes, plus the per-persona category grid.
- **E2** `outputs/task14/E2_error_attribution.md` / `.csv` -- Arm A error attribution; the header is written even when empty.

Both are regenerated from this run's frozen CSVs only.

## 9. Gates

| gate | criterion | evidence | verdict |
|---|---|---|---|
| C1 | 84/84 calls in registered order; invariants held or deviations logged; transcripts and call log complete | 84/84 evaluation calls + 1 discarded warm-up; order honoured = True; I1/I2/I3 held on 85/85; I4 deviations 0 | PASS |
| C2 | one parse / one rubric / one category pass; categories per the register; halt-and-escalate checked | 28 cells interpreted once each; 28 categories assigned once each; halt-and-escalate occurrences 0 | PASS |
| C3 | outcome statement in the registered form; elicit.py ships the selected arm with the Sec. 1 interface; symmetric per-arm report; E1/E2 written | statement emitted with all counts filled; elicit.py ships Arm A; interface checks 4/4; E1 and E2 written | PASS |

## 10. Deliverables written by this stage

- `outputs/task14/stage_c_report.md` (this file)
- `outputs/task14/transcripts/{A,B}_<persona>_rep{1,2,3}.txt` (84) and `stage_c_warmup_discarded.txt`
- `outputs/task14/outcomes.csv`, `extraction_accuracy.csv`
- `outputs/task14/stage_c_call_log.csv`, `determinism_log.csv` (Stage A rows preserved; Stage C rows appended)
- `outputs/task14/E1_outcome_categories.md` / `.csv`, `E2_error_attribution.md` / `.csv`
- `outputs/task14/selection.json`; `src/task14/elicit.py` ships Arm A

## 11. Stop line

STAGE C COMPLETE -- gates C1/C2/C3 PASS. One-shot discipline held: 84 evaluation generations, one parse pass, one rubric pass, one category assignment, one selection.
