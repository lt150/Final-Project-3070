# Task 13 Stage B -- extractor validation, template development, freeze

Environment pin

- interpreter: `C:\Users\tianc\Desktop\Final Project\etf-advisor\.venv\Scripts\python.exe`
- python 3.13.0, numpy 2.4.6, pandas 3.0.3, scipy 1.17.1, pyarrow 25.0.0
- Ollama server version: `0.32.5`
- model tag: `qwen3.5:9b`, digest `6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7`
- parameter size `9.7B`, quantization `Q4_K_M`, family `qwen35`, architecture `qwen35`

Frozen options block, sent verbatim on every call:

```
model      = 'qwen3.5:9b'
endpoint   = 'http://localhost:11434/api/generate'
think      = False      # invariant I2
stream     = False
keep_alive = '60m'
options    = {'temperature': 0, 'top_k': 1, 'seed': 42, 'num_predict': 512, 'num_ctx': 8192, 'num_gpu': 0}
penalties  = {'presence_penalty': 0.0, 'frequency_penalty': 0.0, 'repeat_penalty': 1.0}
```

Standard in force entering this stage (from Stage A): **repeat-exact** -- the registered k = 3 byte-identical determinism standard. Gate B3 re-measures it under the frozen FINAL options block and restates it below.

Registered order, which is load-bearing: the extractor is validated on the registered synthetic fixtures BEFORE any model-generated text is parsed. Gate B1 therefore runs first and this driver refuses to proceed on failure.

## 1. Gate B1 -- extractor and adjudicator against the registered fixtures

### Extractor fixtures F1-F7 

Expected claim sets are exhaustive per fixture and compared order-insensitively on the canonical tuple `(type, subject, basis/object, value, stated_precision)`. Sentence classification and the T8 screen/adjudication outcome are checked alongside.

| fixture | expected claims | extracted | missing | extra | sentence classes match | T8 flagged/violating | T8 as registered | verdict |
|---|---|---|---|---|---|---|---|---|
| F1 | 5 | 5 | 0 | 0 | yes | 0/0 | yes | PASS |
| F2 | 2 | 2 | 0 | 0 | yes | 0/0 | yes | PASS |
| F3 | 2 | 2 | 0 | 0 | yes | 0/0 | yes | PASS |
| F4 | 2 | 2 | 0 | 0 | yes | 0/0 | yes | PASS |
| F5 | 1 | 1 | 0 | 0 | yes | 0/0 | yes | PASS |
| F6 | 0 | 0 | 0 | 0 | yes | 1/1 | yes | PASS |
| F7 | 0 | 0 | 0 | 0 | yes | 0/0 | yes | PASS |

### Matching-rule micro-fixtures M1-M5 

| micro-fixture | frozen value | claim | stated precision | registered outcome | adjudicator outcome | verdict |
|---|---|---|---|---|---|---|
| M1 | `0.337415` | `33.7` | `pct.1` | PASS | PASS | reproduced |
| M2 | `0.337415` | `34` | `pct.0` | PASS | PASS | reproduced |
| M3 | `0.337415` | `33.8` | `pct.1` | FAIL | FAIL | reproduced |
| M4 | `-0.01834` | `-1.8` | `pp.1` | PASS | PASS | reproduced |
| M5a | `0.0125` | `1.250` | `pct.3` | PASS | PASS | reproduced |
| M5b | `0.0125` | `1.3` | `pct.1` | FAIL | FAIL | reproduced |


**Gate B1: PASS** -- 7/7 extractor fixtures 

## 2. Gate B2 -- static prompt-size bound

The rendered prompt for each of the 12 cells must be <= 12,000 characters (a conservative upper bound keeping token count inside invariant I4's budget of 7,680 prompt tokens; the real `prompt_eval_count` is additionally verified per generation at Stage C). Rendering is not generation -- **no generation is run on any non-anchor cell at this stage.**

| date | profile | prompt chars | verdict | note |
|---|---|---|---|---|
| 2012-02-01 | conservative | 2923 | PASS |  |
| 2012-02-01 | balanced | 2920 | PASS |  |
| 2012-02-01 | growth | 2918 | PASS |  |
| 2020-03-02 | conservative | 2923 | PASS |  |
| 2020-03-02 | balanced | 2919 | PASS |  |
| 2020-03-02 | growth | 2916 | PASS |  |
| 2022-06-01 | conservative | 2923 | PASS |  |
| 2022-06-01 | balanced | 2920 | PASS |  |
| 2022-06-01 | growth | 2919 | PASS |  |
| 2024-12-02 | conservative | 2922 | PASS |  |
| 2024-12-02 | balanced | 2920 | PASS | anchor (development cell) |
| 2024-12-02 | growth | 2918 | PASS |  |

Maximum over the 12 cells: 2,923 characters. **Gate B2: PASS.**

## 3. Template and serializer development (anchor cell only)

Development cell: **2024-12-02 / balanced** -- verified present in the frozen table at Stage A. Every development generation was made against this cell and no other; the 11 evaluation cells were rendered (Gate B2) but never generated.

### Penalty knobs -- in-force values at dev start

| knob | value in force at dev start | source |
|---|---|---|
| presence_penalty | 1.5 | model Modelfile |
| frequency_penalty | UNSPECIFIED in Modelfile; server built-in default | not exposed by any Ollama API endpoint |
| repeat_penalty | UNSPECIFIED in Modelfile; server built-in default | not exposed by any Ollama API endpoint |

The two server built-in defaults are recorded as UNSPECIFIED rather than asserted: Ollama exposes the Modelfile parameter block via `/api/show`, but not the server's own fallback values for parameters the Modelfile omits.

### Dev iteration log (Penalty knobs: every value tried)

| iter | template sha256 (16) | presence | frequency | repeat | chars | sentences | claims | pass | fail | unaccounted | T8 flag/viol | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `be13bc2e47cee7b4` | shipped-default | server-default | server-default | 665 | 7 | 15 | 15 | 0 | 1 | 0/0 | iter1: dev-start baseline, shipped/server penalty defaults in force (no penalty keys sent) |
| 2 | `76e883588a5e17d0` | shipped-default | server-default | server-default | 660 | 7 | 17 | 17 | 0 | 0 | 0/0 | iter2: FACTS attribution labels renamed to target/source ticker; slot-name rule added; shipped/server penalty defaults |
| 3 | `76e883588a5e17d0` | 0.0 | 0.0 | 1.0 | 660 | 7 | 17 | 17 | 0 | 0 | 0/0 | iter3: T13-A1 knob sweep, all three penalties neutralised |

Dev iteration count: **3**.

What each iteration changed, and why:

- **iter1** exposed a serializer defect, not a model defect: the FACTS block labelled the attribution rows `output ...` and `input ...`, and the sentence plan used the slots `[output]` / `[input]`, so the model wrote the literal words *output* and *input* where tickers belonged. The unaccounted-numeric rule caught it -- the attribution numeral could not be bound to an input/output pair and was reported as an UNACCOUNTED NUMERIC. This is the designed behaviour: under-extraction is punished, not rewarded.
- **iter2** renamed those FACTS labels to `attribution target ticker` / `attribution source ticker` and added an explicit style rule that a bracketed slot names a FACTS entry and must be replaced by that entry's value. 17/17 claims passed, 0 unaccounted.
- **iter3** neutralised all three penalty knobs with the template unchanged. The narration was **byte-identical to iter2**, so the knob setting is not outcome-determining at the anchor cell.

## 4. Freeze 

| artifact | SHA-256 |
|---|---|
| options block (canonical JSON, below) | `aaafb46f9870a225f2aec4691e228dc209eea446917a247851ed0fc3cfe8f4b5` |

The registered artifact is the options BLOCK, not merely the file holding it, so the hash covers the frozen options, the penalty knobs and the per-call top-level fields together:

```
{"endpoint":"http://localhost:11434/api/generate","keep_alive":"60m","model":"qwen3.5:9b","options":{"frequency_penalty":0.0,"num_ctx":8192,"num_gpu":0,"num_predict":512,"presence_penalty":0.0,"repeat_penalty":1.0,"seed":42,"temperature":0,"top_k":1},"stream":false,"think":false}
```

### Final frozen penalty values

| knob | frozen value |
|---|---|---|
| presence_penalty | 0.0 |
| frequency_penalty | 0.0 |
| repeat_penalty | 1.0 |

Choice recorded with its reason: iter2 (shipped/server defaults) and iter3 (all three neutralised) produced byte-identical anchor narrations, so this selection is **not fitted to an outcome**. Neutralising is chosen because a presence penalty suppresses re-emission of tokens already present in the recent context and is directionally adversarial to a task whose success criterion is exact echoing of grounding digits -- a pressure that would act on the 11 unseen evaluation cells, several of which require the same ticker to recur. The countervailing risk, degenerate repetition at temperature 0, was looked for and not observed. These values are stated in every subsequent report header.

## 5. Gate B3 addendum -- determinism re-probe under the FROZEN final options

Required after ratification: the k = 3 byte-identical probe is re-run under the exact frozen final options block (penalty knobs included), and the standard-in-force line is restated from that result. Registered probe prompt, one discarded warm-up, then k = 3.

| repeat | chars | sha256 (16) | == rep1 | done_reason | prompt_eval_count |
|---|---|---|---|---|---|
| 1 | 495 | 0da353ea288fc516 | yes | `stop` | 24 |
| 2 | 495 | 0da353ea288fc516 | yes | `stop` | 24 |
| 3 | 495 | 0da353ea288fc516 | yes | `stop` | 24 |

**Byte-identical across k = 3 under the frozen final options: YES.**

Cross-check against the Stage A probe (which ran under the shipped penalty defaults): the Stage B probe text is **different from** the Stage A probe text. The penalty knobs therefore do move output on the probe prompt; the frozen block is the one measured here, and it is the block Stage C runs under.

**Determinism standard IN FORCE for Stage C: repeat-exact** -- k = 3 byte-identical, re-measured under the frozen final options block. The fallback is NOT activated.

Reproducibility standard: **repr-exact where achievable; transcript-freeze with disclosure where not.** Narration is the first component in the stack without repr-exact guarantees.

## 6. End-to-end dry run on the anchor cell

**Explicitly outside the headline.** The anchor is the development cell; its results are reported here and again alongside the Stage C tables, always flagged, and never inside the 11-cell headline counts.

Narration (quoted ASCII-escaped; raw bytes on disk at `outputs/task13/narrations/dev_anchor_dry_run.txt`):

```
For the balanced profile on 2024-12-02, the advisor allocates 14.1% to SPY, 11.7% to QQQ, 6.0% to EFA, 5.5% to EEM, 28.3% to AGG, 23.5% to IEF, 5.8% to GLD and 4.9% to VNQ. AGG is the largest position in the portfolio. The model forecasts a daily volatility of 0.32% for AGG, the lowest in the universe. Compared with the classical benchmark, the advisor overweights AGG and reduces its exposure to SPY. The attributions decompose the advisor-versus-classical difference. The largest single contribution to the AGG weight difference comes from the AGG forecast, at 3.7 percentage points. This explanation is generated automatically and is not financial advice.
```

Non-ASCII characters in the quoted narration: 0. `done_reason` `stop`; `prompt_eval_count` 770 (I4 budget 7,680, within); `eval_count` 200 of a 512 cap.

| claim type | extracted | passed |
|---|---|---|
| T1 | 8 | 8 |
| T2 | 1 | 1 |
| T3 | 1 | 1 |
| T4 | 2 | 2 |
| T5 | 3 | 3 |
| T6 | 2 | 2 |

| metric | value |
|---|---|
| sentences classified | 7 |
| claim-bearing sentences | 5 |
| no-claim sentences | 2 |
| claims extracted | 17 |
| claims passed | 17 |
| claims failed | 0 |
| unaccounted numerics | 0 |
| T8 flagged sentences | 0 |
| T8 violations | 0 |
| soft qualitative observations | 0 |

Sentence classification table:

| # | class | subclass | T8 flagged | T8 violation | sentence |
|---|---|---|---|---|---|
| 0 | claim-bearing | - | False | False | For the balanced profile on 2024-12-02, the advisor allocates 14.1% to SPY, 11.7% to QQQ, 6.0% to EFA, 5.5% to EEM, 28.3% to AGG, 23.5% to IEF, 5.8% to GLD and 4.9% to VNQ. |
| 1 | claim-bearing | - | False | False | AGG is the largest position in the portfolio. |
| 2 | claim-bearing | - | False | False | The model forecasts a daily volatility of 0.32% for AGG, the lowest in the universe. |
| 3 | claim-bearing | - | False | False | Compared with the classical benchmark, the advisor overweights AGG and reduces its exposure to SPY. |
| 4 | no-claim | registered-verbatim | False | False | The attributions decompose the advisor-versus-classical difference. |
| 5 | claim-bearing | - | False | False | The largest single contribution to the AGG weight difference comes from the AGG forecast, at 3.7 percentage points. |
| 6 | no-claim | boilerplate | False | False | This explanation is generated automatically and is not financial advice. |

## 7. What this design can and cannot support (recorded before Stage C)

The frozen template is highly constrained: it supplies a FACTS block holding only the numbers the narration needs, and a seven-sentence plan whose wording matches the registered fixture forms. That is a deliberate consequence of the registered design -- the extractor is rule-based and English-specific, and the fixtures F1-F7 were hand-authored at planning level as the target sentence shapes -- but it bounds the claim Stage C can make. A supported verdict will say that THIS narration layer ties out to the frozen artifacts; it will not say that free-form LLM narration of this Recommendation is faithful, and Stage C must not be read that way.

Two properties keep the test from being vacuous. First, the model must still select the correct value from the FACTS block for each slot, and iter1 shows the selection can go wrong. Second, the unaccounted-numeric rule makes silence expensive: any numeral the extractor cannot bind to a claim is an UNACCOUNTED NUMERIC on the T7 hard-fail path, so an extractor that quietly declines to parse an awkward sentence fails the run rather than flattering it.

The covariance-source screen is stricter than the narration needs: any sentence mentioning covariance, Sigma or a correlation matrix raises a T6 claim that passes only if it contains the frozen `COVARIANCE_SOURCE` literal (`Sigma = D_advisor(walk-forward OLS) * R_250d * D_advisor`) verbatim. The template instructs the model not to mention the risk model at all, so this screen can only ever cost the run, never help it.

## 8. Gates

| gate | criterion | evidence | verdict |
|---|---|---|---|
| B1 | extractor exact on F1-F7; adjudicator exact on M1-M5 | 7/7 fixtures, 6/6 micro-fixtures reproduced; sentence classes and T8 outcomes as registered | PASS |
| B2 | rendered prompt <= 12,000 chars for all 12 cells; no non-anchor generation | max 2,923 chars; generations this stage = 5 (probe x4 + anchor dry run x1), all on the anchor cell or the registered probe prompt | PASS |
| B3 | freeze complete (hashes recorded); anchor dry run executed and recorded; dev iteration log present; determinism re-probe under the frozen final options | 3 hashes recorded; dry run done_reason `stop`; dev log 3 iterations; re-probe byte-identical = True | PASS |

## 9. Deliverables written by this stage

- `outputs/task13/stage_b_report.md` (this file)
- `outputs/task13/dev_iteration_log.csv` (penalty knob iteration log)
- `outputs/task13/stage_b_call_log.csv` (per-call log)
- `outputs/task13/narrations/dev_iter{1,2,3}_anchor.txt`, `dev_anchor_dry_run.txt`, `b3_probe_rep{1,2,3}.txt`
- frozen: options block (hashes in Sec. 4)

STAGE B COMPLETE -- gates B1/B2/B3 PASS. 