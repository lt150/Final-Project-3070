# Task 13 Stage C -- one-shot narration evaluation

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

## 0. Boundary

> The Task 12 verdict is frozen ground truth. If narration fails faithfulness, that is a Task 13 finding about the narration layer and re-litigates nothing upstream. attributions_main.csv, the Recommendation interface, and all Task 9-12 outputs are read-only.

## 1. Run integrity

| item | value |
|---|---|
| environment drift check | `/api/version` re-read before the first generation: `0.32.5` vs the Stage A pin `0.32.5` -- no drift; model digest matched as well |
| discarded warm-up | 1 (excluded from all counts) |
| evaluation generations | 36 / 36 |
| registered order honoured | yes -- dates ascending, conservative/balanced/growth within date, k = 3 consecutive per cell |
| transcripts written immediately | yes, one file per generation |
| I1 model == pinned tag | held on 36/36 |
| I2 no thinking content | held on 36/36 |
| I3 done == true | held on 36/36 |
| I3 done_reason values | `stop` x36 |
| I4 prompt_eval_count <= 7680 | held on 36/36; observed range 768-770 |
| eval_count range (cap 512) | 198-200 |
| per-call log | `outputs/task13/stage_c_call_log.csv`, 37 rows (warm-up included, flagged) |

Bookkeeping reconciliation carried in at ratification: Stage B made 8 generations in total: 5 from the stage driver (1 discarded warm-up + 3 determinism re-probes + 1 anchor dry run) and 3 from the development harness (dev iterations 1-3). All 8 were on the anchor cell or the registered probe prompt; no non-anchor cell was ever generated before this stage.

## 2. Determinism finding

| date | profile | flag | agreement (rep1/2/3) | response chars | rep1 sha256 (16) | byte-identical | first divergence |
|---|---|---|---|---|---|---|---|
| 2012-02-01 | conservative |  | === | 663; 663; 663 | 1175158ad6222851 | YES | - |
| 2012-02-01 | balanced |  | === | 660; 660; 660 | 6519acedf8cc3ab9 | YES | - |
| 2012-02-01 | growth |  | === | 658; 658; 658 | 2f612a038425ca08 | YES | - |
| 2020-03-02 | conservative |  | === | 663; 663; 663 | 9e95240b1b187a51 | YES | - |
| 2020-03-02 | balanced |  | === | 659; 659; 659 | 718897b4e7e60a5b | YES | - |
| 2020-03-02 | growth |  | === | 656; 656; 656 | a02283a0404ea98d | YES | - |
| 2022-06-01 | conservative |  | === | 663; 663; 663 | 99f859cf0e6bc4aa | YES | - |
| 2022-06-01 | balanced |  | === | 660; 660; 660 | 546ad58b0b9be787 | YES | - |
| 2022-06-01 | growth |  | === | 659; 659; 659 | db55cab141c0f778 | YES | - |
| 2024-12-02 | conservative |  | === | 662; 662; 662 | 9d0fb181788c6292 | YES | - |
| 2024-12-02 | balanced | anchor | === | 660; 660; 660 | 5b38652b480760c1 | YES | - |
| 2024-12-02 | growth |  | === | 658; 658; 658 | 84ac9b2b4a88ec5d | YES | - |

Cells whose k = 3 repeats were byte-identical: **12 / 12**. Divergent cells: **0**.

The registered k = 3 byte-identical standard held on every cell, so the fallback was not activated. The evaluated transcript is the first repeat per cell either way.

## 3. Per-cell results (evaluated transcript = first repeat)

| date | profile | flag | sentences | claims | T1-T6 claims | T1-T6 pass | T1-T6 fail | T7 | unaccounted | T8 flagged | T8 violations |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2012-02-01 | conservative |  | 7 | 17 | 17 | 17 | 0 | 0 | 0 | 0 | 0 |
| 2012-02-01 | balanced |  | 7 | 17 | 17 | 17 | 0 | 0 | 0 | 0 | 0 |
| 2012-02-01 | growth |  | 7 | 17 | 17 | 17 | 0 | 0 | 0 | 0 | 0 |
| 2020-03-02 | conservative |  | 7 | 17 | 17 | 17 | 0 | 0 | 0 | 0 | 0 |
| 2020-03-02 | balanced |  | 7 | 17 | 17 | 17 | 0 | 0 | 0 | 0 | 0 |
| 2020-03-02 | growth |  | 7 | 17 | 17 | 17 | 0 | 0 | 0 | 0 | 0 |
| 2022-06-01 | conservative |  | 7 | 17 | 17 | 17 | 0 | 0 | 0 | 0 | 0 |
| 2022-06-01 | balanced |  | 7 | 17 | 17 | 17 | 0 | 0 | 0 | 0 | 0 |
| 2022-06-01 | growth |  | 7 | 17 | 17 | 17 | 0 | 0 | 0 | 0 | 0 |
| 2024-12-02 | conservative |  | 7 | 17 | 17 | 17 | 0 | 0 | 0 | 0 | 0 |
| 2024-12-02 | balanced | **anchor (dev)** | 7 | 17 | 17 | 17 | 0 | 0 | 0 | 0 | 0 |
| 2024-12-02 | growth |  | 7 | 17 | 17 | 17 | 0 | 0 | 0 | 0 | 0 |

## 4. Aggregates

### Per claim type

| type | name | headline extracted (11 cells) | headline pass | headline fail | anchor extracted | anchor pass |
|---|---|---|---|---|---|---|
| T1 | numeric weight claim | 88 | 88 | 0 | 8 | 8 |
| T2 | numeric forecast-volatility claim | 11 | 11 | 0 | 1 | 1 |
| T3 | numeric attribution claim | 11 | 11 | 0 | 1 | 1 |
| T4 | directional/comparative claim | 22 | 22 | 0 | 2 | 2 |
| T5 | ranking claim | 33 | 33 | 0 | 3 | 3 |
| T6 | provenance claim | 22 | 22 | 0 | 2 | 2 |
| T7 | FABRICATED claim | 0 | 0 | 0 | 0 | 0 |

### Headline and anchor

| scope | T1-T6 claims | passed | failed | pass rate | T7 claims | unaccounted numerics | T8 violations |
|---|---|---|---|---|---|---|---|
| **headline: 11 non-development cells** | 187 | 187 | 0 | 100.0% | 0 | 0 | 0 |
| anchor cell (development, flagged, outside the headline) | 17 | 17 | 0 | 100.0% | 0 | 0 | 0 |
| all 12 cells (hard-condition scope) | 204 | 204 | 0 | - | 0 | 0 | 0 |

### Registered conditions

| condition | scope | required | measured | met |
|---|---|---|---|---|
| zero T7 (fabricated claims) | all 12 cells | 0 | 0 | yes |
| zero unaccounted numerics (T7 hard-fail path) | all 12 cells | 0 | 0 | yes |
| zero T8 violations | all 12 cells | 0 | 0 | yes |
| 100% of extracted T1-T6 claims pass | 11 headline cells | 187/187 | 187/187 | yes |

### Failing claims

None. Every extracted claim across all 12 cells ties out to the frozen artifacts under the registered matching rule.

### T8 screen -- flagged sentences listed verbatim for review

Flags any sentence containing both a profile token {conservative, balanced, growth, profile} and a forecast token {forecast, volatility, vol, predicted, expected}. A flagged sentence VIOLATES iff it attributes a forecast value or forecast difference to the profile choice. The d-vector is profile-invariant at fixed date (verified at Stage A tie-out (b): 0 breaches).

No sentence in any of the 12 evaluated transcripts was flagged: no narration placed a profile token and a forecast token in the same sentence. Zero flagged implies zero violations.

### Soft observations (recorded, never adjudicated)

No phrase from the registered qualitative-quantity lexicon appeared in any evaluated transcript. The template's style rule instructs numerals only.

## 5. Verdict 

> Narration faithfulness is supported: every extracted claim across the 11 registered evaluation cells ties out to the frozen artifacts under the registered matching rule (187/187 claims), no fabricated claims were detected, and no narration implies profile-dependent forecasts.

> This verdict attaches to the narration layer as frozen at Gate B3 -- the registered template, serializer, and options block; it does not certify free-form narration of the Recommendation.

(Amendment: the companion sentence is registered to travel with the verdict wherever it is quoted.)

The anchor cell is the development cell. Its results appear in every table above, always flagged, and are excluded from the headline counts; for the record it contributed 17/17 passing T1-T6 claims.

## 6. Exhibits

Both are regenerated from the frozen CSVs and the frozen transcripts only; no value in either is recomputed from anything else.

- **N1** -- `outputs/task13/N1_anchor_annotated.md`: the anchor-cell narration with every claim span annotated by type and pass/fail.
- **N2** -- `outputs/task13/N2_claims_summary.md` (and `N2_claims_summary.csv`): claims summary, type x cell, pass counts.

## 7. Gates

| gate | criterion | evidence | verdict |
|---|---|---|---|
| C1 | 36/36 generations in registered order; invariants I1-I4 held or deviations logged; per-call log complete | 36/36 completed; order honoured = True; I1/I2/I3 held on all 36; I4 deviations logged = 0; call log stage_c_call_log.csv written | PASS |
| C2 | every sentence classified; unaccounted-numeric scan run over every transcript | 84 sentences classified across 12 transcripts; scan run on all 12 (results feed the verdict, not this gate) | PASS |
| C3 | verdict emitted per the registered vocabulary; symmetric report; exhibits from frozen data only | one verdict form emitted with counts filled, plus the companion sentence; N1 and N2 written | PASS |

## 8. Deliverables written by this stage

- `outputs/task13/stage_c_report.md` 
- `outputs/task13/narrations/<date>_<profile>_rep{1,2,3}.txt` (36) and `stage_c_warmup_discarded.txt`
- `outputs/task13/claims_extracted.csv`, `sentence_classes.csv`, `unaccounted_numerics.csv`, `t8_screen.csv`
- `outputs/task13/determinism_log.csv` (Stage A probe rows preserved; Stage C rows appended)
- `outputs/task13/stage_c_call_log.csv`
- `outputs/task13/N1_anchor_annotated.md`, `N2_claims_summary.md`, `N2_claims_summary.csv`

STAGE C COMPLETE -- gates C1/C2/C3 PASS. One-shot discipline held: 36 generations, one extraction pass, one adjudication pass. 
