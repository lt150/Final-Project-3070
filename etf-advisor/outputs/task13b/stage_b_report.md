# Task 13b Stage B -- one-shot readability computation

Environment pin (registered convention): everything evidentiary runs under the repo venv; the system interpreter is never used.

- interpreter: `C:\Users\tianc\Desktop\Final Project\etf-advisor\.venv\Scripts\python.exe`
- python 3.13.0, numpy 2.4.6, pandas 3.0.3, scipy 1.17.1, pyarrow 25.0.0 (pinned stack, undisturbed by this task's install)
- readability machinery pinned at Stage A: **textstat 0.7.13**, **pyphen 0.18.1** (syllable backend), nltk 3.10.3

Compute FRE and FKGL once over the twelve frozen `_rep1` narration transcripts, emit the exhibit, and report the verdict against the targets registered before any computation existed.  This is the readability half of revised RQ3; the factual-consistency half is carried by the Task 13 verdict.

Extraction rule: each metric is computed on the transcript file contents verbatim -- read as UTF-8 text, passed to textstat with no stripping and no normalisation.  Gate A2 established every file carries no CR/LF, so this text read equals the raw bytes; n_words / n_sentences are textstat's own tokenisation (the inputs to its FRE/FKGL).

Rounding: FRE and FKGL are reported at one decimal place; the 11-cell means are computed at full precision and rounded last (so a mean need not equal the mean of the rounded per-cell values shown).  The registered R1/R2/R3 thresholds are evaluated on the full-precision quantities, consistent with the same rounding rule.

## Results -- all 12 cells (anchor development cell flagged and excluded from the means)

| cell_id | profile | date | flag | n_words | n_sentences | FRE | FKGL |
|---|---|---|---|---|---|---|---|
| 2012-02-01_conservative | conservative | 2012-02-01 |  | 106 | 17 | 59.2 | 6.5 |
| 2012-02-01_balanced | balanced | 2012-02-01 |  | 106 | 17 | 60.8 | 6.3 |
| 2012-02-01_growth | growth | 2012-02-01 |  | 106 | 17 | 61.6 | 6.2 |
| 2020-03-02_conservative | conservative | 2020-03-02 |  | 106 | 17 | 59.2 | 6.5 |
| 2020-03-02_balanced | balanced | 2020-03-02 |  | 106 | 17 | 60.8 | 6.3 |
| 2020-03-02_growth | growth | 2020-03-02 |  | 106 | 17 | 61.6 | 6.2 |
| 2022-06-01_conservative | conservative | 2022-06-01 |  | 106 | 17 | 59.2 | 6.5 |
| 2022-06-01_balanced | balanced | 2022-06-01 |  | 106 | 17 | 60.8 | 6.3 |
| 2022-06-01_growth | growth | 2022-06-01 |  | 106 | 17 | 61.6 | 6.2 |
| 2024-12-02_conservative | conservative | 2024-12-02 |  | 106 | 17 | 59.2 | 6.5 |
| 2024-12-02_balanced | balanced | 2024-12-02 | **anchor (dev)** | 106 | 17 | 60.8 | 6.3 |
| 2024-12-02_growth | growth | 2024-12-02 |  | 106 | 17 | 61.6 | 6.2 |
| **mean (11 eval cells)** |  |  | anchor excluded | 106.0 | 17.0 | **60.5** | **6.4** |

Exhibit written: `C:/Users/tianc/Desktop/Final Project/etf-advisor/outputs/task13b/readability_results.csv` (12 cell rows + one means row over the 11 evaluation cells).

## Registered conditions R1/R2/R3 

| condition | registered rule | measured | target | met |
|---|---|---|---|---|
| R1 (mean ease) | mean FRE over 11 eval cells >= 50 | 60.5461 (full); 60.5 (1dp) | >= 50 | yes |
| R2 (mean grade) | mean FKGL over 11 eval cells <= 12 | 6.3634 (full); 6.4 (1dp) | <= 12 | yes |
| R3 (per-cell floors) | no eval cell FRE < 30 and none FKGL > 16 | min FRE 59.2401; max FKGL 6.5455 (full) | FRE >= 30, FKGL <= 16 | yes |

## Verdict 

**Readability is supported.**

## Filled chapter sentence 

> Readability is supported: mean Flesch Reading Ease 60.5 and mean Flesch–Kincaid grade 6.4 across the 11 evaluation cells, against targets registered before computation (FRE ≥ 50, FKGL ≤ 12, with per-cell floors).

## Stop line

STAGE B COMPLETE 