# Task 13 Stage A -- pin capture, grounding tie-outs, determinism measurement

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
```

## Module and boundary

Build the narration layer: local-LLM narration (Ollama, `qwen3.5:9b`) of a frozen Recommendation and its attributions, plus a pre-registered evaluation of the narration's faithfulness TO those frozen artifacts.

> The Task 12 verdict is frozen ground truth. If narration fails faithfulness, that is a Task 13 finding about the narration layer and re-litigates nothing upstream. attributions_main.csv, the Recommendation interface, and all Task 9-12 outputs are read-only.

## Pin capture (Stage A step 1) -- imported from the live host

| field | value as reported |
|---|---|
| server_version | `0.32.5` |
| model_tag | `qwen3.5:9b` |
| model_digest | `6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7` |
| model_size_bytes | `6594474711` |
| parameter_size | `9.7B` |
| quantization_level | `Q4_K_M` |
| format | `gguf` |
| family | `qwen35` |
| families | `['qwen35']` |
| parent_model | `(none)` |
| architecture | `qwen35` |
| parameter_count | `9653104368` |
| file_type | `15` |
| native_context_length | `262144` |
| template | `{{ .Prompt }}` |
| model_default_parameters | `presence_penalty 1.5\ntemperature 1\ntop_k 20\ntop_p 0.95` (newlines shown as `\n`; see the note below the table) |
| capabilities | `['completion', 'vision', 'tools', 'thinking']` |
| requires | `0.17.1` |
| renderer_parser | `TEMPLATE {{ .Prompt }}; RENDERER qwen3.5; PARSER qwen3.5` |

Fields reported ABSENT: none.


## Import posture

No new Python installs are authorized in this task; `shap` and `torch` are imported nowhere in task13 code; all HTTP to Ollama is stdlib `urllib.request` (no `ollama` package, no `requests`).

| file | statement | target |
|---|---|---|
| __init__.py | `from __future__ import annotations` | `__future__` |
| __init__.py | `import sys` | `sys` |
| __init__.py | `from datetime import datetime, timezone` | `datetime` |
| __init__.py | `from pathlib import Path` | `pathlib` |
| __init__.py | `from typing import Sequence` | `typing` |
| __init__.py | `import numpy` | `numpy` |
| __init__.py | `import pandas` | `pandas` |
| __init__.py | `import pyarrow` | `pyarrow` |
| __init__.py | `import scipy` | `scipy` |
| __init__.py | `import ast` | `ast` |
| client.py | `from __future__ import annotations` | `__future__` |
| client.py | `import json` | `json` |
| client.py | `import time` | `time` |
| client.py | `import urllib.error` | `urllib` |
| client.py | `import urllib.request` | `urllib` |
| client.py | `from dataclasses import dataclass, field` | `dataclasses` |
| client.py | `from typing import Any` | `typing` |
| client.py | `from . import ENDPOINT, KEEP_ALIVE, I4_PROMPT_BUDGET, MODEL, OPTIONS, SHOW_ENDPOINT, STREAM, TAGS_ENDPOINT, THINK, VERSION_ENDPOINT` | `src.task13.` |
| grounding.py | `from __future__ import annotations` | `__future__` |
| grounding.py | `import csv` | `csv` |
| grounding.py | `from dataclasses import dataclass` | `dataclasses` |
| grounding.py | `from pathlib import Path` | `pathlib` |
| grounding.py | `from . import ANCHOR_CELL, ATTRIBUTIONS_MAIN, PROFILE_ORDER, SMOKE_RECOMMENDATIONS, SMOKE_WEIGHTS_CANDIDATES` | `src.task13.` |
| stage_a.py | `from __future__ import annotations` | `__future__` |
| stage_a.py | `import csv` | `csv` |
| stage_a.py | `import json` | `json` |
| stage_a.py | `import os` | `os` |
| stage_a.py | `import sys` | `sys` |
| stage_a.py | `from pathlib import Path` | `pathlib` |
| stage_a.py | `from . import PROFILE_ORDER, ANCHOR_CELL, EFFICIENCY_TOL, FORBIDDEN_IMPORTS, I4_PROMPT_BUDGET, K_REPEATS, NARRATION_DIR, OUT_DIR, PROBE_PROMPT, ascii_escape, env_header, md_table, package_imports, write_report` | `src.task13.` |
| stage_a.py | `from . import client` | `src.task13.` |
| stage_a.py | `from . import grounding` | `src.task13.` |
| stage_a.py | `import winreg` | `winreg` |

Forbidden targets ['shap', 'torch', 'requests', 'ollama']: 0 present.

## Grounding tie-outs (Stage A step 2)

### Schema binding -- READ from the frozen headers, never retyped

| surface | resolved path / read value |
|---|---|
| advisor weights | `C:/Users/tianc/Desktop/Final Project/etf-advisor/outputs/task10/smoke_recommendations.csv (w_* columns)` |
| forecast_vols | `C:/Users/tianc/Desktop/Final Project/etf-advisor/outputs/task10/smoke_recommendations.csv (d_* columns)` |
| classical weights | `C:/Users/tianc/Desktop/Final Project/etf-advisor/outputs/task9/smoke_weights.csv (allocator_id=rb_classical)` |
| attributions | `C:/Users/tianc/Desktop/Final Project/etf-advisor/outputs/task12/attributions_main.csv (long: feature x output -> phi)` |
| cell dates | `C:/Users/tianc/Desktop/Final Project/etf-advisor/outputs/task10/smoke_recommendations.csv (date column, sorted)` |
| canonical ticker order | `C:/Users/tianc/Desktop/Final Project/etf-advisor/outputs/task10/smoke_recommendations.csv (w_* header order)` |
| smoke_weights.csv resolution | `C:/Users/tianc/Desktop/Final Project/etf-advisor/outputs/task9/smoke_weights.csv` (registered rule: task10 first, then task9) |
| canonical ticker order | `SPY, QQQ, EFA, EEM, AGG, IEF, GLD, VNQ` |
| cell dates | `2012-02-01, 2020-03-02, 2022-06-01, 2024-12-02` |
| profile order | `conservative, balanced, growth` |
| attribution orientation | long form `date, profile, feature, output, phi`; `feature` = input ticker, `output` = output ticker |

Every grounding surface was supplied by a frozen CSV. `recommend()` was NOT called: `forecast_vols` are present on `smoke_recommendations.csv` as the `d_*` columns, so the fallback (calling frozen `recommend()` read-only with a repr-exact cross-check) was not needed and was not exercised.

### (a) Loader reproduces the frozen CSV values repr-exactly

| surface | check | values checked | mismatches | first mismatch |
|---|---|---|---|---|
| advisor weights | `repr(float(token)) == token` | 96 | 0 | - |
| forecast_vols | `repr(float(token)) == token` | 96 | 0 | - |
| classical weights | `repr(float(token)) == token` | 96 | 0 | - |
| attributions | `repr(float(token)) == token` | 768 | 0 | - |

### (b) Cell completeness

| check | expected | got | verdict |
|---|---|---|---|
| cells (dates x profiles) | 12 | 12 | PASS |
| tickers | 8 | 8 | PASS |
| advisor-weight cells | 12 | 12 | PASS |
| forecast_vol cells | 12 | 12 | PASS |
| classical-weight cells | 12 | 12 | PASS |
| attribution cells | 12 | 12 | PASS |
| advisor entries (cells x tickers) | 96 | 96 | PASS |
| forecast_vol entries | 96 | 96 | PASS |
| classical entries | 96 | 96 | PASS |
| attribution rows | 768 | 768 | PASS |
| attribution entries (cells x feature x output) | 768 | 768 | PASS |
| anchor cell present | True | True | PASS |
| attribution (feature,output) holes | 0 | 0 | PASS |
| d-vector profile-invariance breaches | 0 | 0 | PASS |
| advisor weights: smoke_recommendations vs smoke_weights rb_advisor (repr-exact) | 0 | 0 | PASS |

### (c) Efficiency cross-check

Per (cell, output): `sum_features phi == w_advisor - w_classical`, tolerance 1e-12 (engine arithmetic). This reproduces the certified Task 12 identity purely to prove the loaders read the artifacts correctly; it re-litigates nothing upstream.

| date | profile | max abs gap | verdict |
|---|---|---|---|
| 2012-02-01 | conservative | 1.388e-17 | PASS |
| 2012-02-01 | balanced | 3.469e-17 | PASS |
| 2012-02-01 | growth | 2.776e-17 | PASS |
| 2020-03-02 | conservative | 6.939e-18 | PASS |
| 2020-03-02 | balanced | 5.204e-18 | PASS |
| 2020-03-02 | growth | 6.939e-18 | PASS |
| 2022-06-01 | conservative | 6.939e-17 | PASS |
| 2022-06-01 | balanced | 1.214e-17 | PASS |
| 2022-06-01 | growth | 6.505e-18 | PASS |
| 2024-12-02 | conservative | 6.939e-18 | PASS |
| 2024-12-02 | balanced | 6.939e-18 | PASS |
| 2024-12-02 | growth | 1.388e-17 | PASS |

Worst gap over all 12 cells: `6.938894e-17` (tolerance `1e-12`).

## Determinism measurement (Stage A step 3)

Anchor cell `2024-12-02 / balanced`; registered probe prompt (fixed, deliberately NOT the production template):

> Describe in exactly three sentences what a diversified ETF portfolio is.

One discarded warm-up call, then k = 3 warm generations under the frozen options block.

| seq | purpose | prompt_eval_count | I4 (<= 7680) | eval_count | done_reason | load_duration_s | wall_s | eval tok/s |
|---|---|---|---|---|---|---|---|---|
| 1 | stage_a:warmup_discarded | 24 | PASS | 88 | `stop` | 10.57 | 23.58 | 8.38 |
| 2 | stage_a:probe_rep1 | 24 | PASS | 88 | `stop` | 0.35 | 13.12 | 8.36 |
| 3 | stage_a:probe_rep2 | 24 | PASS | 88 | `stop` | 0.35 | 13.20 | 8.29 |
| 4 | stage_a:probe_rep3 | 24 | PASS | 88 | `stop` | 0.35 | 13.16 | 8.29 |

| invariant | exercised on | outcome |
|---|---|---|
| I1 model == pinned tag | 4 calls | held on all |
| I2 no thinking content | 4 calls | held on all |
| I3 done == true | 4 calls | held on all |
| I4 prompt_eval_count <= 7680 | 4 calls | held on all |

I2 evidence strings, per call: ['none', 'none', 'none', 'none'].

### Byte-identical comparison across the k = 3 repeats

| repeat | response chars | sha256 | == rep1 |
|---|---|---|---|
| 1 | 553 | e478f60d4193b77a | yes |
| 2 | 553 | e478f60d4193b77a | yes |
| 3 | 553 | e478f60d4193b77a | yes |

**Byte-identical across k = 3: YES.** First divergence: -.

Determinism standard IN FORCE: **repeat-exact** -- the k = 3 repeats of the registered probe were byte-identical, so the registered k = 3 byte-identical standard is carried into Stage C unmodified.

Reproducibility standard, stated in the registered words: **repr-exact where achievable; transcript-freeze with disclosure where not.** Narration is the first component in the stack without repr-exact guarantees.

### Probe transcript, repeat 1

```
A diversified ETF portfolio consists of multiple exchange-traded funds that collectively hold assets across various sectors, geographies, and asset classes to spread investment risk. By combining these broad-market or thematic funds into a single holding structure, investors gain instant exposure to thousands of individual securities without needing to buy each one separately. This strategy aims to smooth out the volatility inherent in any single market segment while providing balanced long-term growth potential through reduced concentration risk.
```

## Gates

| gate | criterion | evidence | verdict |
|---|---|---|---|
| A1 | pin captured complete; I1-I4 implemented and exercised on the probe calls; I2 held | 20 pin fields, 0 ABSENT; 4 probe calls; I2 held on all | PASS |
| A2 | grounding tie-outs (a)-(c) all pass | (a) 0 mismatches over 1056 values; (b) 0 failures over 15 checks; (c) worst gap 6.939e-17 <= 1e-12 | PASS |
| A3 | determinism check EXECUTED and standard in force recorded | warm-up + k=3 probe generations executed; byte-identical = True; recorded standard | PASS |

## Deliverables written by this stage

- `outputs/task13/stage_a_report.md` (this file)
- `outputs/task13/stage_a_call_log.csv` (per-call log, 4 calls)
- `outputs/task13/determinism_log.csv` (probe repeats)
- `outputs/task13/narrations/probe_rep{1,2,3}.txt`, `probe_warmup_discarded.txt`


STAGE A COMPLETE -- gates A1/A2/A3 PASS. 