# Task 13b Stage A -- machinery and input verification

Environment pin (registered convention): everything evidentiary runs under the repo venv; the system interpreter is never used.

- interpreter: `C:\Users\tianc\Desktop\Final Project\etf-advisor\.venv\Scripts\python.exe`
- python 3.13.0, numpy 2.4.6, pandas 3.0.3, scipy 1.17.1, pyarrow 25.0.0 (pinned stack, undisturbed by this task's install)
- readability machinery pinned at Stage A: **textstat 0.7.13**, **pyphen 0.18.1** (syllable backend), nltk 3.10.3

Verify the readability machinery and the frozen inputs BEFORE any evidentiary computation.  **Hard rule: Stage A performs no readability computation on any frozen transcript.**  The only FRE/FKGL calls here are on synthetic fixtures inside Gate A4.

Task 13 Stage C report of record: `C:/Users/tianc/Desktop/Final Project/etf-advisor/outputs/task13/task13_stage_c_report.md` (supplies the cell map, the per-cell rep1 SHA-256 prefixes, and the response-char counts used as frozen-integrity anchors).

## Gate A1 -- cell map (twelve transcripts, anchor flagged)

**PASS**

- resolved input directory: `C:/Users/tianc/Desktop/Final Project/etf-advisor/outputs/task13/narrations` (the directory the Task 13 Stage C report Section 8 wrote the `<date>_<profile>_rep{1,2,3}.txt` transcripts to, and the directory the Task 15 interface reads by file read).
- files present: 12 / 12; registered cells: 12; count exact: True.
- anchor development cell identified and flagged exactly once (2024-12-02 / balanced): True.

| cell_id | date | profile | flag | resolved path | exists |
|---|---|---|---|---|---|
| 2012-02-01_conservative | 2012-02-01 | conservative |  | C:/Users/tianc/Desktop/Final Project/etf-advisor/outputs/task13/narrations/2012-02-01_conservative_rep1.txt | yes |
| 2012-02-01_balanced | 2012-02-01 | balanced |  | C:/Users/tianc/Desktop/Final Project/etf-advisor/outputs/task13/narrations/2012-02-01_balanced_rep1.txt | yes |
| 2012-02-01_growth | 2012-02-01 | growth |  | C:/Users/tianc/Desktop/Final Project/etf-advisor/outputs/task13/narrations/2012-02-01_growth_rep1.txt | yes |
| 2020-03-02_conservative | 2020-03-02 | conservative |  | C:/Users/tianc/Desktop/Final Project/etf-advisor/outputs/task13/narrations/2020-03-02_conservative_rep1.txt | yes |
| 2020-03-02_balanced | 2020-03-02 | balanced |  | C:/Users/tianc/Desktop/Final Project/etf-advisor/outputs/task13/narrations/2020-03-02_balanced_rep1.txt | yes |
| 2020-03-02_growth | 2020-03-02 | growth |  | C:/Users/tianc/Desktop/Final Project/etf-advisor/outputs/task13/narrations/2020-03-02_growth_rep1.txt | yes |
| 2022-06-01_conservative | 2022-06-01 | conservative |  | C:/Users/tianc/Desktop/Final Project/etf-advisor/outputs/task13/narrations/2022-06-01_conservative_rep1.txt | yes |
| 2022-06-01_balanced | 2022-06-01 | balanced |  | C:/Users/tianc/Desktop/Final Project/etf-advisor/outputs/task13/narrations/2022-06-01_balanced_rep1.txt | yes |
| 2022-06-01_growth | 2022-06-01 | growth |  | C:/Users/tianc/Desktop/Final Project/etf-advisor/outputs/task13/narrations/2022-06-01_growth_rep1.txt | yes |
| 2024-12-02_conservative | 2024-12-02 | conservative |  | C:/Users/tianc/Desktop/Final Project/etf-advisor/outputs/task13/narrations/2024-12-02_conservative_rep1.txt | yes |
| 2024-12-02_balanced | 2024-12-02 | balanced | anchor (dev) | C:/Users/tianc/Desktop/Final Project/etf-advisor/outputs/task13/narrations/2024-12-02_balanced_rep1.txt | yes |
| 2024-12-02_growth | 2024-12-02 | growth |  | C:/Users/tianc/Desktop/Final Project/etf-advisor/outputs/task13/narrations/2024-12-02_growth_rep1.txt | yes |

## Gate A2 -- frozen-integrity (SHA-256 vs Task 13 Stage C records)

**PASS**

- The Task 13 Stage C report records only the first 16 hex of each rep1 SHA-256 (Section 2 determinism table); Gate A2 recomputes the full SHA-256 of each on-disk file and confirms its 16-hex prefix equals the registered value.  The full 64-hex digests are recorded here as the complete frozen-integrity anchor for downstream reference.
- Every file carries no CR and no LF (single-line prose), so the raw-byte digest and the text-mode digest coincide (last column pair) and the extraction rule's `verbatim file contents` is unambiguous between a byte read and a text read.
- File sizes (bytes) equal the Task 13 Stage C `response chars` for all twelve cells, an independent corroboration of the hash match.

| cell_id | flag | size_bytes | resp_chars(rpt) | size==rpt | sha16(text) | expected | match | raw==exp | full sha256 |
|---|---|---|---|---|---|---|---|---|---|
| 2012-02-01_conservative |  | 663 | 663 | yes | 1175158ad6222851 | 1175158ad6222851 | yes | yes | 1175158ad6222851bfdff489d9eb3cdde6bb01b6018100f3c2e5516b1c07d53c |
| 2012-02-01_balanced |  | 660 | 660 | yes | 6519acedf8cc3ab9 | 6519acedf8cc3ab9 | yes | yes | 6519acedf8cc3ab93bb4a05eb0a15ec761cbd625e8d983e0e0023d026901ec5e |
| 2012-02-01_growth |  | 658 | 658 | yes | 2f612a038425ca08 | 2f612a038425ca08 | yes | yes | 2f612a038425ca08a54701f81c9188ccc7efe4a5559331df09679cb2d84cdcb7 |
| 2020-03-02_conservative |  | 663 | 663 | yes | 9e95240b1b187a51 | 9e95240b1b187a51 | yes | yes | 9e95240b1b187a51bc53ae9d9a7d1d3b064f99a372d7c0f88c8a5fb981a48f95 |
| 2020-03-02_balanced |  | 659 | 659 | yes | 718897b4e7e60a5b | 718897b4e7e60a5b | yes | yes | 718897b4e7e60a5bf478bd1249ea15e33dfa4e5e5f4c3cd30039b6c8acaab56a |
| 2020-03-02_growth |  | 656 | 656 | yes | a02283a0404ea98d | a02283a0404ea98d | yes | yes | a02283a0404ea98dd1f1bbd05bd28ee6fca39562070b7f82e5e53927361be73d |
| 2022-06-01_conservative |  | 663 | 663 | yes | 99f859cf0e6bc4aa | 99f859cf0e6bc4aa | yes | yes | 99f859cf0e6bc4aaee58a280e64502b90448503f84d162e7a74e8806908bcc91 |
| 2022-06-01_balanced |  | 660 | 660 | yes | 546ad58b0b9be787 | 546ad58b0b9be787 | yes | yes | 546ad58b0b9be7874076c2976ba293dfbfc0429aafac724c7e22ab992a51c294 |
| 2022-06-01_growth |  | 659 | 659 | yes | db55cab141c0f778 | db55cab141c0f778 | yes | yes | db55cab141c0f7784709a99f3d34ee19c684cddd3970ffc868b8607d3bb24f44 |
| 2024-12-02_conservative |  | 662 | 662 | yes | 9d0fb181788c6292 | 9d0fb181788c6292 | yes | yes | 9d0fb181788c62927f26bc3d36f4aac4ccf8e711328993db6fd66198d192018b |
| 2024-12-02_balanced | anchor | 660 | 660 | yes | 5b38652b480760c1 | 5b38652b480760c1 | yes | yes | 5b38652b480760c1b0be0e116837754be573a12b6c80fb81cf05d7ce3386d1de |
| 2024-12-02_growth |  | 658 | 658 | yes | 84ac9b2b4a88ec5d | 84ac9b2b4a88ec5d | yes | yes | 84ac9b2b4a88ec5d1751c3d4ac344f18c60a2bbf3303e4df9bb4f9968916d720 |

## Gate A3 -- format check (plain narration prose, no wrapper)

**PASS**

- Registered markers scanned for (any hit == a serialization wrapper or non-displayed material, which per brief Sec. 1 halts the task): `{`, `}`, `[`, `]`, `<`, `>`, ```, `````, `"response"`, `'response'`, `\n`, `\t`, `\u`, `<think`, `</think`, `system\n`, `assistant\n`, `user\n`.
- Every transcript is a single line of prose, begins with a capital letter, ends with a period, and contains none of the markers -- i.e. plain narration prose exactly as displayed to the user.

| cell_id | flag | single_line | starts_upper | ends_period | wrapper hits | verdict |
|---|---|---|---|---|---|---|
| 2012-02-01_conservative |  | yes | yes | yes | none | PLAIN |
| 2012-02-01_balanced |  | yes | yes | yes | none | PLAIN |
| 2012-02-01_growth |  | yes | yes | yes | none | PLAIN |
| 2020-03-02_conservative |  | yes | yes | yes | none | PLAIN |
| 2020-03-02_balanced |  | yes | yes | yes | none | PLAIN |
| 2020-03-02_growth |  | yes | yes | yes | none | PLAIN |
| 2022-06-01_conservative |  | yes | yes | yes | none | PLAIN |
| 2022-06-01_balanced |  | yes | yes | yes | none | PLAIN |
| 2022-06-01_growth |  | yes | yes | yes | none | PLAIN |
| 2024-12-02_conservative |  | yes | yes | yes | none | PLAIN |
| 2024-12-02_balanced | anchor | yes | yes | yes | none | PLAIN |
| 2024-12-02_growth |  | yes | yes | yes | none | PLAIN |

## Gate A4 -- tool determinism + direction fixture (synthetic text only)

**PASS**

- python 3.13.0, textstat 0.7.13, pyphen 0.18.1, nltk 3.10.3.
- The fixture (`src/task13b/_fixture.py`) is invoked twice by this driver via subprocess, i.e. two independent OS processes; their stdout is compared byte-for-byte.  No frozen transcript is read by the fixture or by this gate.

| check | result |
|---|---|
| determinism (2 separate processes, byte-identical stdout) | yes |
| default language == en_US (library default) | en_US -> yes |
| direction FRE: simple > complex | 111.5342 > -200.6800 -> True |
| direction FKGL: simple < complex | -0.9725 < 45.2250 -> True |
| fixture PARAGRAPH (FRE / FKGL) | 110.39266666666667 / -0.06799999999999784 |

## Stage A verdict

All four gates PASS.  The twelve frozen `_rep1` transcripts are present, tie out byte-for-byte to the Task 13 Stage C records, and are plain narration prose as displayed; the textstat/pyphen machinery is deterministic across processes and directionally sane, with the registered en_US default language and versions pinned above.  The extraction rule is satisfiable verbatim (no wrapper, no internal newlines).

**STOP -- Stage A complete. 