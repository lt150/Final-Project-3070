# ETF Advisor

Final-year project: a hybrid AI financial advisor that elicits an investor's
risk preference, recommends an ETF portfolio across 8 ETFs, and explains the
recommendation in plain language.

> Educational project only. Not financial advice.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run src/task15/app.py
```

The risk-preference questionnaire calls a local LLM through
[Ollama](https://ollama.com). Install Ollama, then:

```bash
ollama pull qwen3.5:9b
```

## Project structure

| Folder           | What it does                                                                                           |
| ---------------- | ------------------------------------------------------------------------------------------------------ |
| `src/data/`      | Data pipeline (Task 7): downloads ETF prices and builds the dataset. Run `python -m src.data.pipeline` |
| `src/task8/`     | LSTM forward-volatility forecaster                                                                     |
| `src/task9/`     | Forecaster deployment, shared covariance, baseline allocators                                          |
| `src/task10/`    | Recommendation layer (`recommend(profile, as_of)`)                                                     |
| `src/task11/`    | Backtesting harness and hypothesis H1                                                                  |
| `src/task12/`    | SHAP explainability and attribution faithfulness                                                       |
| `src/task13/`    | LLM narration of a recommendation, and its faithfulness evaluation                                     |
| `src/task13b/`   | Readability scores of the Task 13 narrations                                                           |
| `src/task14/`    | LLM risk-preference elicitor (two prompt designs compared)                                             |
| `src/task15/`    | Streamlit demo app combining all of the above                                                          |
| `outputs/taskN/` | Reports, figures and results for each task                                                             |
| `notebooks/`     | Exploratory data analysis                                                                              |

## Reproducing results

Each task has stage scripts (`stage_a.py`, `stage_b.py`, ...) that write a
report to `outputs/taskN/`. These reports are the frozen record of the
results. Some scripts refuse to overwrite a failure record, and the Task 13 and
14 scripts need Ollama running.

`data/` is included
