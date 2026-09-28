"""Task 15 -- Streamlit demonstration UI over the frozen components.

Composition only:

- Displayed values come from the frozen artifacts, read through the frozen
  Task 13 readers (``src.task13.grounding`` / ``src.task13.serialize``); the
  date domain and the weights are the frozen
  ``outputs/task10/smoke_recommendations.csv`` cells.
- Narration text is served from the frozen Task 13 transcripts
  (``outputs/task13/narrations/``, first repeat per cell).  The app performs
  no narration generation; the only live model call in the app is
  elicitation.
- The app writes nothing anywhere.  Everything under ``src/task9-14`` and
  ``outputs/task9-14`` is read-only, and no ``docs/`` path and no ``.md``
  file is read at runtime.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

# Streamlit runs this file as a script, so the repo root goes on sys.path
# before the src.* imports.  The frozen packages resolve their own artifact
# paths from __file__, never from the working directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.task9.allocators import PROFILES
from src.task13 import NARRATION_DIR
from src.task13.adjudicate import d_advisor, d_phi, d_vol
from src.task13.grounding import Cell, load as load_grounding
from src.task13.serialize import (
    attribution_input,
    attribution_output,
    pct1,
    pct_2sf,
    pp1,
)
from src.task14 import PERSONA_PATH
from src.task14.elicit import ElicitationResult, elicit
from src.task15 import ui as ui_copy

# --------------------------------------------------------------------------- #
# Registered fixtures
# --------------------------------------------------------------------------- #
# The five questions, transcribed once from docs/task14_planning_register.md
# Sec. 1
QUESTIONS: dict[str, str] = {
    "Q1": "When do you expect to need most of this money?",
    "Q2": "Suppose your portfolio lost 20% of its value within a year. "
          "What would you most likely do?",
    "Q3": "What matters more to you: protecting what you have, or growing it "
          "as much as possible over time?",
    "Q4": "How stable is your income, and do you have an emergency fund "
          "covering several months of expenses?",
    "Q5": "Have you invested in markets before, and how comfortable are you "
          "with market fluctuations?",
}

# Persona prefills: one clear-cut persona per profile, read at runtime from
# src/task14/personas.json.  Chosen ids: P-C1 (conservative), P-B1 (balanced),
# P-G1 (growth) -- each recorded "correct profile" (credit 1.0) for the
# shipped arm in the frozen Task 14 outcomes (outputs/task14/outcomes.csv).
PERSONA_IDS: tuple[str, ...] = ("P-C1", "P-B1", "P-G1")
PERSONA_NONE = "-- select a persona --"

# Exact registered strings.
DISCLOSURE_SCOPE = ("This demonstration is restricted to the four registered "
                    "dates and the three registered risk profiles.")
DISCLOSURE_OOD = ("The model being evaluated was tested only on a predefined set of approved user/persona profiles, "
                  "it has not been established that it works reliably with arbitary, unexpected text "
                  "relative to this evaluation.")
MALFORMED_BANNER = ("Elicitation returned runtime status: malformed. Your "
                    "answers are preserved; adjust any answer and resubmit.")
PROVENANCE_LINE = ("Narration shown is the frozen registered transcript for "
                   "this cell.")
SESSION_NOTE = ("Profiles elicited this session are stored for this browser "
                "session only and clear on refresh.")


# --------------------------------------------------------------------------- #
# Frozen-component access
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def grounding():
    """The frozen Task 13 grounding, loaded once per server process."""
    return load_grounding()


def load_personas() -> dict[str, dict]:
    data = json.loads(PERSONA_PATH.read_text(encoding="utf-8"))
    by_id = {p["id"]: p for p in data["personas"]}
    return {pid: by_id[pid] for pid in PERSONA_IDS}


def narration_text(date: str, profile: str) -> str:
    """The frozen registered transcript for the cell: its first repeat."""
    return (NARRATION_DIR / f"{date}_{profile}_rep1.txt").read_text(
        encoding="utf-8")


def md_table(header: tuple[str, str], rows: list[tuple[str, str]]) -> str:
    # Rendered as a markdown table rather than st.table/st.dataframe so the
    # display path involves no Arrow serialization of the frozen values.
    lines = [f"| {header[0]} | {header[1]} |", "| --- | --- |"]
    lines += [f"| {a} | {b} |" for a, b in rows]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
def init_state() -> None:
    ss = st.session_state
    ss.setdefault("view", "elicit")
    ss.setdefault("recall", {})            # profile -> ElicitationResult
    ss.setdefault("current_profile", None)
    ss.setdefault("clarify_q", None)       # 1..5 while a re-ask is pending
    ss.setdefault("malformed", False)
    # Streamlit culls a keyed widget's session_state entry after any run that
    # does not render the widget; re-assigning each run keeps typed answers,
    # the persona selection and the date selection alive across view switches
    # (keep all other answers).
    for qk in QUESTIONS:
        ss[f"ans_{qk}"] = ss.get(f"ans_{qk}", "")
    for wk in ("persona_choice", "date_choice", "recall_choice"):
        if wk in ss:
            ss[wk] = ss[wk]


def apply_persona() -> None:
    """Prefill all five fields from the chosen persona; fields stay editable."""
    pid = st.session_state.persona_choice
    if pid == PERSONA_NONE:
        return
    persona = load_personas()[pid]
    for qk, text in persona["answers"].items():
        st.session_state[f"ans_{qk}"] = text


def go_elicit() -> None:
    st.session_state.view = "elicit"


def go_recommend() -> None:
    """Return to the recommendation view for the current recalled profile --
    no model call.  Needed when only one profile has been elicited: the recall
    radio's on_change never fires for an already-selected sole option."""
    ss = st.session_state
    if ss.current_profile is None and ss.recall:
        ss.current_profile = next(iter(ss.recall))
    ss.view = "recommend"


def recall_switch() -> None:
    """Switch among profiles elicited this session -- no model call."""
    ss = st.session_state
    ss.current_profile = ss.recall_choice
    ss.view = "recommend"


def handle_result(result: ElicitationResult) -> None:
    """Route the frozen contract's outcomes; exactly one holds per call."""
    ss = st.session_state
    ss.malformed = False
    if result.malformed:
        ss.clarify_q = None
        ss.malformed = True
    elif result.profile is not None:
        ss.clarify_q = None
        ss.recall[result.profile] = result
        ss.current_profile = result.profile
        ss.view = "recommend"
    elif result.clarify_question is not None:
        ss.clarify_q = int(result.clarify_question)
    else:
        # Unreachable for the shipped Arm A;
        # surfaced rather than defaulted if it ever occurs.
        st.error("Elicitation returned a clarify signal without a question "
                 f"index: {result!r}")
        return
    st.rerun()


# --------------------------------------------------------------------------- #
# Screens
# --------------------------------------------------------------------------- #
def sidebar() -> None:
    ss = st.session_state
    with st.sidebar:
        st.subheader("Session recall")
        st.caption(SESSION_NOTE)
        elicited = [p for p in PROFILES if p in ss.recall]
        if elicited:
            if (ss.current_profile in elicited
                    and ss.get("recall_choice") != ss.current_profile):
                ss.recall_choice = ss.current_profile
            st.radio("Profiles elicited this session", elicited,
                     key="recall_choice", on_change=recall_switch)
        else:
            st.caption("No profiles elicited yet this session.")
        if ss.view == "recommend":
            st.button("New elicitation", on_click=go_elicit)
        elif ss.recall:
            st.button("Back to recommendation", on_click=go_recommend)


def elicitation_screen() -> None:
    ss = st.session_state
    st.header("Risk-preference elicitation")
    st.markdown(DISCLOSURE_SCOPE)
    st.markdown(DISCLOSURE_OOD)
    ui_copy.render_not_advice()
    ui_copy.render_glossary()

    if ss.malformed:
        st.error(MALFORMED_BANNER)

    try:
        personas = load_personas()
    except Exception as exc:
        st.error(f"{type(exc).__name__}: {exc}")
        st.stop()

    st.selectbox(
        "Load registered persona",
        options=(PERSONA_NONE, *PERSONA_IDS),
        format_func=lambda pid: (
            pid if pid == PERSONA_NONE
            else f"{pid} ({personas[pid]['expected_profile']})"),
        key="persona_choice",
        on_change=apply_persona,
    )

    clarify_q = ss.clarify_q
    if clarify_q is not None:
        st.warning(
            f"Clarification requested for Q{clarify_q}: "
            f"\"{QUESTIONS[f'Q{clarify_q}']}\" -- please revise that answer "
            "only; your other answers are kept, and resubmitting runs a new "
            "elicitation call."
        )

    for i, qk in enumerate(QUESTIONS, start=1):
        qline = f"**Q{i}.** {QUESTIONS[qk]}"
        if clarify_q == i:
            qline = f":red[{qline}]"
        st.markdown(qline)
        st.text_input(f"Answer to Q{i}", key=f"ans_{qk}",
                      label_visibility="collapsed")
        # captions define a word in Q2/Q4 only; never hint at an answer.
        if qk in ui_copy.QUESTION_CAPTIONS:
            st.caption(ui_copy.QUESTION_CAPTIONS[qk])

    if st.button("Submit answers", type="primary"):
        answers = {qk: ss[f"ans_{qk}"] for qk in QUESTIONS}
        result = None
        with st.spinner("Eliciting risk profile -- expect roughly 20-60 s on "
                        "CPU; the first call adds ~10-25 s of model load..."):
            try:
                result = elicit(answers)
            except Exception as exc:
                st.error(f"{type(exc).__name__}: {exc}")
        if result is not None:
            handle_result(result)

    ui_copy.render_faq()                 # bottom of the main area


def recommendation_view() -> None:
    ss = st.session_state
    profile = ss.current_profile
    st.header("Recommendation")
    ui_copy.render_not_advice()
    ui_copy.render_glossary()

    try:
        g = grounding()
        dates = sorted({c.date for c in g.cells})
    except Exception as exc:
        st.error(f"{type(exc).__name__}: {exc}")
        st.stop()
        return

    date = st.selectbox("As-of date (registered dates only)", dates,
                        key="date_choice")
    st.markdown(f"Risk profile, from elicitation this session: **{profile}**")

    try:
        cell = Cell(date, profile)

        # Plain-language header + optional weight bars.  The pct1 text
        # values remain the displayed values, at the same format and precision;
        # each bar is only a visual proportional to the same frozen weight.
        st.subheader("Your recommended mix")
        for t in g.tickers:
            w = d_advisor(g, cell, t)
            c1, c2, c3 = st.columns([1, 5, 1], vertical_alignment="center")
            c1.markdown(f"**{t}**")
            c2.progress(float(w))
            c3.markdown(pct1(w))
        ui_copy.render_fund_holdings()   # near the weights

        st.subheader("How much these funds typically move day to day")
        st.markdown(md_table(
            ("ticker", "forecast daily volatility"),
            [(t, pct_2sf(d_vol(g, cell, t))) for t in g.tickers],
        ))
        st.caption("Daily volatility, percent (the frozen x100 presentation, "
                   "2 significant figures).")

        st.subheader("What influenced this mix")
        out = attribution_output(g, cell)
        inp = attribution_input(g, cell, out)
        st.markdown("\n".join([
            f"- attribution target ticker: {out}",
            f"- attribution source ticker: {inp}",
            f"- attribution value: {pp1(d_phi(g, cell, inp, out))} "
            "percentage points",
        ]))

        st.subheader("The explanation")
        st.markdown(narration_text(date, profile))
        st.caption(PROVENANCE_LINE)      # provenance stays directly beneath
    except Exception as exc:
        st.error(f"{type(exc).__name__}: {exc}")
        st.stop()

    ui_copy.render_faq()                 # bottom of the main area


def main() -> None:
    st.set_page_config(page_title="ETF Advisor -- demonstration UI (Task 15)")
    st.title("ETF Advisor")
    init_state()
    sidebar()
    if st.session_state.view == "recommend" and st.session_state.current_profile:
        recommendation_view()
    else:
        elicitation_screen()


main()
