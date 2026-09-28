"""Task 15: beginner-friendly UI.

The app still reads no ``docs/`` path and no ``.md`` file at runtime.
"""

from __future__ import annotations

import streamlit as st

# --------------------------------------------------------------------------- #
# 2.1 -- not-advice line (new registered exact string), rendered on BOTH screens
# --------------------------------------------------------------------------- #
NOT_ADVICE = ("This is a university demonstration for educational purposes. "
              "It is not financial advice.")

# --------------------------------------------------------------------------- #
# 3.1 -- glossary (term, definition), pasted verbatim
# --------------------------------------------------------------------------- #
GLOSSARY: tuple[tuple[str, str], ...] = (
    ("ETF (exchange-traded fund)",
     "a single investment that holds a whole basket of stocks or bonds. "
     "Buying one share spreads your money across many holdings at once."),
    ("Risk profile",
     "a label (conservative, balanced, or growth) describing how much "
     "short-term movement in value an investor can accept in exchange for "
     "higher potential long-term returns."),
    ("Weight (allocation)",
     "the share of the portfolio placed in each fund. All the weights together "
     "add up to 100%."),
    ("Volatility",
     "how much a fund's price typically moves in a single day. The figures "
     "shown here are daily; bigger numbers mean a bumpier ride."),
    ("Attribution",
     "a measure of how much each input (such as a fund's forecast volatility) "
     "pushed the recommendation toward its final mix. It explains influence; "
     "it does not predict returns."),
    ("Diversification",
     "spreading money across different kinds of investments so that no single "
     "one can sink the whole portfolio."),
    ("Emergency fund",
     "savings you could live on for a few months if your income stopped."),
    ("Index",
     "a published list that tracks a market segment, such as the S&P 500 "
     "tracking about 500 large US companies. Index funds simply follow the "
     "list."),
)

# --------------------------------------------------------------------------- #
# 3.2 -- fund one-line descriptions (ticker, description) + roles sentence
# --------------------------------------------------------------------------- #
FUND_HOLDINGS: tuple[tuple[str, str], ...] = (
    ("SPY", "tracks the S&P 500, about 500 of the largest US companies."),
    ("QQQ", "tracks the Nasdaq-100, 100 large US companies, technology-heavy."),
    ("EFA", "large companies in developed markets outside North America "
            "(Europe, Japan, Australia)."),
    ("EEM", "large companies in emerging markets (China, India, Brazil and "
            "others)."),
    ("AGG", "a broad basket of investment-grade US bonds (government and "
            "corporate)."),
    ("IEF", "US government bonds maturing in 7 to 10 years."),
    ("GLD", "physical gold."),
    ("VNQ", "US real-estate companies (REITs)."),
)
FUND_ROLES = ("As broad roles: bond funds (AGG, IEF) tend to steady a "
              "portfolio; stock funds (SPY, QQQ, EFA, EEM) drive long-run "
              "growth; gold (GLD) and real estate (VNQ) behave differently "
              "from both, which helps diversification.")

# --------------------------------------------------------------------------- #
# 3.3 -- FAQ (question, answer), pasted verbatim
# --------------------------------------------------------------------------- #
FAQ: tuple[tuple[str, str], ...] = (
    ("Is this financial advice?",
     "No. This is a university project demonstration for educational purposes "
     "only. Nothing shown here is a recommendation to buy or sell anything."),
    ("Why only four dates?",
     "The demonstration covers four dates for which the full recommendation "
     "and explanation pipeline was prepared and checked in earlier stages of "
     "the project. Other dates would produce numbers without their "
     "accompanying explanation, so they are not offered."),
    ("Why can't I pick my own risk profile?",
     "Matching people to a suitable risk level is the point of the system. The "
     "five questions assess both willingness to take risk and the financial "
     "capacity to absorb it, and capacity can cap the result: wanting maximum "
     "growth does not by itself make it suitable. Letting users pick a profile "
     "directly would skip that safeguard."),
)

# --------------------------------------------------------------------------- #
# 3.4 -- question captions (define a word used in the question; never hint at
# what to answer).  Q2 and Q4 only -- none on Q1, Q3, Q5.
# --------------------------------------------------------------------------- #
Q2_CAPTION = "Portfolio: the whole collection of investments held together."
Q4_CAPTION = ("Emergency fund: savings you could live on for a few months if "
              "your income stopped.")

QUESTION_CAPTIONS: dict[str, str] = {"Q2": Q2_CAPTION, "Q4": Q4_CAPTION}


# --------------------------------------------------------------------------- #
# Render helpers (Task-15 chrome; add expanders/captions around frozen content)
# --------------------------------------------------------------------------- #
def render_not_advice() -> None:
    """2.1 -- rendered with the disclosure lines, both screens."""
    st.markdown(NOT_ADVICE)


def render_glossary() -> None:
    """2.2 -- collapsed glossary expander, both screens."""
    with st.expander("New to investing? Key terms", expanded=False):
        st.markdown("\n".join(f"- **{term}:** {definition}"
                              for term, definition in GLOSSARY))


def render_fund_holdings() -> None:
    """2.3 -- collapsed fund-holdings expander, near the weights."""
    with st.expander("What's inside these funds?", expanded=False):
        st.markdown("\n".join(f"- **{ticker}:** {desc}"
                              for ticker, desc in FUND_HOLDINGS))
        st.markdown(FUND_ROLES)


def render_faq() -> None:
    """2.4 -- collapsed FAQ expander, bottom of the main area, both screens."""
    with st.expander("Questions people often ask", expanded=False):
        st.markdown("\n\n".join(f"**{q}** {a}" for q, a in FAQ))
