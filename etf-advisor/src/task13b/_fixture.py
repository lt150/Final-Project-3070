"""Gate A4 fixture -- SYNTHETIC TEXT ONLY.

Prints FRE/FKGL for the three registered synthetic fixtures, one metric per
line, at full float precision.  This module is invoked twice by stage_a.py in
two SEPARATE processes; identical stdout is the tool-determinism evidence.  It
reads NO transcript and computes nothing on any frozen artifact.
"""

from __future__ import annotations

import textstat

from . import FIXTURE_COMPLEX, FIXTURE_PARAGRAPH, FIXTURE_SIMPLE, machinery_versions

# Registered metric: language en_US = library default.
_DEFAULT_LANG = getattr(textstat.textstat, "_textstatistics__lang", "<unavailable>")


def _emit() -> None:
    mv = machinery_versions()
    print(f"PYTHON_VERSION\t{mv['python']}")
    print(f"TEXTSTAT_VERSION\t{mv['textstat']}")
    print(f"PYPHEN_VERSION\t{mv['pyphen']}")
    print(f"NLTK_VERSION\t{mv['nltk']}")
    print(f"DEFAULT_LANG\t{_DEFAULT_LANG}")
    print(f"PARAGRAPH_FRE\t{textstat.flesch_reading_ease(FIXTURE_PARAGRAPH)!r}")
    print(f"PARAGRAPH_FKGL\t{textstat.flesch_kincaid_grade(FIXTURE_PARAGRAPH)!r}")
    print(f"SIMPLE_FRE\t{textstat.flesch_reading_ease(FIXTURE_SIMPLE)!r}")
    print(f"SIMPLE_FKGL\t{textstat.flesch_kincaid_grade(FIXTURE_SIMPLE)!r}")
    print(f"COMPLEX_FRE\t{textstat.flesch_reading_ease(FIXTURE_COMPLEX)!r}")
    print(f"COMPLEX_FKGL\t{textstat.flesch_kincaid_grade(FIXTURE_COMPLEX)!r}")


if __name__ == "__main__":
    _emit()
