"""Parse citation strings back into structured {instrument, article, paragraph, point}.


Unknown fields resolve to None / "" — the downstream metrics tolerate this.
"""

from __future__ import annotations

import re
from typing import Optional

_INSTRUMENT_ALIASES = {
    "mifid ii": "MIFID2_L1",
    "mifid 2": "MIFID2_L1",
    "mifid-ii": "MIFID2_L1",
    "mifid": "MIFID2_L1",
    "sfdr": "SFDR_L1",
    # CELEX numbers — match the bare digits+letter form that appears both
    # in filenames ("celex_32014l0065_en_txt.pdf") and in explicit
    # "(CELEX 32014L0065)" suffixes. Lowercase because the head is lowered.
    "32014l0065": "MIFID2_L1",
    "32019r2088": "SFDR_L1",
}

_ARTICLE_RE = re.compile(r"Art\.\s*(\d+[A-Za-z]?)", re.IGNORECASE)
_PARA_RE = re.compile(r"para\.?\s*(\d+[a-z]?)", re.IGNORECASE)
_POINT_RE = re.compile(r"\(([a-z])\)")
_CELEX_RE = re.compile(r"CELEX\s+([0-9A-Z]+)", re.IGNORECASE)


def parse_citation(s: str) -> dict:
    """Return {instrument, article, paragraph, point, celex} for a citation string."""
    instrument = ""
    head = s.split(",", 1)[0].strip().lower().strip("[]")
    whole = s.lower()
    for alias, inst_id in _INSTRUMENT_ALIASES.items():
        if alias in head or alias in whole:
            instrument = inst_id
            break

    article = ""
    m = _ARTICLE_RE.search(s)
    if m:
        article = m.group(1)

    paragraph: Optional[str] = None
    m = _PARA_RE.search(s)
    if m:
        paragraph = m.group(1)

    point: Optional[str] = None
    m = _POINT_RE.search(s)
    if m:
        point = m.group(1)

    celex = ""
    m = _CELEX_RE.search(s)
    if m:
        celex = m.group(1)

    return {
        "instrument": instrument,
        "article": article,
        "paragraph": paragraph,
        "point": point,
        "celex": celex,
    }
