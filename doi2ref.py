#!/usr/bin/env python3
"""
doi2ref.py

Turn a DOI into a compact reference string like:
"C.-P. Hsu et al., Nat Commun 13, 2022"

Uses Crossref's REST API (no API key required).
"""

from __future__ import annotations

import re
import sys
from typing import Any, Dict, Optional

import requests



CROSSREF_WORKS_URL = "https://api.crossref.org/works/{}"

# Optional: add your own journal abbreviations here (Crossref often returns full titles)
JOURNAL_ABBREV = {
    "Nature Communications": "Nat Commun",
    "Nature": "Nature",
    "Science": "Science",
    "Proceedings of the National Academy of Sciences": "PNAS",
    "Physical Review Letters": "Phys Rev Lett",
    "Physical Review B": "Phys Rev B",
}


def _clean_doi(doi: str) -> str:
    doi = doi.strip()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
    return doi


def _get_crossref_work(doi: str, timeout_s: int = 20) -> Dict[str, Any]:
    url = CROSSREF_WORKS_URL.format(requests.utils.quote(doi))
    headers = {
        # Polite User-Agent; set this to something identifiable if you publish/ship it.
        "User-Agent": "doi2ref/1.0 (mailto:your.email@example.com)"
    }
    r = requests.get(url, headers=headers, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    if "message" not in data:
        raise ValueError("Unexpected Crossref response: missing 'message'")
    return data["message"]


def _initials_from_given(given: str) -> str:
    """
    Convert given names to initials.
    - Hyphenated parts keep hyphens: "Chun-Peng" -> "C.-P."
    - Multiple names: "John Ronald" -> "J. R."
    """
    given = (given or "").strip()
    if not given:
        return ""

    parts = re.split(r"\s+", given)

    out_parts = []
    for part in parts:
        subparts = part.split("-")
        sub_inits = []
        for sp in subparts:
            sp = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ]", "", sp)  # keep letters
            if sp:
                sub_inits.append(sp[0].upper() + ".")
        if sub_inits:
            # If hyphenated: "C." + "-"+ "P." => "C.-P."
            out_parts.append("-".join(sub_inits))

    return " ".join(out_parts).strip()


def _format_author(author: Dict[str, Any]) -> str:
    given = author.get("given", "") or ""
    family = author.get("family", "") or ""
    initials = _initials_from_given(given)
    if initials and family:
        return f"{initials} {family}"
    return family or initials or ""


def _get_year(work: Dict[str, Any]) -> Optional[int]:
    # Crossref typically provides year in issued -> date-parts
    for key in ("issued", "published-print", "published-online", "created"):
        obj = work.get(key)
        if obj and "date-parts" in obj and obj["date-parts"]:
            dp = obj["date-parts"][0]
            if dp and isinstance(dp[0], int):
                return dp[0]
    return None


def _get_journal_title(work: Dict[str, Any]) -> str:
    # Prefer Crossref's short title if present
    short = work.get("short-container-title") or []
    if short:
        title = short[0]
    else:
        cont = work.get("container-title") or []
        title = cont[0] if cont else ""

    # Apply local abbreviation mapping if available
    return JOURNAL_ABBREV.get(title, title)


def doi_to_compact_reference(doi: str) -> str:
    """
    Returns a string like:
    "C.-P. Hsu et al., Nat Commun 13, 2022"
    """
    doi = _clean_doi(doi)
    work = _get_crossref_work(doi)

    authors = work.get("author") or []
    first_author = ""
    if authors:
        # Using tqdm in the loop (per your preference). This is fast, but keeps consistent style.
        formatted = []
        for a in authors:
            formatted.append(_format_author(a))
        formatted = [x for x in formatted if x]

        if formatted:
            first_author = formatted[0]

    author_str = first_author if first_author else "Unknown"
    if len(authors) > 1:
        author_str += " et al."

    journal = _get_journal_title(work) or "Unknown journal"
    volume = work.get("volume") or ""
    year = _get_year(work) or ""

    vol_part = f" {volume}" if volume else ""
    year_part = f"{year}" if year else ""

    # Final formatting
    if year_part:
        return f"{author_str}, {journal}{vol_part}, {year_part}"
    else:
        return f"{author_str}, {journal}{vol_part}"


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: doi2ref.py <DOI or doi.org URL>", file=sys.stderr)
        return 2
    doi = argv[1]
    try:
        ref = doi_to_compact_reference(doi)
        print(ref)
        return 0
    except requests.HTTPError as e:
        print(f"HTTP error while looking up DOI: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
