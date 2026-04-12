"""
RAG FMEA Adapter
================
Translates retrieved policy chunks into FMEA score adjustments.

The adapter scans chunk text for known risk keywords and returns a
FMEAOverride that the RAGGovernor applies before calling fmea_table().
Each matched rule is traced back to the specific chunk and sentence that
triggered it, enabling human-readable block justifications.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from .rag_retriever import RetrievedChunk


@dataclass
class PolicyCitation:
    """
    Traceability record: which policy document/section triggered a rule.

    document : str   policy document stem, e.g. "file_ops_policy"
    section  : str   section heading from the document
    keyword  : str   the keyword that matched
    quote    : str   the sentence containing the keyword (≤ 140 chars)
    severity_delta  : int  contribution of this citation to S adjustment
    detection_delta : int  contribution of this citation to D adjustment
    """
    document:        str
    section:         str
    keyword:         str
    quote:           str
    severity_delta:  int
    detection_delta: int


@dataclass
class FMEAOverride:
    """
    Adjustments to FMEA inputs derived from retrieved policy context.

    severity_delta  : int              added to base Severity score (clipped 1-10)
    detection_delta : int              added to base Detection score (clipped 1-10)
    notes           : list[str]        human-readable note for each matched rule
    citations       : list[PolicyCitation]  per-rule traceability records
    """
    severity_delta:  int                    = 0
    detection_delta: int                    = 0
    notes:           list[str]              = field(default_factory=list)
    citations:       list[PolicyCitation]   = field(default_factory=list)


# ── Keyword → adjustment map ─────────────────────────────────────────────── #
#
# Each entry: (keyword_in_chunk_text, severity_delta, detection_delta, note)
# Matches are case-insensitive.  The first matching rule per keyword class wins.
#
_RULES: list[tuple[str, int, int, str]] = [
    # Severity escalations
    ("always critical",        +2,  0, "Policy: operation is always CRITICAL (+2 severity)"),
    ("critical",               +1,  0, "Policy: critical risk flag in matching section (+1 severity)"),
    ("prohibited",             +2,  0, "Policy: operation is prohibited without confirmation (+2 severity)"),
    ("escalate to human",      +2,  0, "Policy: human escalation required (+2 severity)"),
    ("escalate",               +1,  0, "Policy: escalation flag in matching section (+1 severity)"),
    ("irreversible",           +1,  0, "Policy: operation is irreversible (+1 severity)"),
    ("permanently lost",       +2,  0, "Policy: data permanently lost if no backup (+2 severity)"),
    ("backup required",        +1,  0, "Policy: backup required before operation (+1 severity)"),
    ("production",             +1,  0, "Policy: production data in scope (+1 severity)"),
    ("pii",                    +1,  0, "Policy: PII exposure risk (+1 severity)"),

    # Detection escalations (hard to detect = higher D score)
    ("hard to detect",          0, +2, "Policy: operation is hard to detect (+2 detection)"),
    ("audit",                   0, +1, "Policy: audit requirement increases detection difficulty (+1 detection)"),
    ("silently",                0, +2, "Policy: agents can modify silently (+2 detection)"),
    ("no audit log",            0, +2, "Policy: no audit log — post-incident detection impossible (+2 detection)"),
    ("autocommit",              0, +2, "Policy: autocommit mode — no rollback (+2 detection)"),

    # Severity reductions (policy confirms low risk)
    ("low-risk",               -1,  0, "Policy: operation confirmed low-risk (−1 severity)"),
    ("no-op",                  -2,  0, "Policy: operation is a no-op (−2 severity)"),
    ("zero risk",              -2,  0, "Policy: policy confirms zero risk (−2 severity)"),
    ("aggregate only",         -1,  0, "Policy: aggregate-only query reduces risk (−1 severity)"),
]


def _extract_quote(text: str, keyword: str, max_len: int = 140) -> str:
    """
    Return the first prose sentence in *text* that contains *keyword*
    (case-insensitive), truncated to *max_len* characters.

    Markdown heading lines (starting with #) and table rows (starting with |)
    are skipped so the quote always comes from policy body text.
    """
    sentences = re.split(r"(?<=[.!?])\s+|\n", text)
    for sent in sentences:
        stripped = sent.strip()
        # Skip headings, table rows, empty lines
        if not stripped or stripped.startswith("#") or stripped.startswith("|"):
            continue
        if keyword in stripped.lower():
            return stripped[:max_len] + ("…" if len(stripped) > max_len else "")
    # Fallback: first non-heading, non-empty line
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("|"):
            return line[:max_len] + ("…" if len(line) > max_len else "")
    return text[:max_len]


def adapt(chunks: list[RetrievedChunk]) -> FMEAOverride:
    """
    Scan retrieved chunks for policy keywords and build a FMEAOverride.

    Rules are applied once per keyword class across all chunks (no double
    counting).  For each matched rule the chunk that contained the keyword
    is recorded as a PolicyCitation for downstream justification rendering.

    Parameters
    ----------
    chunks : list[RetrievedChunk]

    Returns
    -------
    FMEAOverride
    """
    override = FMEAOverride()
    matched_keywords: set[str] = set()

    for keyword, s_delta, d_delta, note in _RULES:
        if keyword in matched_keywords:
            continue

        # Find the highest-scoring chunk that contains this keyword
        source_chunk: Optional[RetrievedChunk] = None
        for chunk in chunks:
            if keyword in chunk.text.lower():
                # chunks are already ordered by descending score
                source_chunk = chunk
                break

        if source_chunk is None:
            continue

        override.severity_delta  += s_delta
        override.detection_delta += d_delta
        override.notes.append(note)
        matched_keywords.add(keyword)

        quote = _extract_quote(source_chunk.text, keyword)
        override.citations.append(PolicyCitation(
            document=source_chunk.source,
            section=source_chunk.section,
            keyword=keyword,
            quote=quote,
            severity_delta=s_delta,
            detection_delta=d_delta,
        ))

    return override


def top_blocking_citations(override: FMEAOverride, n: int = 2) -> list[PolicyCitation]:
    """
    Return up to *n* citations with the highest combined severity impact,
    deduplicated by (document, section) so each justification line comes
    from a distinct policy section.
    """
    ranked = sorted(
        override.citations,
        key=lambda c: (c.severity_delta, c.detection_delta),
        reverse=True,
    )
    seen: set[tuple[str, str]] = set()
    result: list[PolicyCitation] = []
    for cit in ranked:
        key = (cit.document, cit.section)
        if key not in seen:
            seen.add(key)
            result.append(cit)
        if len(result) >= n:
            break
    return result
