"""
RAG Governor
============
DICGovernor subclass that retrieves relevant policy context before the FMEA
stage and applies score adjustments derived from that context.

The 7-stage pipeline is otherwise identical to DICGovernor.  The only change
is that Stage 3 (FMEA) receives RAG-adjusted Severity and Detection deltas,
and the stage log carries an extra "rag_context" entry between stages 2 and 3.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..dic_llm.dic_governor import DICGovernor, DICDecision, RPN_THRESHOLD, SANDBOX_ROOT
from ..dic_llm.file_action  import FileAction, FileOp
from ..dic_llm.risk_fmea    import fmea_table, max_rpn, FMEAItem, occ_from_prob
from ..dic_llm.critical_path import reversibility_profile
from ..dic_llm.circuit_breaker import CircuitBreakerConfig

from .rag_retriever    import RAGRetriever, RetrievedChunk
from .rag_fmea_adapter import adapt, FMEAOverride, top_blocking_citations


class RAGGovernor(DICGovernor):
    """
    DICGovernor with RAG-augmented FMEA scoring.

    Before Stage 3 (FMEA), the governor:
    1. Converts the proposed action to a retrieval query
    2. Fetches the top-k most relevant policy chunks from Chroma
    3. Scans chunks for risk keywords → FMEAOverride (severity_delta, detection_delta)
    4. Applies the override to Severity and Detection in fmea_table()

    The stage_log gains an extra "rag_context" entry containing the
    retrieved chunks and the applied override for full auditability.

    Parameters
    ----------
    top_k : int
        Number of policy chunks to retrieve per action (default 3).
    sandbox_root : Path
        Passed through to DICGovernor.
    rpn_threshold : int
        Passed through to DICGovernor.
    circuit_breaker_config : CircuitBreakerConfig | None
        Passed through to DICGovernor.
    """

    def __init__(
        self,
        top_k:                  int  = 3,
        sandbox_root:           Path = SANDBOX_ROOT,
        rpn_threshold:          int  = RPN_THRESHOLD,
        circuit_breaker_config: CircuitBreakerConfig | None = None,
    ) -> None:
        super().__init__(
            sandbox_root=sandbox_root,
            rpn_threshold=rpn_threshold,
            circuit_breaker_config=circuit_breaker_config,
        )
        self._retriever = RAGRetriever(top_k=top_k)

    # ------------------------------------------------------------------ #
    #  Override evaluate() — inject RAG between Stage 2 and Stage 3       #
    # ------------------------------------------------------------------ #

    def evaluate(self, action: FileAction) -> DICDecision:
        self._step += 1
        stage_log: list = []

        # ── 1. Branching ─────────────────────────────────────────────── #
        scope_ok, scope_msg = self._scope_check(action)
        stage_log.append({"stage": "branching", "pass": scope_ok, "detail": scope_msg})

        if not scope_ok:
            return self._block(action, scope_msg, stage_log)

        # ── 2. Critical Path ──────────────────────────────────────────── #
        cp = reversibility_profile(action, self.sandbox_root)
        stage_log.append({
            "stage":          "critical_path",
            "reversibility":  cp.reversibility.value,
            "file_exists":    cp.file_exists,
            "p_irreversible": cp.p_irreversible,
            "side_effects":   cp.side_effects,
        })

        # ── 2b. RAG Context ───────────────────────────────────────────── #
        chunks: list[RetrievedChunk] = []
        override = FMEAOverride()

        if action.op != FileOp.DONE:
            chunks   = self._retriever.query(action)
            override = adapt(chunks)

        stage_log.append({
            "stage":    "rag_context",
            "chunks":   [
                {
                    "section": c.section,
                    "source":  c.source,
                    "score":   c.score,
                    "text_preview": c.text[:120].replace("\n", " "),
                }
                for c in chunks
            ],
            "override": {
                "severity_delta":  override.severity_delta,
                "detection_delta": override.detection_delta,
                "notes":           override.notes,
            },
            "citations": [
                {
                    "document":        c.document,
                    "section":         c.section,
                    "keyword":         c.keyword,
                    "quote":           c.quote,
                    "severity_delta":  c.severity_delta,
                    "detection_delta": c.detection_delta,
                }
                for c in override.citations
            ],
        })

        # ── 3. FMEA (with RAG override applied) ──────────────────────── #
        table  = _fmea_table_with_override(
            action, cp.file_exists, self.llm_risk_tracker.mean, override
        )
        m_rpn  = max_rpn(table)
        fmea_serial = {k: vars(v) for k, v in table.items()}
        stage_log.append({
            "stage":   "fmea",
            "table":   fmea_serial,
            "max_rpn": m_rpn,
        })

        # ── 4. Decision Gate ─────────────────────────────────────────── #
        gate_blocked = m_rpn >= self.rpn_threshold
        stage_log.append({
            "stage":     "decision_gate",
            "max_rpn":   m_rpn,
            "threshold": self.rpn_threshold,
            "blocked":   gate_blocked,
        })

        # ── 5. Circuit Breaker ────────────────────────────────────────── #
        cb = self.circuit_breaker.evaluate(m_rpn)
        stage_log.append({
            "stage":  "circuit_breaker",
            "state":  cb.state.value,
            "reason": cb.reason,
        })

        cb_blocked   = cb.state.value == "stop"
        cb_escalated = cb.state.value == "escalate"

        # ── 6. Utility ───────────────────────────────────────────────── #
        utility = self._utility(action, cp, m_rpn)
        stage_log.append({"stage": "utility", "score": utility})

        # ── 7. Belief Update ──────────────────────────────────────────── #
        risky = m_rpn >= self.rpn_threshold // 2
        self.llm_risk_tracker.update(risky)
        stage_log.append({
            "stage":         "belief_update",
            "risky_signal":  risky,
            "llm_risk_mean": self.llm_risk_tracker.mean,
            "beta":          repr(self.llm_risk_tracker),
        })

        # ── Final verdict ─────────────────────────────────────────────── #
        if cb_escalated:
            self.escalation_count += 1
            reason = (
                "ESCALATE: all actions exceed risk threshold, "
                "human confirmation required"
            )
            _attach_justification(stage_log, override, reason)
            return self._build(False, action, reason, cp, fmea_serial, m_rpn, cb, utility, stage_log)

        if gate_blocked or cb_blocked:
            reason = cb.reason if cb_blocked else f"RPN {m_rpn} ≥ threshold {self.rpn_threshold}"
            _attach_justification(stage_log, override, reason)
            return self._build(False, action, reason, cp, fmea_serial, m_rpn, cb, utility, stage_log)

        return self._build(True, action, None, cp, fmea_serial, m_rpn, cb, utility, stage_log)


# ── Block justification ───────────────────────────────────────────────────── #

def _doc_display_name(stem: str) -> str:
    """'file_ops_policy' → 'File Ops Policy'"""
    return stem.replace("_", " ").title()


def _attach_justification(stage_log: list, override: FMEAOverride, block_reason: str) -> None:
    """
    Build a human-readable block justification from the top-impact citations
    and append a 'block_justification' entry to the stage log.

    Format per citation:
        Blocked per <Document Name>, <Section>: "<quote>"
    """
    top = top_blocking_citations(override, n=2)
    lines: list[str] = []

    for cit in top:
        doc   = _doc_display_name(cit.document)
        sect  = cit.section.lstrip("#").strip()
        quote = cit.quote.strip()
        lines.append(f'Blocked per {doc}, {sect}: "{quote}"')

    if not lines:
        # No RAG citations available — fall back to the block reason alone
        lines.append(f"Blocked: {block_reason}")

    stage_log.append({
        "stage":              "block_justification",
        "lines":              lines,
        "primary_document":   top[0].document if top else None,
        "primary_section":    top[0].section  if top else None,
    })


# ── FMEA with RAG override ────────────────────────────────────────────────── #

def _clamp(value: int, lo: int = 1, hi: int = 10) -> int:
    return max(lo, min(hi, value))


def _fmea_table_with_override(
    action:        FileAction,
    file_exists:   bool,
    llm_risk_mean: float,
    override:      FMEAOverride,
) -> dict:
    """
    Build FMEA table identical to risk_fmea.fmea_table() but apply
    severity_delta and detection_delta from the RAG override.
    """
    o_base = occ_from_prob(llm_risk_mean)
    s_delta = override.severity_delta
    d_delta = override.detection_delta
    table: dict = {}

    if action.op == FileOp.DELETE:
        s, o, d, r = _clamp(10 + s_delta), min(10, o_base + 2), _clamp(2 + d_delta), 10
        table["permanent_data_loss"] = FMEAItem(
            "File permanently deleted — unrecoverable without backup",
            s, o, d, r, s * o * d * r,
        )
        s2, o2, d2, r2 = _clamp(9 + s_delta), o_base, _clamp(7 + d_delta), 10
        table["wrong_file_deleted"] = FMEAItem(
            "Wrong file targeted — path confusion or LLM hallucination",
            s2, o2, d2, r2, s2 * o2 * d2 * r2,
        )

    elif action.op == FileOp.WRITE:
        if file_exists:
            s, o, d, r = _clamp(8 + s_delta), o_base, _clamp(3 + d_delta), 8
            table["overwrite_data_loss"] = FMEAItem(
                "Existing file overwritten — original content lost",
                s, o, d, r, s * o * d * r,
            )
        s2, o2, d2, r2 = _clamp(4 + s_delta), max(1, o_base - 2), _clamp(7 + d_delta), 3
        table["unintended_write"] = FMEAItem(
            "File created/modified with incorrect content",
            s2, o2, d2, r2, s2 * o2 * d2 * r2,
        )

    elif action.op == FileOp.READ:
        s, o, d, r = _clamp(5 + s_delta), max(1, o_base - 3), _clamp(5 + d_delta), 1
        table["sensitive_read"] = FMEAItem(
            "Sensitive file contents exposed to LLM context",
            s, o, d, r, s * o * d * r,
        )

    return table
