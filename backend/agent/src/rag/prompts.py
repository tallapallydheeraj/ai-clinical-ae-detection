"""Unified prompt configuration for AE decision workflow.

All system/user prompts and heuristic hint text consolidated here to simplify maintenance.
"""

BASE_DECISION_HINT = (
    "Focus on whether the assessment implies any AE or Safety Information per the contract. "
    "Common triggers include overdose, abuse, withdrawal, lack of effect, product exposure during pregnancy, "
    "medication errors, off-label use with AE, suspected infection transmission, occupational or accidental exposure, "
    "therapy stopped or changed due to side effects, new hospitalization, allergic reactions, "
    "drug-drug interactions, or lack of therapeutic effect. "
    "If no explicit signal is present, return False."
)

DECISION_SYSTEM_PROMPT = (
    "You are a pharmacovigilance assistant deciding whether clinical assessment content indicates an Adverse Event per contract criteria. "
    "Follow regulatory seriousness principles and be conservative when uncertain. Return ONLY valid JSON following the required Decision schema."
)

DECISION_USER_PROMPT_TEMPLATE = (
    "Therapy: {therapy}\n"
    "Assessment summary:\n{summary}\n\n"
    "Contract excerpts:\n{contract}\n\n"
    "Respond JSON with keys: is_adverse_event (bool), matched_criteria (list[str]), explanation (str), reasoning (str), recommended_next_steps (list[str]), confidence (0-1)."
)

__all__ = [
    "BASE_DECISION_HINT",
    "DECISION_SYSTEM_PROMPT",
    "DECISION_USER_PROMPT_TEMPLATE",
]
