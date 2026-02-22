"""
FHIR utility functions for clinical assessment processing.

FHIR (Fast Healthcare Interoperability Resources) is a standard for exchanging healthcare information electronically.
This module provides helpers to load, filter, and summarize FHIR-based assessment data, including:
 - Loading and parsing FHIR assessment JSON files
 - Extracting therapy type and flattening answers
 - Filtering important questions using assessment rules and dynamic filters
 - Summarizing assessments for downstream processing or LLM prompts
"""

from typing import Any, Dict, List, Optional
import json
from .assessment_rules import match_assessment


def load_assessment(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_therapy_type(assessment: Dict[str, Any]) -> Optional[str]:
    part = assessment.get("partOf", {})
    for ident in part.get("identifier", []):
        if ident.get("assigner") == "therapyType":
            return str(ident.get("value")).strip()
    return None


def flatten_answers(items: List[Dict[str, Any]]) -> List[str]:
    lines = []
    for node in items or []:
        q = node.get("text") or node.get("linkId") or ""
        ans = node.get("answer", [])
        vals = []
        for a in ans:
            if "valueString" in a:
                vals.append(str(a["valueString"]))
            elif "valueDecimal" in a:
                vals.append(str(a["valueDecimal"]))
            elif "valueCoding" in a and isinstance(a["valueCoding"], dict):
                lab = a["valueCoding"].get("display") or a["valueCoding"].get("code")
                if lab is not None:
                    vals.append(str(lab))
        if q or vals:
            lines.append(f"{q}: {'; '.join(vals) if vals else ''}".strip())
    return lines


def filter_important_items(
    assessment: Dict[str, Any],
    therapy_type: Optional[str] = None,
    use_keywords: bool = True,
) -> List[Dict[str, Any]]:
    """
    Filter assessment items to include only important questions.

    Args:
        assessment: The assessment dictionary
        therapy_type: The therapy type for specialized filtering
        use_keywords: Whether to also filter by keywords in question text

    Returns:
        Filtered list of important assessment items
    """
    from .question_filter import filter_important_items as _filter_items

    items = assessment.get("item", [])
    if not therapy_type:
        therapy_type = extract_therapy_type(assessment)

    return _filter_items(
        items,
        therapy_type,
        use_keywords,
        assessment_title=assessment.get("questionnaireTitle"),
    )


def summarize_assessment(
    assessment: Dict[str, Any],
    filter_important: bool = False,
    therapy_type: Optional[str] = None,
) -> str:
    """
    Create a summary of the assessment.

    Args:
        assessment: The assessment dictionary
        filter_important: Whether to filter to only important questions
        therapy_type: The therapy type (if not provided, will be extracted)

    Returns:
        Assessment summary string
    """
    title = assessment.get("questionnaireTitle", "")
    status = assessment.get("status", "")
    idv = assessment.get("id", "")
    who = assessment.get("source", {}).get("reference", "Patient")

    items = assessment.get("item", [])
    matched_rule = match_assessment(assessment)
    if filter_important:
        if not therapy_type:
            therapy_type = extract_therapy_type(assessment)
        items = filter_important_items(assessment, therapy_type, use_keywords=True)
    lines = flatten_answers(items)
    body = "\n".join(lines)
    filter_note = " (filtered to important questions)" if filter_important else ""
    rule_note = " (rule matched)" if matched_rule else " (no rule matched)"
    return f"Assessment {idv} titled '{title}' status {status} about {who}{filter_note}{rule_note}. Key answers:\n{body}"
