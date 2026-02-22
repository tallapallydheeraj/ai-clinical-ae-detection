"""Minimal assessment matching rules.

Responsibilities now limited to deciding IF an assessment should be processed.
Question-level filtering fragments moved to dynamic question filtering logic.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

ASSESSMENT_RULES: List[Dict[str, Any]] = [{}]


def _get_nested(assessment: Dict[str, Any], path: str) -> Any:
    parts = path.split(".")
    cur: Any = assessment
    for p in parts:
        idx = None
        if "]" in p:
            name, rest = p.split("[")
            idx = int(rest.rstrip("]"))
            p = name
        if isinstance(cur, dict):
            cur = cur.get(p)
        else:
            return None
        if idx is not None:
            if isinstance(cur, list) and 0 <= idx < len(cur):
                cur = cur[idx]
            else:
                return None
    return cur


def _rule_matches(rule: Dict[str, Any], assessment: Dict[str, Any]) -> bool:
    match = rule.get("match", {})
    for key, expected in match.items():
        val = _get_nested(assessment, key)
        if isinstance(expected, list):
            if val not in expected:
                return False
        else:
            if val != expected:
                return False
    return True


def match_assessment(assessment: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for r in ASSESSMENT_RULES:
        if _rule_matches(r, assessment):
            return r
    return None


def should_process_assessment(assessment: Dict[str, Any]) -> bool:
    return match_assessment(assessment) is not None


__all__ = [
    "should_process_assessment",
    "ASSESSMENT_RULES",
    "match_assessment",
]
