"""Important question filtering utilities (ID + optional keyword based)."""

from typing import Dict, List, Set

IMPORTANT_ITEM_IDS: Set[str] = {
    "MEDICATION",
    "ADEPTCOMPLAINT",
    "ADETIMEEVENT",
    "DCTX",
    "DOSE",
    "HOSPITALIZATION",
    "RESOLVED",
    "CAUSALITY",
    "MDCONTACT",
    "SPOKEWITH",
    "DATE",
    "REPORT",
}

IMPORTANT_ITEM_KEYWORDS: Set[str] = set()  # global keyword set (optional)

# Per assessment type dynamic fragments (CASE-SENSITIVE substring match on linkId ONLY).
# Keys should align with questionnaireTitle or other identifiers.
DYNAMIC_FRAGMENT_MAP: Dict[str, Set[str]] = {
    "Telephonic Nurse AE PC Documentation": {
        "ADEPTCOMPLAINT",
        "HOSPITAL",
        "STOP",
        "DOSE",
        "RESOLVED",
        "CAUSAL",
        "REPORT",
    }
}


def get_important_item_ids(therapy_type: str | None = None) -> Set[str]:
    return IMPORTANT_ITEM_IDS


def is_item_important_by_keyword(text: str) -> bool:
    if not text:
        return False
    tl = text.lower()
    return any(k in tl for k in IMPORTANT_ITEM_KEYWORDS)


def filter_important_items(
    items: List[Dict],
    therapy_type: str | None = None,
    use_keywords: bool = True,
    assessment_title: str | None = None,
) -> List[Dict]:
    """Filter items by static IDs plus optional keywords and dynamic fragments.

    Args:
        items: list of question dicts
        therapy_type: optional therapy scoping (currently unused in id set)
        use_keywords: include keyword text matching
        assessment_title: used to pull dynamic fragments from DYNAMIC_FRAGMENT_MAP
    """
    ids = get_important_item_ids(therapy_type)
    fragments: Set[str] = set()
    if assessment_title and assessment_title in DYNAMIC_FRAGMENT_MAP:
        fragments = DYNAMIC_FRAGMENT_MAP[assessment_title]
    out: List[Dict] = []
    for it in items or []:
        lid = it.get("linkId", "")
        txt = it.get("text", "") or ""
        dynamic_hit = any(fr in lid for fr in fragments) if fragments else False
        if (
            lid in ids
            or dynamic_hit
            or (use_keywords and is_item_important_by_keyword(txt))
        ):
            out.append(it)
    return out


__all__ = ["filter_important_items", "get_important_item_ids", "DYNAMIC_FRAGMENT_MAP"]
