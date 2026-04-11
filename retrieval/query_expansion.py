"""
retrieval/query_expansion.py — Expand user query into legal terminology

Two-stage expansion:
  1. Rule-based synonym map (fast, offline)
  2. Gemini LLM expansion for complex queries
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

LEGAL_SYNONYMS: dict[str, list[str]] = {
    # Consumer rights
    "refund": ["refund", "reimbursement", "compensation", "return of price", "Section 39(1)(b)"],
    "replacement": ["replacement", "exchange", "defect-free goods", "Section 39(1)(a)"],
    "compensation": ["compensation", "damages", "loss", "injury", "Section 39(1)(c)"],
    "complaint": ["complaint", "grievance", "dispute", "consumer dispute", "Section 35"],
    "defect": ["defect", "defective goods", "manufacturing defect", "product defect", "Section 83"],
    "deficiency": ["deficiency", "deficiency in service", "service deficiency", "Section 2(11)"],
    "unfair trade practice": ["unfair trade practice", "deceptive practice", "misleading advertisement", "Section 2(47)"],
    "consumer": ["consumer", "buyer", "purchaser", "end user", "Section 2(7)"],
    "redressal": ["redressal", "forum", "commission", "district commission", "consumer court"],
    "product liability": ["product liability", "manufacturer liability", "Chapter VI", "Section 82"],
    "warranty": ["warranty", "guarantee", "express warranty", "Section 84"],
    # Jurisdiction
    "district forum": ["district commission", "district consumer disputes redressal commission", "Section 28"],
    "state commission": ["state consumer disputes redressal commission", "Section 42"],
    "national commission": ["national consumer disputes redressal commission", "Section 53"],
    # Processes
    "appeal": ["appeal", "revision", "Section 41", "Section 51"],
    "mediation": ["mediation", "settlement", "Section 74", "Section 75"],
    "penalty": ["penalty", "punishment", "fine", "imprisonment", "Section 88", "Section 89"],
    # Food Safety
    "food": ["food safety", "FSSAI", "Food Safety and Standards Act 2006", "adulteration"],
    "adulteration": ["adulteration", "adulterated food", "food standard", "Section 89"],
    # Telecom
    "billing": ["billing dispute", "tariff", "overcharging", "TRAI Consumer Regulations"],
    "service quality": ["service quality", "QoS", "TRAI standards", "broadband speed"],
}


def rule_based_expansion(query: str) -> str:
    """Append relevant legal terms to the query based on keyword matching."""
    query_lower = query.lower()
    expansions = set()
    for keyword, synonyms in LEGAL_SYNONYMS.items():
        if keyword in query_lower:
            expansions.update(synonyms)

    if expansions:
        expansion_str = " ".join(expansions)
        return f"{query} {expansion_str}"
    return query


async def llm_expand_query(query: str) -> str:
    """
    Use Gemini to expand a complex legal query into richer search terms.
    Returns the expanded string (original query + legal terms).
    """
    try:
        from llm.gemini_client import generate_text
        from llm.prompts import QUERY_EXPANSION_PROMPT

        prompt = QUERY_EXPANSION_PROMPT.format(query=query)
        expanded = await generate_text(prompt, max_tokens=150)
        # Combine original + LLM expansion
        return f"{query} {expanded.strip()}"
    except Exception:
        # Fallback to rule-based if LLM fails
        return rule_based_expansion(query)


async def expand_query(query: str, use_llm: bool = True) -> dict:
    """
    Full query expansion pipeline.
    Returns: {original, rule_expanded, final_expanded}
    """
    rule_expanded = rule_based_expansion(query)
    if use_llm:
        final = await llm_expand_query(rule_expanded)
    else:
        final = rule_expanded

    return {
        "original": query,
        "rule_expanded": rule_expanded,
        "final_expanded": final,
    }
