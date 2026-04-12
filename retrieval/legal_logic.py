"""
retrieval/legal_logic.py — Rule-based legal logic for Indian Consumer Law
"""

from typing import Dict, Any, List
import re
from datetime import datetime

def calculate_jurisdiction(claim_value_inr: float) -> Dict[str, Any]:
    """
    Returns the appropriate consumer commission based on pecuniary jurisdiction (2019 Act).
    
    Limits:
    - District Commission: Up to ₹50 Lakh.
    - State Commission: Above ₹50 Lakh to ₹2 Crore.
    - National Commission (NCDRC): Above ₹2 Crore.
    """
    
    if claim_value_inr <= 50_00_000:
        return {
            "forum": "District Consumer Disputes Redressal Commission",
            "level": "District",
            "limit_info": "Up to ₹50 Lakh"
        }
    elif claim_value_inr <= 2_00_00_000:
        return {
            "forum": "State Consumer Disputes Redressal Commission",
            "level": "State",
            "limit_info": "₹50 Lakh to ₹2 Crore"
        }
    else:
        return {
            "forum": "National Consumer Disputes Redressal Commission (NCDRC)",
            "level": "National",
            "limit_info": "Above ₹2 Crore"
        }

def check_limitation_status(dates_found: List[str] = None) -> str:
    """
    Provides a standardized advisory on the 2-year limitation period.
    """
    return (
        "Under Section 69 of the Consumer Protection Act, 2019, a complaint must be filed "
        "within **two years** from the date on which the cause of action has arisen. "
        "If your incident occurred more than 2 years ago, you may need to file a "
        "condonation of delay application with sufficient cause."
    )

def get_legal_check_context(claim_value: float = 0, language: str = "English") -> str:
    """
    Returns a formatted string to be injected into the LLM prompt.
    """
    juris = calculate_jurisdiction(claim_value)
    
    context = f"\n[SYSTEM LEGAL CALCULATION]\n"
    context += f"- Calculated Jurisdiction: {juris['forum']} ({juris['level']} Level)\n"
    context += f"- Pecuniary Limit: {juris['limit_info']}\n"
    context += f"- Claim Value Provided: ₹{claim_value:,.2f}\n"
    context += f"- Limitation Period: 2 Years from cause of action.\n"
    context += f"- Output Language: {language}\n"
    
    return context
