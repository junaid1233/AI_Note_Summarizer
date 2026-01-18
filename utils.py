# utils.py
"""
Utility functions for AI Study Assistant
"""

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List
import streamlit as st


def generate_card_id(content: str) -> str:
    """Generate unique ID for flashcard"""
    return hashlib.md5(content.encode()).hexdigest()[:8]


def format_time(minutes: int) -> str:
    """Format minutes to human-readable time"""
    if minutes < 60:
        return f"{minutes} min"
    else:
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours}h {mins}m"


def calculate_mastery_score(card: Dict[str, Any]) -> float:
    """Calculate mastery score for a card"""
    base_score = 1.0 if card.get("mastered", False) else 0.0
    review_factor = min(card.get("review_count", 0) * 0.1, 0.5)
    return base_score + review_factor


def export_to_csv(flashcards: List[Dict[str, Any]], filename: str = "flashcards.csv"):
    """Export flashcards to CSV"""
    import pandas as pd

    data = []
    for card in flashcards:
        data.append({
            "Question": card["question"],
            "Answer": card["answer"],
            "Subject": card["subject"],
            "Mastered": "Yes" if card.get("mastered", False) else "No",
            "Review Count": card.get("review_count", 0),
            "Created": card["created"],
            "Last Reviewed": card.get("last_reviewed", "")
        })

    df = pd.DataFrame(data)
    return df.to_csv(index=False)


def save_session_data():
    """Save session data to file"""
    data = {
        "flashcards": st.session_state.get("flashcards", []),
        "summaries": st.session_state.get("summaries", []),
        "last_saved": datetime.now().isoformat()
    }

    with open("study_session_backup.json", "w") as f:
        json.dump(data, f, indent=2)


def load_session_data():
    """Load session data from file"""
    try:
        with open("study_session_backup.json", "r") as f:
            data = json.load(f)
            st.session_state.flashcards = data.get("flashcards", [])
            st.session_state.summaries = data.get("summaries", [])
    except FileNotFoundError:
        pass