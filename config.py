# config.py
"""
Configuration file for AI Study Assistant
"""

import os
from dotenv import load_dotenv

load_dotenv()

# AI API Keys (Optional - for enhanced features)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# App Configuration
APP_CONFIG = {
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "supported_formats": [".pdf", ".txt", ".docx", ".md", ".jpg", ".png"],
    "default_subjects": [
        "General", "Mathematics", "Physics", "Chemistry", "Biology",
        "History", "Geography", "Literature", "Computer Science",
        "Economics", "Psychology", "Philosophy", "Languages"
    ],
    "study_modes": ["Learn New", "Review", "Test Yourself", "Spaced Repetition"],
    "summarization_methods": ["Transformers (BART)", "Extractive NLP", "Key Points", "Bullet Points"]
}

# UI Configuration
UI_CONFIG = {
    "theme": {
        "primary": "#667eea",
        "secondary": "#764ba2",
        "success": "#10B981",
        "warning": "#F59E0B",
        "error": "#EF4444"
    },
    "flashcard_size": (800, 500),
    "max_cards_per_session": 50
}