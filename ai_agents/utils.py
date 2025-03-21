"""Utility functions to weather agent"""
import os
from dotenv import load_dotenv

load_dotenv()

def get_api_key() -> str:
    """Get OpenAI API key from environment variables.

    Returns:
        str: The API key.

    Raises:
        ValueError: If API key is not configured.

    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not configured.")
    return api_key