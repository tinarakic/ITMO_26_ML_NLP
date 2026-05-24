import requests
import re

MODEL_SERVER_URL = "http://model-server:8000/generate"


AVAILABLE_MODELS = [
    "Qwen2.5-1.5B-Instruct"
]

LANGUAGES = [
    "English",
    "French",
    "German",
    "Spanish",
    "Italian",
    "Chinese",
    "Japanese",
    "Korean",
]


def build_translation_prompt(poem: str, target_language: str) -> str:
   return f"""
Translate the Russian poem into {target_language}.

Requirements:
- Preserve the Russian sentence structure as closely as possible.
- Keep the poetic style and rhythm.
- Preserve imagery and symbolism.
- Prefer poetic diction over literal prose.
- Do NOT summarize NOR explain.
- Output ONLY the translated verses.
- Preserve EXACT line breaks.
- Each input line must correspond to one output line.

Here's the Russian verses:
{poem}
"""

def extract_translation(response_text: str) -> str:
    """
    Normalize model output and force proper line breaks.
    """

    text = response_text.strip()

    # normalize Windows/mac newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # split into lines
    lines = [line.strip() for line in text.split("\n")]

    # remove empty lines
    lines = [line for line in lines if line]

    # join back with proper line breaks
    return "\n".join(lines)


def translate_poem(poem: str, target_language: str, model: str = "qwen") -> str:
    if not poem.strip():
        raise ValueError("Poem is empty.")

    prompt = build_translation_prompt(poem, target_language)

    try:
        response = requests.post(
            MODEL_SERVER_URL,
            json={"prompt": prompt},
            timeout=180
        )
    except Exception as e:
        raise RuntimeError(f"Request failed: {str(e)}")

    if response.status_code != 200:
        raise RuntimeError(f"Model server error: {response.text}")

    data = response.json()

    if "text" not in data:
        raise RuntimeError(f"Invalid response format: {data}")

    return extract_translation(data["text"])
