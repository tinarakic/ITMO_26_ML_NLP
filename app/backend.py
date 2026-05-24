import requests

MODEL_SERVER_URL = "http://model-server:8000/generate"


AVAILABLE_MODELS = [
    "Qwen2.5-1.5B"
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
Translate the following Russian verses into {target_language}.

Requirements:
- Preserve the Russian sentence structure as closely as possible.
- Keep the poetic style and rhythm.
- Preserve imagery and symbolism.
- Prefer poetic diction over literal prose.
- Do NOT summarize.
- Do NOT explain.
- Output ONLY the translated verses.
- Do not lose any of the original verse parts.
- Preserve EXACT line breaks.
- Each input line must correspond to one output line.
- Do NOT merge or split lines.
- Do NOT add or remove lines.

Here's the Russian verses:
{poem}
"""


def extract_translation(response_text: str, prompt: str) -> str:
    cleaned = response_text.strip()
    prompt_clean = prompt.strip()

    if cleaned.startswith(prompt_clean):
        cleaned = cleaned[len(prompt_clean):].strip()

    return cleaned


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

    return extract_translation(data["text"], prompt)