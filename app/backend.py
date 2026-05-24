import requests

MODEL_SERVER_URL = "http://model-server:8000/generate"


AVAILABLE_MODELS = [
    "qwen",
    "default"
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
You are a professional literary translator.

Translate the Russian poem into {target_language}.

Rules:
- preserve the original meaning as accurately as possible;
- preserve emotional tone and poetic imagery;
- make the result sound natural and poetic in {target_language};
- do not add a title;
- do not add explanations;
- do not invent new metaphors that are not present in the original;
- preserve line breaks as closely as possible;
- return only the translated poem.

Important:
The translation should be poetic, but semantic accuracy is more important than rhyme.

Poem:
{poem}
"""


def extract_translation(response_text: str) -> str:
    """
    Model server returns full text — we return it directly.
    """
    return response_text.strip()


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