from openai import OpenAI


MODEL_ENDPOINTS = {
    "Qwen 7B": {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "base_url": "http://vllm-qwen:8000/v1",
    },
    "Mistral 7B": {
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "base_url": "http://vllm-mistral:8000/v1",
    },
}


AVAILABLE_MODELS = list(MODEL_ENDPOINTS.keys())


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


def get_client(base_url: str):
    return OpenAI(
        base_url=base_url,
        api_key="EMPTY",
    )


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
- do not invent new metaphors;
- preserve line breaks;
- return only the translated poem.

Poem:
{poem}
"""


def extract_message_text(response) -> str:
    message = response.choices[0].message

    if message.content:
        return message.content.strip()

    return "Model returned empty response."


def translate_poem(poem: str, target_language: str, model: str) -> str:
    if model not in MODEL_ENDPOINTS:
        raise ValueError("Unknown model.")

    model_config = MODEL_ENDPOINTS[model]

    client = get_client(model_config["base_url"])

    prompt = build_translation_prompt(poem, target_language)

    response = client.chat.completions.create(
        model=model_config["model"],
        messages=[
            {
                "role": "system",
                "content": "You are a professional literary translator.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.8,
        max_tokens=1000,
    )

    return extract_message_text(response)