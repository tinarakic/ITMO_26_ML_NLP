import base64
from pathlib import Path

import streamlit as st

from backend import (
    AVAILABLE_MODELS,
    LANGUAGES,
    build_translation_prompt,
    translate_poem,
)


# =========================
# Page config
# =========================

st.set_page_config(
    page_title="AI Poetry Translator",
    page_icon="📜",
    layout="centered",
)


# =========================
# Styles
# =========================

def image_to_base64(image_path: str) -> str:
    path = Path(image_path)
    with path.open("rb") as image_file:
        return base64.b64encode(image_file.read()).decode()


def set_page_style():
    background_image = image_to_base64("background.png")

    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/png;base64,{background_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        [data-testid="stHeader"] {{
            background: rgba(0, 0, 0, 0);
        }}

        [data-testid="stSidebar"] {{
            background-color: rgba(255, 250, 240, 0.86);
            border-right: 1px solid rgba(120, 90, 60, 0.25);
        }}

        .block-container {{
            background-color: rgba(255, 250, 240, 0.84);
            padding: 2.5rem 3rem;
            border-radius: 24px;
            margin-top: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(60, 40, 20, 0.18);
            border: 1px solid rgba(120, 90, 60, 0.18);
        }}

        h1, h2, h3 {{
            color: #4a2f22;
        }}

        p, label, div {{
            color: #3b2a22;
        }}

        .stTextArea textarea {{
            background-color: rgba(255, 255, 255, 0.78);
            border-radius: 14px;
            border: 1px solid rgba(120, 90, 60, 0.35);
            color: #2f211b;
        }}

        .stSelectbox div[data-baseweb="select"] > div {{
            background-color: rgba(255, 255, 255, 0.78);
            border-radius: 12px;
        }}

        .stButton > button {{
            border-radius: 14px;
            padding: 0.6rem 1.4rem;
            background-color: #7b4b35;
            color: white;
            border: none;
            font-weight: 600;
        }}

        .stButton > button:hover {{
            background-color: #5f3828;
            color: white;
            border: none;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


set_page_style()


# =========================
# UI
# =========================

st.title("📜 AI Poetry Translator")

st.write(
    "Переводчик русских стихов. "
    "Введите стихотворение, выберите язык и модель."
)

with st.sidebar:
    st.header("Settings")

    selected_model = AVAILABLE_MODELS[0]

    st.markdown(
    f"""
    <div style="
        background-color: rgba(255,255,255,0.78);
        padding: 0.7rem 1rem;
        border-radius: 12px;
        border: 1px solid rgba(120, 90, 60, 0.35);
        margin-bottom: 1rem;
    ">
        <strong>Model:</strong> {selected_model}
    </div>
    """,
    unsafe_allow_html=True,
)

    target_language = st.selectbox(
        "Target language",
        LANGUAGES,
        index=0,
    )


default_poem = """Я вас любил: любовь еще, быть может,
В душе моей угасла не совсем;
Но пусть она вас больше не тревожит;
Я не хочу печалить вас ничем."""

poem = st.text_area(
    "Введите стихотворение на русском:",
    value=default_poem,
    height=220,
)

translate_button = st.button("Translate", type="primary")

if translate_button:
    try:
        with st.spinner("Модель переводит стихотворение..."):
            translation = translate_poem(
                poem=poem,
                target_language=target_language,
                model=selected_model,
            )

        st.subheader("Translation")
        st.text(translation)

        with st.expander("Prompt used"):
            st.code(build_translation_prompt(poem, target_language))

    except Exception as ex:
        st.error("Ошибка при переводе.")
        st.code(str(ex))
