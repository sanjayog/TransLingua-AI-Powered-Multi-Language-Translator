import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

st.set_page_config(page_title="TransLingua", layout="centered")

@st.cache_resource
def load_model(src_lang, tgt_lang):
    model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    return model, tokenizer

def translate_text(text, src_lang, tgt_lang):
    model, tokenizer = load_model(src_lang, tgt_lang)
    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
    return tokenizer.decode(translated[0], skip_special_tokens=True)

def main():
    st.markdown(
    """
    <div style="background-color:#FFE5B4;padding:10px;margin-bottom:30px">
    <h1 style="color:black;text-align:center;">TransLingua</h1>
    <h3 style="color:black;text-align:center;">AI-Powered Multi-Language Translator</h3>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.header("Translation Settings")

    language_dict = {
        "English": "en",
        "French": "fr",
        "German": "de",
        "Spanish": "es",
        "Italian": "it",
        "Russian": "ru",
        "Chinese": "zh",
        "Japanese": "ja",
        "Korean": "ko",
        "Arabic": "ar",
        "Urdu": "ur",
    }

    src_lang = st.sidebar.selectbox("Source Language", list(language_dict.keys()))
    tgt_lang = st.sidebar.selectbox("Target Language", list(language_dict.keys()))

    text = st.text_area("Enter Text to Translate")

    if st.button("Translate"):
        if text.strip() == "":
            st.warning("Please enter text to translate.")
        else:
            with st.spinner("Translating..."):
                try:
                    translated_text = translate_text(
                        text,
                        language_dict[src_lang],
                        language_dict[tgt_lang]
                    )
                    st.success(translated_text)
                except Exception:
                    st.error("Translation model not available for this language pair.")

    st.sidebar.info("Powered by Hugging Face MarianMT Models")

if __name__ == "__main__":
    main()
