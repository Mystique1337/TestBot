import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import requests
from gtts import gTTS
from io import BytesIO
import base64

# Load a small, open-access Hugging Face model
@st.cache_resource
def load_model():
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

text_generator = load_model()

# Fetch Bible verse from API
def fetch_bible_verse(reference):
    try:
        response = requests.get(f"https://bible-api.com/{reference.replace(' ', '%20')}")
        if response.status_code == 200:
            data = response.json()
            return data['text']
        else:
            return "Sorry, the verse could not be found. Please check your input."
    except:
        return "There was an error retrieving the verse."

# Generate explanation using distilgpt2
def explain_bible_verse(verse_text):
    prompt = (
        f"Explain this Bible verse in simple terms, like a kind pastor teaching a child. "
        f"Include a real-life example and a moral lesson:\n\n\"{verse_text}\"\nExplanation:"
    )
    result = text_generator(prompt, max_new_tokens=150, temperature=0.9, do_sample=True)[0]["generated_text"]
    explanation = result.replace(prompt, "").strip()
    return explanation

# Convert explanation to speech
def text_to_speech(text):
    tts = gTTS(text, lang='en')
    mp3_fp = BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    return mp3_fp

# Create audio download link
def get_audio_download_link(audio_bytes, filename):
    b64 = base64.b64encode(audio_bytes.read()).decode()
    href = f'<a href="data:audio/mp3;base64,{b64}" download="{filename}">Download Audio</a>'
    return href

# Streamlit app
def main():
    st.set_page_config(page_title="Bible Verse Explainer", layout="centered")
    st.title("📖 Bible Verse Explainer (Hugging Face GPT-2)")

    st.markdown("""
    👉 Enter a Bible verse (e.g., `John 3:16`)  
    🤖 AI will explain it in simple words, with a real-life example and a moral lesson  
    🎧 You can listen to the explanation and download it as audio
    """)

    verse_ref = st.text_input("🔍 Enter Bible Verse Reference (e.g., John 3:16):")

    if verse_ref:
        with st.spinner("Fetching verse..."):
            verse_text = fetch_bible_verse(verse_ref)

        if "Sorry" not in verse_text and "error" not in verse_text:
            st.subheader("📜 Bible Verse")
            st.write(verse_text)

            with st.spinner("Generating explanation..."):
                explanation = explain_bible_verse(verse_text)

            st.subheader("💬 Explanation")
            st.write(explanation)

            with st.spinner("Creating audio..."):
                audio_fp = text_to_speech(explanation)
                st.audio(audio_fp, format='audio/mp3')
                st.markdown(get_audio_download_link(audio_fp, "bible_explanation.mp3"), unsafe_allow_html=True)
        else:
            st.error(verse_text)

if __name__ == "__main__":
    main()
