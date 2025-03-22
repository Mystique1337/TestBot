import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from gtts import gTTS
from io import BytesIO
import base64
import requests

# Load Mistral-7B-Instruct model (open-source, no login needed)
@st.cache_resource
def load_mistral_model():
    model_name = "tiiuae/falcon-7b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

llm = load_mistral_model()

# Function to fetch a Bible verse from an API
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

# Function to explain Bible verse using Mistral

def explain_bible_verse_mistral(verse_text):
    prompt = (
        f"[INST]You are a kind and wise Bible teacher. Read the verse below and explain it clearly. "
        f"Include a practical example and highlight the moral lessons.\n\n"
        f"BIBLE VERSE:\n\"{verse_text}\"\n\n"
        f"EXPLANATION:[/INST]"
    )

    result = llm(
        prompt,
        max_new_tokens=350,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        top_k=50
    )

    explanation = result[0]['generated_text'].replace(prompt, "").strip()
    return explanation

# Function to convert explanation to speech
def text_to_speech(text):
    tts = gTTS(text, lang='en')
    mp3_fp = BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    return mp3_fp

# Function to create a download link for the audio file
def get_audio_download_link(audio_bytes, filename):
    b64 = base64.b64encode(audio_bytes.read()).decode()
    href = f'<a href="data:audio/mp3;base64,{b64}" download="{filename}">Download Explanation Audio</a>'
    return href

# Main function to run the Streamlit app
def main():
    st.set_page_config(page_title="Bible Verse Explainer", layout="centered")
    st.title("📖 Bible Verse Explainer with Moral Lessons (Powered by Mistral 7B)")

    st.write("""
    Welcome! This app allows you to:
    1. Enter a Bible verse reference (e.g., John 3:16)
    2. Automatically fetch and display the verse
    3. Get a detailed explanation with real-life examples and moral lessons
    4. Listen to the explanation and download it as audio
    """)

    st.markdown("""
    🔎 **Tips for Quality Inputs:**
    - Use standard formats like `John 3:16`, `Genesis 1:1`, or `Psalm 23:1`.
    - Ensure the book name is spelled correctly.
    - Use only one verse reference at a time for best results.
    """)

    verse_ref = st.text_input("🔍 Enter Bible Verse Reference (e.g., John 3:16):")

    if verse_ref:
        with st.spinner("Fetching verse..."):
            verse_text = fetch_bible_verse(verse_ref)

        if "Sorry" not in verse_text and "error" not in verse_text:
            st.subheader("📜 Bible Verse")
            st.write(verse_text)

            with st.spinner("Generating explanation with Mistral..."):
                explanation = explain_bible_verse_mistral(verse_text)
            st.subheader("💬 Explanation")
            st.write(explanation)

            with st.spinner("Creating audio..."):
                audio_fp = text_to_speech(explanation)
                st.audio(audio_fp, format='audio/mp3')

                download_link = get_audio_download_link(audio_fp, "bible_explanation.mp3")
                st.markdown(download_link, unsafe_allow_html=True)
        else:
            st.error(verse_text)
    else:
        st.info("Please enter a Bible verse reference above to begin.")

if __name__ == '__main__':
    main()
