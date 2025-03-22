import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import requests
from gtts import gTTS
from io import BytesIO
import base64

# Load a free Hugging Face model (no API key required)
@st.cache_resource
@st.cache_resource
def load_hf_model():
    model_name = "HuggingFaceH4/zephyr-7b-beta"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    return pipeline("text-generation", model=model, tokenizer=tokenizer)


text_generator = load_hf_model()

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

# Function to explain Bible verse using a Hugging Face model
def explain_bible_verse_hf(verse_text):
    prompt = (
        f"You are a thoughtful and compassionate Bible teacher.\n"
        f"Explain the Bible verse below clearly and in detail.\n"
        f"Include real-life examples and moral lessons.\n\n"
        f"BIBLE VERSE: \"{verse_text}\"\n\nEXPLANATION:"
    )

    result = text_generator(prompt, max_new_tokens=300, temperature=0.7, do_sample=True)[0]['generated_text']
    return result.replace(prompt, "").strip()

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
    st.title("📖 Bible Verse Explainer with Hugging Face Model")

    st.write("""
    Welcome! This app allows you to:
    1. Enter a Bible verse reference (e.g., John 3:16)
    2. Automatically fetch and display the verse
    3. Get a detailed explanation with real-life examples and moral lessons (Powered by Hugging Face LLM)
    4. Listen to the explanation and download it as audio
    """)

    st.markdown("""
    🔎 **Tips for Quality Inputs:**
    - Use standard formats like `John 3:16`, `Genesis 1:1`, or `Psalm 23:1`
    - Ensure the book name is spelled correctly
    - Use only one verse reference at a time
    """)

    verse_ref = st.text_input("🔍 Enter Bible Verse Reference (e.g., John 3:16):")

    if verse_ref:
        with st.spinner("Fetching verse..."):
            verse_text = fetch_bible_verse(verse_ref)

        if "Sorry" not in verse_text and "error" not in verse_text:
            st.subheader("📜 Bible Verse")
            st.write(verse_text)

            with st.spinner("Generating explanation using Hugging Face model..."):
                explanation = explain_bible_verse_hf(verse_text)
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
