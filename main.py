import streamlit as st
from openai import OpenAI
import requests
from gtts import gTTS
from io import BytesIO
import base64

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

# Function to explain Bible verse using OpenAI GPT model (latest SDK)
def explain_bible_verse_openai(verse_text, api_key):
    client = OpenAI(api_key=api_key)

    system_prompt = (
        "You are a thoughtful and compassionate Bible teacher. "
        "Explain the Bible verse provided in clear, detailed language. "
        "Include a real-life example and a moral lesson."
    )

    user_prompt = (
        f"Bible Verse:\n\"{verse_text}\"\n\n"
        "Please explain this verse like a pastor teaching a congregation. "
        "Make it easy to understand, include a real-life example, and highlight the moral lessons."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Use "gpt-3.5-turbo" if you don't have GPT-4 access
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )

        explanation = response.choices[0].message.content.strip()
        return explanation
    except Exception as e:
        return f"An error occurred while generating the explanation: {e}"

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
    st.title("📖 Bible Verse Explainer with GPT (New SDK)")

    st.write("""
    Welcome! This app allows you to:
    1. Enter a Bible verse reference (e.g., John 3:16)
    2. Automatically fetch and display the verse
    3. Get a detailed explanation with real-life examples and moral lessons (Powered by GPT)
    4. Listen to the explanation and download it as audio
    """)

    st.markdown("""
    🔎 **Tips for Quality Inputs:**
    - Use standard formats like `John 3:16`, `Genesis 1:1`, or `Psalm 23:1`
    - Ensure the book name is spelled correctly
    - Use only one verse reference at a time
    """)

    openai_api_key = st.text_input("🔐 Enter your OpenAI API Key:", type="password")
    verse_ref = st.text_input("🔍 Enter Bible Verse Reference (e.g., John 3:16):")

    if verse_ref and openai_api_key:
        with st.spinner("Fetching verse..."):
            verse_text = fetch_bible_verse(verse_ref)

        if "Sorry" not in verse_text and "error" not in verse_text:
            st.subheader("📜 Bible Verse")
            st.write(verse_text)

            with st.spinner("Generating explanation using GPT..."):
                explanation = explain_bible_verse_openai(verse_text, openai_api_key)
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
        st.info("Please enter your OpenAI API key and a Bible verse reference above to begin.")

if __name__ == '__main__':
    main()
