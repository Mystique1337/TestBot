import streamlit as st
import requests
from openai import OpenAI
from gtts import gTTS
from io import BytesIO
import base64

# ✅ Set Streamlit page config first
st.set_page_config(page_title="Bible Verse Explainer", layout="centered")

# 🔐 Input API Key from sidebar
api_key = st.sidebar.text_input("🔑 Enter OpenRouter API Key", type="password")

# 🎯 Model selection
model_choice = st.sidebar.selectbox(
    "🤖 Choose AI Model",
    [
        "mistralai/mistral-7b-instruct",
        "openai/gpt-3.5-turbo",
        "google/gemma-3-4b-it:free",
        "deepseek/deepseek-r1:free"
    ],
    index=0
)

# 📖 Fetch Bible verse from API
def fetch_bible_verse(reference):
    try:
        response = requests.get(f"https://bible-api.com/{reference.replace(' ', '%20')}")
        if response.status_code == 200:
            return response.json()['text']
        else:
            return "Sorry, the verse could not be found."
    except:
        return "There was an error retrieving the verse."

# 🤖 Explain Bible verse using OpenRouter API (OpenAI SDK v1.x)
def explain_bible_verse_openrouter(verse_text, api_key, model_name):
    try:
        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a kind and wise Bible teacher who explains verses in simple terms with examples and moral lessons."
                },
                {
                    "role": "user",
                    "content": f"Explain this Bible verse simply, with a real-life example and moral lesson:\n\n{verse_text}"
                }
            ],
            temperature=0.7,
            max_tokens=400
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"❌ Error generating explanation: {e}"

# 🎧 Convert text to speech
def text_to_speech(text):
    tts = gTTS(text, lang='en')
    mp3_fp = BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    return mp3_fp

# 📥 Audio download link
def get_audio_download_link(audio_bytes, filename):
    b64 = base64.b64encode(audio_bytes.read()).decode()
    return f'<a href="data:audio/mp3;base64,{b64}" download="{filename}">Download Audio</a>'

# 🧠 App Interface
st.title("📖 Bible Verse Explainer (Powered by OpenRouter AI)")

st.markdown("""
Enter a Bible verse reference like `John 3:16`. This app will:
- Fetch the verse
- Explain it using AI (with examples and morals)
- Convert it to audio for listening or download
""")

verse_ref = st.text_input("🔍 Bible Verse Reference:")

if verse_ref and api_key:
    with st.spinner("📖 Fetching verse..."):
        verse_text = fetch_bible_verse(verse_ref)

    if "Sorry" not in verse_text and "error" not in verse_text:
        st.subheader("📜 Bible Verse")
        st.write(verse_text)

        with st.spinner("💬 Generating explanation..."):
            explanation = explain_bible_verse_openrouter(verse_text, api_key, model_choice)

        st.subheader("💬 Explanation")
        st.write(explanation)

        with st.spinner("🎧 Creating audio..."):
            audio_fp = text_to_speech(explanation)
            st.audio(audio_fp, format='audio/mp3')
            st.markdown(get_audio_download_link(audio_fp, "bible_explanation.mp3"), unsafe_allow_html=True)
    else:
        st.error(verse_text)

elif not api_key:
    st.warning("⚠️ Please enter your OpenRouter API key in the sidebar to begin.")
