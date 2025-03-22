import streamlit as st
from transformers import pipeline
from gtts import gTTS
from io import BytesIO
import base64
import requests

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




def explain_bible_verse_detailed(verse_text):
    """
    This function provides a detailed, context-rich explanation of a Bible verse using an advanced LLM.
    It generates a summary, then crafts a detailed explanation with real-life examples and moral lessons.
    """
    try:
        # Load the summarization pipeline
        summarizer = pipeline("summarization")
        
        # Generate a detailed, natural summary
        explanation_summary = summarizer(
            verse_text,
            max_length=120,
            min_length=60,
            do_sample=False
        )[0]['summary_text']

        # Build a context-rich, AI-driven teaching explanation
        full_explanation_prompt = (
            f"The following Bible verse is:\n\n\"{verse_text}\"\n\n"
            f"Summarize it, explain it in detail like a pastor teaching a congregation, "
            f"include real-life examples that help people relate to it, and highlight moral lessons from it.\n\n"
            f"Here’s a summary: {explanation_summary}\n\n"
            f"Now write a thoughtful explanation."
        )

        # Use a more advanced LLM for a conversational, pastoral tone
        # Here, we're using Mistral 7B, which is known for its efficiency and performance[1]
        explainer = pipeline("text-generation", model="mistralai/Mistral-7B-v0.1")
        detailed = explainer(full_explanation_prompt, max_length=300, do_sample=True)[0]['generated_text']

        # Clean up and format
        detailed_explanation = detailed.replace(full_explanation_prompt, "").strip()

        return detailed_explanation

    except ModuleNotFoundError:
        return "Error: The 'transformers' library is not installed. Please install it using 'pip install transformers'."

# Example input for testing
verse_text = "Love is patient, love is kind. It does not envy, it does not boast, it is not proud."
explanation = explain_bible_verse_detailed(verse_text)
print(explanation)

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
    st.title("📖 Bible Verse Explainer with Moral Lessons")

    st.write("""
    Welcome! This app allows you to:
    1. Enter a Bible verse reference (e.g., John 3:16)
    2. Automatically fetch and display the verse
    3. Get a detailed, conversational explanation with examples and moral lessons
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

            with st.spinner("Generating explanation..."):
                explanation = explain_bible_verse_detailed(verse_text)
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
