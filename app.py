import streamlit as st
import os
import time
from main import process_audio  # Assuming the updated process_audio is imported

# Set up page configuration
st.set_page_config(
    page_title="Audio Translator & Analyzer",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar navigation
st.sidebar.title("🎤 Audio Translator")
st.sidebar.markdown("Translate and analyze audio files in one place!")
st.sidebar.write("---")

# Navigation
page = st.sidebar.radio("Navigation", ["Home", "About", "Feedback"])

# Info Section in Sidebar
st.sidebar.write("---")
st.sidebar.markdown("#### Info")
st.sidebar.info("This app uses speech-to-text, text analysis, and text-to-speech to process audio files.", icon="ℹ️")

# Home Page
if page == "Home":
    st.title("🎹 Audio Processing and Translation App")
    st.markdown(
        """
        **Upload an MP3 file** or enter **text**, and this app will:
        - Transcribe the audio to text (for .wav files)
        - Analyze its formality, emotion, and sentiment
        - Summarize the text
        - Translate the text to a target language
        - Convert the translated text to speech
        """
    )

    # File upload section
    st.subheader("Upload and Configure")

    # File uploader for audio
    uploaded_audio = st.file_uploader("Upload a .wav file", type=["wav"], label_visibility="collapsed")

    # Text input field
    input_text = st.text_area(
        "Or, input text for analysis",
        height=None,  # Dynamically adjusts to content
        placeholder="Type or paste your text here...",
        key="dynamic_text_area"
    )

    # Language selection
    target_lang_1 = st.selectbox(
        "Select Target Language for Translation", ["hi", "ja", "fr"], index=0
    )

    # Process Button
    if st.button("Process"):
        if not uploaded_audio and not input_text.strip():
            st.warning("Please upload an audio file or enter text to process.")
        else:
            start_time = time.time()  # Start timer

            try:
                # Handle audio input
                if uploaded_audio:
                    input_dir = "./input/"
                    output_dir = "./output/"
                    os.makedirs(input_dir, exist_ok=True)
                    os.makedirs(output_dir, exist_ok=True)

                    input_audio_path = os.path.join(input_dir, uploaded_audio.name)
                    with open(input_audio_path, "wb") as f:
                        f.write(uploaded_audio.getbuffer())

                    st.success(f"File uploaded: {uploaded_audio.name}")
                    st.info("Processing the audio...")

                    results = process_audio(input_audio_path, target_lang_1)  # Audio processing

                # Handle text input
                elif input_text.strip():
                    st.info("Processing the text input...")
                    results = process_audio(input_text.strip(), target_lang_1, is_text=True)  # Text processing

                # Display results
                st.subheader("📜 Original Text")
                st.text_area("Original Text", results["original_text"], height=150)

                st.subheader("🖋️ Text Analysis")
                st.write(f"**Formality:** {results['formality']}")
                st.write(f"**Emotion:** {results['emotion']}")
                st.write(f"**Sentiment:** {results['sentiment']}")

                st.subheader("📖 Summary")
                st.text_area("Summary", results["summary"], height=100)

                st.subheader("🌍 Translated Text")
                st.text_area(f"Translated Text ({target_lang_1})", results["translated_text_1"], height=150)

                # Handle audio playback and download
                if "output_mp3_1" in results and os.path.exists(results["output_mp3_1"]):
                    st.subheader(f"🔊 Translated Audio ({target_lang_1})")
                    st.audio(results["output_mp3_1"], format="audio/mp3")

                    with open(results["output_mp3_1"], "rb") as audio_file:
                        st.download_button(
                            label=f"Download Translated Audio ({target_lang_1})",
                            data=audio_file,
                            file_name=os.path.basename(results["output_mp3_1"]),
                            mime="audio/mp3",
                        )

                # Processing time
                processing_time = time.time() - start_time
                st.write(f"Processing Time: {processing_time:.2f} seconds")

            except Exception as e:
                st.error(f"An error occurred: {e}")

# About Page
elif page == "About":
    st.title("About This App")
    st.markdown(
        """
        This application is built to process and analyze audio files. It leverages:
        - **Speech Recognition** for transcribing audio
        - **Machine Learning Models** for emotion, sentiment, and formality analysis
        - **NLP Models** for summarization and translation
        - **Text-to-Speech (TTS)** for generating translated audio

        Created by Akshat Kumar Prajapati (https://github.com/Psyduck21).
        """
    )

# Feedback Page
elif page == "Feedback":
    st.title("🖍️ User Feedback")
    st.markdown("We value your feedback! Please let us know your thoughts about this app.")

    # User Feedback Section
    st.subheader("Your Experience")
    feedback_experience = st.selectbox(
        "How was your experience with this app?",
        ["Excellent", "Good", "Average", "Poor"]
    )

    st.subheader("Comments")
    feedback_comments = st.text_area(
        "Any suggestions or comments?",
        placeholder="Write your feedback here..."
    )

    # Submit Feedback Button
    if st.button("Submit Feedback"):
        feedback_dir = "./feedback/"
        os.makedirs(feedback_dir, exist_ok=True)
        feedback_file = os.path.join(feedback_dir, "feedback.txt")

        with open(feedback_file, "a") as f:
            f.write(f"Experience: {feedback_experience}\n")
            f.write(f"Comments: {feedback_comments}\n")
            f.write("-" * 50 + "\n")

        st.success("Thank you for your feedback!")
