import streamlit as st
import os
from main import process_audio

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
page = st.sidebar.radio("Navigation", ["Home", "About"])

# Info Section in Sidebar
st.sidebar.write("---")
st.sidebar.markdown("#### Info")
st.sidebar.info("This app uses speech-to-text, text analysis, and text-to-speech to process audio files.", icon="ℹ️")

# Home Page
if page == "Home":
    st.title("🎧 Audio Translator & Text Analyzer")
    st.subheader("Upload your audio file and get insights instantly.")
    st.write("---")
    st.markdown(
        """
        **Upload an MP3 file**, and this app will:
        - Transcribe the audio to text
        - Analyze its formality, emotion, and sentiment
        - Summarize the text
        - Translate the text to a target language
        - Convert the translated text to speech
        """
    )

    # File upload
    st.write("### Step 1: Upload Your Audio File")
    uploaded_file = st.file_uploader("Choose an MP3 file", type=["mp3"])

    # Target language selection
    st.write("### Step 2: Select Target Language")
    target_language = st.selectbox(
        "Select the language to translate the audio into:",
        ["hi - Hindi", "more coming soon.."],
    )

    # File upload section
    st.subheader("Upload and Configure")
    
    # Process the file when the user clicks the "Process" button
    if uploaded_file:
        # Save the uploaded file
        input_dir = "./input/"
        output_dir = "./output/"
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        input_path = os.path.join(input_dir, uploaded_file.name)
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"File uploaded successfully: {uploaded_file.name}")

        if st.button("Process"):
            st.info("Processing the file. This might take a few moments...")
            try:
                # Call the process_audio function
                results = process_audio(input_path, target_language)

                # Display results
                st.subheader("📜 Transcription")
                st.text_area("Original Text", results["original_text"], height=150)

                st.subheader("📝 Text Analysis")
                st.write(f"**Formality:** {results['formality']}")
                st.write(f"**Emotion:** {results['emotion']}")
                st.write(f"**Sentiment:** {results['sentiment']}")

                st.subheader("📖 Summary")
                st.text_area("Summary", results["summary"], height=100)

                st.subheader("🌍 Translated Text")
                st.text_area("Translated Text", results["translated_text"], height=150)
                
                # Play original audio
                st.subheader("🎧 Original Audio")
                st.audio(input_path, format="audio/mp3")
                
                # Downloadable MP3 link
                translated_mp3_path = results["output_mp3"]
                if os.path.exists(translated_mp3_path):
                    st.subheader("🔊 Translated Audio")
                    st.audio(translated_mp3_path, format="audio/mp3")

                    with open(translated_mp3_path, "rb") as audio_file:
                        st.download_button(
                            label="Download Translated Audio",
                            data=audio_file,
                            file_name=os.path.basename(translated_mp3_path),
                            mime="audio/mp3",
                        )

                st.success("Processing complete!")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.info("Please upload an MP3 file to start processing.")

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
