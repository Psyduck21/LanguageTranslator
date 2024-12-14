import os
import logging
import joblib
import gdown
from gtts import gTTS
import speech_recognition as sr
import re
from transformers import MarianMTModel, MarianTokenizer, pipeline
from pydub import AudioSegment
from pydub.utils import which

# Set up logging
logging.basicConfig(
    filename="./log/audio_processing.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# # Check for ffmpeg installation
# ffmpeg_path = which("ffmpeg")
# if not ffmpeg_path:
#     logging.error("FFmpeg is not installed or not added to PATH. Please install FFmpeg to proceed.")
#     raise EnvironmentError("FFmpeg is required but not found in PATH.")



# Set up logging
logging.basicConfig(
    filename="./log/audio_processing.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Model paths
base_model_path = "./jobfiles"
translator_model_path = os.path.join(base_model_path, 'translator')
summarizer_model_path = os.path.join(base_model_path, 'summarizer')
print("Translator model path:", translator_model_path)

# Helper function to download .safetensor files
def download_safetensor_file(url, save_dir, filename):
    """Downloads the .safetensor file only if it does not exist."""
    file_path = os.path.join(save_dir, filename)
    if not os.path.exists(file_path):
        os.makedirs(save_dir, exist_ok=True)
        logging.info(f"Downloading model from {url} to {file_path}...")
        try:
            gdown.download(url, file_path, quiet=False)
            logging.info(f"Model downloaded successfully to {file_path}")
        except Exception as e:
            logging.error(f"Error downloading model from {url}: {e}")
            raise
    else:
        logging.info(f"{file_path} already exists. Skipping download.")

# Model URLs (replace with actual file IDs for .safetensor)
translator_model_url = "https://drive.google.com/uc?id=1jnhm9oxrgaonwSGeNVR2vL8EFoy8uzn8"  # Replace with actual file ID
summarizer_model_url = "https://drive.google.com/uc?id=1lFhQPaOhv_YMf-W-pMjZQHXvA5pbnphL"  # Replace with actual file ID

translator_filename = "model.safetensors" 
summarizer_filename = "model.safetensors"

# Download the .safetensor files if they do not exist
if not os.path.exists(os.path.join(translator_model_path, translator_filename)):
    download_safetensor_file(translator_model_url, translator_model_path, translator_filename)

if not os.path.exists(os.path.join(summarizer_model_path, summarizer_filename)):
    download_safetensor_file(summarizer_model_url, summarizer_model_path, summarizer_filename)

logging.info("Loading models...")

# Load formality, emotion, sentiment models and others
formality_model = joblib.load(os.path.join(base_model_path, 'formality_analysis.pkl'))
emotion_model = joblib.load(os.path.join(base_model_path, 'Emotion_classification.pkl'))
sentiment_model = joblib.load(os.path.join(base_model_path, 'sentiment_analysis.pkl'))
tfidf_vectorizer = joblib.load(os.path.join(base_model_path, 'tfidf_vectorizer.pkl'))
scaler = joblib.load(os.path.join(base_model_path, 'scaler.pkl'))

# Load translation model
translator_tokenizer = MarianTokenizer.from_pretrained(translator_model_path, add_prefix_space=False)
translator_model = MarianMTModel.from_pretrained(translator_model_path)

# Load summarizer model
summarizer = pipeline("summarization", model=summarizer_model_path, tokenizer=summarizer_model_path)

logging.info("Models loaded successfully.")

# Mapping dictionaries
emotion_mapping = {0: "Sadness", 1: "Joy", 2: "Love", 3: "Anger", 4: "Fear", 5: "Surprise"}
formality_mapping = {1: "Formal", -1: "Informal"}
sentiment_mapping = {0: "Negative", 2: "Positive"}

# Utility Functions
def preprocess_text(text):
    """Cleans and preprocesses text for model input."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text


def convert_mp3_to_wav(mp3_path):
    """Converts an MP3 file to WAV format."""
    try:
        logging.info(f"Converting MP3 file: {mp3_path}")
        if not os.path.isfile(mp3_path):
            raise FileNotFoundError(f"MP3 file not found: {mp3_path}")

        audio = AudioSegment.from_mp3(mp3_path)
        wav_path = mp3_path.replace(".mp3", ".wav")
        audio.export(wav_path, format="wav")
        logging.info(f"Converted MP3 to WAV: {wav_path}")
        return wav_path
    except Exception as e:
        logging.error(f"Error converting MP3 to WAV: {e}")
        raise


def transcribe_audio(audio_path, language="en-US"):
    """Transcribes an audio file to text."""
    try:
        if audio_path.endswith(".mp3"):
            audio_path = convert_mp3_to_wav(audio_path)

        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)

        transcription = recognizer.recognize_google(audio_data, language=language)
        if not transcription.strip():
            raise ValueError("Transcription is empty.")
        return transcription
    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        raise


def classify_text(text):
    """Classifies text using pre-trained models."""
    try:
        preprocessed_text = preprocess_text(text)
        vectorized_text = tfidf_vectorizer.transform([preprocessed_text])
        scaled_text = scaler.transform(vectorized_text)

        formality = formality_model.predict(scaled_text)[0]
        emotion = emotion_model.predict(scaled_text)[0]
        sentiment = sentiment_model.predict(scaled_text)[0]

        return (
            formality_mapping.get(formality, "Unknown"),
            emotion_mapping.get(emotion, "Unknown"),
            sentiment_mapping.get(sentiment, "Unknown"),
        )
    except Exception as e:
        logging.error(f"Error during text classification: {e}")
        raise


def summarize_text(text):
    """Summarizes input text using a pre-trained model."""
    try:
        summary = summarizer(text, max_length=40, min_length=10, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        logging.error(f"Error during summarization: {e}")
        raise


def translate_text(text, target_lang="hi"):
    """Translates text to the specified language."""
    try:
        inputs = translator_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = translator_model.generate(**inputs)
        return translator_tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        logging.error(f"Error during translation: {e}")
        raise


def text_to_speech(text, output_file="output.mp3"):
    """Converts text to speech and saves it as an MP3 file."""
    try:
        logging.info(f"Converting text to speech. Output file: {output_file}")
        tts = gTTS(text)
        tts.save(output_file)
        logging.info(f"Text-to-speech conversion successful. Saved at: {output_file}")
        return output_file
    except Exception as e:
        logging.error(f"Error during text-to-speech conversion: {e}")
        raise


# Main Function
def process_audio(input_mp3, output_mp3, target_lang="hi"):
    """Processes an audio file and generates results."""
    try:
        logging.info(f"Processing audio file: {input_mp3}")
        if not os.path.isfile(input_mp3):
            raise FileNotFoundError(f"Input file not found: {input_mp3}")

        original_text = transcribe_audio(input_mp3)
        formality, emotion, sentiment = classify_text(original_text)
        summary = summarize_text(original_text)
        translated_text = translate_text(original_text, target_lang=target_lang)

        output_mp3 = f"./output/translated_{os.path.basename(input_mp3).replace('.mp3', '.mp3')}"
        text_to_speech(translated_text, output_mp3)

        return {
            "original_text": original_text,
            "formality": formality,
            "emotion": emotion,
            "sentiment": sentiment,
            "summary": summary,
            "translated_text": translated_text,
            "output_mp3": output_mp3,
        }
    except Exception as e:
        logging.error(f"Error in processing audio: {e}")
        raise


if __name__ == "__main__":
    input_file = "./input/input.mp3"
    try:
        results = process_audio(input_file)
        for key, value in results.items():
            print(f"{key}: {value}")
    except Exception as e:
        print("An error occurred. Check logs for details.")
