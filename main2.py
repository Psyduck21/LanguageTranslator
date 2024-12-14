import os
import logging
import joblib
from gtts import gTTS
import speech_recognition as sr
import re
from transformers import MarianMTModel, MarianTokenizer, pipeline
import torch
from pydub import AudioSegment
from pydub.utils import which
import gdown
import zipfile

# Set up logging
logging.basicConfig(
    filename="./log/audio_processing.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Check for ffmpeg installation
ffmpeg_path = which("ffmpeg")
if not ffmpeg_path:
    logging.error("FFmpeg is not installed or not added to PATH. Please install FFmpeg to proceed.")
    raise EnvironmentError("FFmpeg is required but not found in PATH.")

# Google Drive file and folder IDs
GDRIVE_FILES = {
    "emotion_model": "1J_COkdW3boCXbDSkDOVU_NfRgeNTeak7",
    "formality_model": "1MLESIU4v-bAMgBNam0Cn1PQTyAgCNZKP",
    "sentiment_model": "1XKS1Ki-oX-Vc6xi5L7wnzV7gKArEbZ6a",
    "tfidf_vectorizer": "1k2wnbgh843rUQgNtzJB0LDdDdeX73cfl",
    "scaler": "1OPj2fngI1q9iYmFmX0LgFY5kLR-8rdB-",
}

GDRIVE_FOLDERS = {
    "translator_model": "12-UOf9yyreulmHwzdGNlBR83I3wXzXMS",  # Translator folder ID
    "summarizer_model": "1fgTDsCTu1EttY0c4_hkw3j7y-fON-sqG"   # Summarizer folder ID
}

# Local paths for downloaded files
download_dir = "./jobfiles"
os.makedirs(download_dir, exist_ok=True)

# Function to download files from Google Drive
def download_from_gdrive(file_id, output_path):
    if not os.path.exists(output_path):
        logging.info(f"Downloading {output_path} from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)
        # Extract if the downloaded file is a ZIP archive
        if output_path.endswith('.zip'):
            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(output_path))
            os.remove(output_path)  # Clean up the ZIP file after extraction
    else:
        logging.info(f"{output_path} already exists. Skipping download.")

# Function to download entire folder from Google Drive
def download_folder_from_gdrive(folder_id, output_path):
    if not os.path.exists(output_path):
        logging.info(f"Downloading folder {output_path} from Google Drive...")
        gdown.download_folder(f"https://drive.google.com/drive/folders/{folder_id}", output=output_path, quiet=False)
    else:
        logging.info(f"Folder {output_path} already exists. Skipping download.")

# Download all individual files
for filename, file_id in GDRIVE_FILES.items():
    output_path = os.path.join(download_dir, filename)
    download_from_gdrive(file_id, output_path)

# Download all folders
for folder_name, folder_id in GDRIVE_FOLDERS.items():
    folder_path = os.path.join(download_dir, folder_name)
    download_folder_from_gdrive(folder_id, folder_path)

# Model paths and loading
logging.info("Loading models...")
model_path = download_dir

formality_model = joblib.load(os.path.join(model_path, 'formality_model'))
emotion_model = joblib.load(os.path.join(model_path, 'emotion_model'))
sentiment_model = joblib.load(os.path.join(model_path, 'sentiment_model'))
tfidf_vectorizer = joblib.load(os.path.join(model_path, 'tfidf_vectorizer'))
scaler = joblib.load(os.path.join(model_path, 'scaler'))

translator_model_path = os.path.join(model_path, 'translator_model')
translator_tokenizer = MarianTokenizer.from_pretrained(translator_model_path, add_prefix_space=False)
translator_model = MarianMTModel.from_pretrained(translator_model_path)

summarizer_model_path = os.path.join(model_path, 'summarizer_model')
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
