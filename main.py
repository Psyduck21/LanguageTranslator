import os
import logging
import speech_recognition as sr
import re
from transformers import pipeline, TFMarianMTModel, T5Tokenizer, TFT5ForConditionalGeneration, MarianTokenizer 
from sklearn.feature_extraction.text import CountVectorizer
import audioread
import soundfile as sf
import numpy as np
import joblib
from gtts import gTTS
import warnings
warnings.filterwarnings('ignore')

TF_ENABLE_ONEDNN_OPTS=0

# Set up logging
logging.basicConfig(
    filename="./log/audio_processing.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Model paths and loading
model_path = "./jobfiles"
if not os.path.exists(model_path):
    logging.error(f"Model path does not exist: {model_path}")
    raise FileNotFoundError(f"Model path not found: {model_path}")

logging.info("Loading models...")

formality_model = joblib.load(f'{model_path}/formality.pkl')
emotion_model = joblib.load(f'{model_path}/Emotion_classification.pkl')
sentiment_model = joblib.load(f'{model_path}/sentiment_analysis.pkl')
tfidf_vectorizer = joblib.load(f'{model_path}/tfidf_vectorizer.pkl')
scaler = joblib.load(f'{model_path}/scaler.pkl')
count_vectorizer = joblib.load(f"{model_path}/countvectorizer.pkl")
# Define model directory to store models locally
MODEL_DIR = "./pre-models"

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Example usage:
language_pairs = ['en-fr', 'en-ja', 'en-hi']
translation_pipelines = {}
def load_or_save_model_marianMT(language_pair: str, model_name: str, model_class, tokenizer_class, model_dir: str):
    """
    Check if the model exists locally, load it if it does, otherwise download and save it.
    
    Args:
        language_pair (str): The language pair (e.g., 'en-fr', 'en-ja').
        model_name (str): The model name (e.g., "Helsinki-NLP/opus-mt-en-fr").
        model_class: The class of the model (e.g., MarianMTModel).
        tokenizer_class: The tokenizer class (e.g., MarianTokenizer).
        model_dir (str): Directory where the model will be saved.
        device (int): The device ID for GPU (default is 0 for the first GPU, -1 for CPU).
    
    Returns:
        pipeline: The loaded translation pipeline.
    """
    model_path = os.path.join(model_dir, language_pair)

    # Check if the model exists in the cache
    if language_pair in translation_pipelines:
        logging.info(f"Using cached translation pipeline for {language_pair}")
        return translation_pipelines[language_pair]
    
    # Load model from local storage or download
    if os.path.exists(model_path):
        logging.info(f"Loading {language_pair} model from local storage...")
        model = model_class.from_pretrained(model_path)
        tokenizer = tokenizer_class.from_pretrained(model_path)
    else:
        logging.info(f"{language_pair} model not found locally. Downloading and saving...")
        # Download and save the model locally
        model = model_class.from_pretrained(model_name)
        tokenizer = tokenizer_class.from_pretrained(model_name)
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)

    
    # Store the model and tokenizer in the pipeline cache
    translation_pipelines[language_pair] = {
        'model': model,
        'tokenizer': tokenizer
    }

    return translation_pipelines[language_pair]


def load_all_models(language_pairs, model_dir):
    """
    Load all translation models and cache them in the translation_pipelines dictionary.
    
    Args:
        language_pairs (list): List of language pairs to load (e.g., ['en-fr', 'en-ja']).
        model_dir (str): Directory where models will be saved.
        device (int): The device ID for GPU (default is 0 for the first GPU, -1 for CPU).
    """
    for pair in language_pairs:
        if pair != "en-ja":
            model_name = f"Helsinki-NLP/opus-mt-{pair}"
            load_or_save_model_marianMT(pair, model_name, TFMarianMTModel, MarianTokenizer, model_dir)
        else:
            model_name = "Helsinki-NLP/opus-tatoeba-en-ja"
            load_or_save_model_marianMT(pair, model_name, TFMarianMTModel, MarianTokenizer, model_dir)

load_all_models(language_pairs,MODEL_DIR)

# T5 model for summarization, tone adjustment, and other text2text generation tasks
t5_model = "t5-small"
t5_model_path = f"{MODEL_DIR}/t5"
    
    # Check if the model exists locally
if os.path.exists(t5_model_path):
        logging.info(f"Loading {t5_model} model from local storage...")
        model_t5 = TFT5ForConditionalGeneration.from_pretrained(t5_model, from_pt = True)
        tokenizer_t5 = T5Tokenizer.from_pretrained(t5_model)
else:
        logging.info(f"{t5_model} model not found locally. Downloading and saving...")
        # Download model and save it locally
        model_t5 = TFT5ForConditionalGeneration.from_pretrained(t5_model, from_pt=True)
        tokenizer_t5 = T5Tokenizer.from_pretrained(t5_model)
        model_t5.save_pretrained(t5_model_path)
        tokenizer_t5.save_pretrained(t5_model_path)
        
summarizer = pipeline("summarization", model=model_t5, tokenizer=tokenizer_t5)

logging.info(translation_pipelines)
logging.info("Models loaded successfully.")

# Mapping dictionaries
emotion_mapping = {0: "Sadness", 1: "Joy", 2: "Love", 3: "Anger", 4: "Fear", 5: "Surprise"}
formality_mapping = {2: "Formal", 1: "Informal"}
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


# def convert_mp3_to_wav(mp3_path):
#     try:
#         # Open the MP3 file using audioread
#         logging.info("Converting mp3 to wav file.")
#         with audioread.audio_open(mp3_path) as f:
#             # Read the raw audio data and sample rate
#             audio_data = []
#             for buf in f:
#                 audio_data.append(buf)

#             # Convert byte data to numpy array of appropriate dtype
#             audio_data = np.concatenate([np.frombuffer(b, dtype=np.int16) for b in audio_data])

#             # Define the WAV path
#             wav_path = mp3_path.replace(".mp3", ".wav")

#             # Write to WAV using soundfile
#             sf.write(wav_path, audio_data, f.samplerate)
#         logging.info(f"Done Converting and save to {mp3_path}")
#         return wav_path
#     except Exception as e:
#         print(f"Error during conversion: {e}")
#         raise


def transcribe_audio(audio_path):
    """Transcribes an audio file to text and detects the language of the transcription."""
    try:
        logging.info(f"Starting transcription for: {audio_path}")
        
        # Use SpeechRecognition to transcribe the audio
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)

        # Perform the transcription using Google Web Speech API
        transcription = recognizer.recognize_google(audio_data)

        if not transcription.strip():
            raise ValueError("Transcription is empty.")

        logging.info(f"Transcription completed: {transcription}")

        
        # Return transcription if language is English
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
        nb_vectorized = count_vectorizer.transform([preprocessed_text])
        
        formality = formality_model.predict(nb_vectorized)[0]
        emotion = emotion_model.predict(scaled_text)[0]
        sentiment = sentiment_model.predict(nb_vectorized)[0]
        logging.info("Done Classification")

        return (
            formality_mapping.get(formality, "Unknown"),
            emotion_mapping.get(emotion, "Unknown"),
            sentiment_mapping.get(sentiment, "Unknown"),
        )
    except Exception as e:
        logging.error(f"Error during text classification: {e}")
        raise


def summarize_text(text):
    """Summarizes input text using a pre-trained model with dynamic length constraints."""
    try:
        logging.info(f" text: {text}")
        # Calculate max and min length based on the length of the text
        text_len = len(text.split())  # Assuming word count is used to determine text length

        logging.info(f" text length {text_len}")
        max_length = max(15, int(text_len * 0.8))  # max_length is 10% of text length, with a minimum of 10 words
        min_length = max(5, int(text_len * 0.05))  # min_length is 5% of text length, with a minimum of 5 words
        logging.info(f" Maax length {max_length}")
        logging.info(f" Min length {min_length}")
        
        # Generate the summary using the T5 model with dynamic length
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        
        # Log the type and content of the summary for debugging purposes
        logging.info(f"Summary output: {summary}")
        
        # Check if the output is a list of dictionaries
        return summary[0]['summary_text']
    
    except Exception as e:
        logging.error(f"Error during summarization: {e}")
        raise


def adjust_formality(adjust_tone, text):
    """
    Adjusts the tone of the input text using the tone adjustment pipeline.
    
    Args:
    adjust_tone (pipeline): Tone adjustment pipeline.
    text (str): The informal text to make formal.
    
    Returns:
    str: The formal version of the input text.
    """
    try:
        formal_text = adjust_tone(f"Make formal: {text}")
        logging.info("Done making formal")
        return formal_text[0]['generated_text']
    except Exception as e:
        logging.error(f"Error during making formal {e}")
        raise

def translate_text(text, target_lang="hi"):
    """
    Translates the input text into the target language using the MarianMT model (TensorFlow version).

    Args:
        text (str): The text to be translated.
        target_lang (str): The target language code (e.g., "hi" for Hindi, "fr" for French).
        device (int): The device ID for GPU (default is 0 for the first GPU, -1 for CPU).
    
    Returns:
        str: The translated text.
    """
    try:
        # Language pair identifier for caching (source language is assumed to be English here)
        language_pair = f"en-{target_lang}"

        # Load or get the translation pipeline for the target language
        translation_pipeline = translation_pipelines[language_pair]

        # Tokenizer and model from the pipeline
        tokenizer = translation_pipeline['tokenizer']
        model = translation_pipeline['model']

        # Tokenize the input text for translation
        inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True)

        # Perform the translation (generate output from the model)
        outputs = model.generate(inputs['input_ids'])

        # Decode the translated tensor output into text
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Log the result and return the translated text
        logging.info(f"Done Translating: {translated_text}")
        return translated_text

    except Exception as e:
        logging.error(f"Error occurred in translation: {e}")
        raise

def text_to_speech(text, output_file):
    """Converts text to speech and saves it as an MP3 file."""
    try:
        logging.info(f"Converting text to speech. Output file: {output_file}")
        tts = gTTS(text,slow=False)
        tts.save(output_file)
        logging.info(f"Text-to-speech conversion successful. Saved at: {output_file}")
        return output_file
    except Exception as e:
        logging.error(f"Error during text-to-speech conversion: {e}")
        raise


def process_audio(input_data, target_lang_1="hi", is_text=False):
    """Processes either an audio file or text content provided by the user and generates results."""
    try:
        logging.info(f"Processing input data")

        # If it's text input (provided by the user)
        if is_text:
            original_text = input_data
            logging.info(f"Processing user input text : {original_text}")

        # If it's an audio file, perform transcription
        else:
            logging.info("Transcribing Audio")
            original_text = transcribe_audio(input_data)

        # Classify text (formality, emotion, sentiment)
        logging.info("Classifying text")
        formality, emotion, sentiment = classify_text(original_text)

        # Summarize text
        logging.info("Summarizing text")
        summary = summarize_text(original_text)

        # Translate to the selected language
        logging.info(f"Translating text to {target_lang_1}")
        translated_text_1 = translate_text(original_text, target_lang_1)
        
        logging.info(f'{target_lang_1} : {translated_text_1}')
                
        output_mp3_1 = f'./output/{target_lang_1}.mp3'

        # Convert translated text to speech (text-to-speech)
        logging.info("Converting translated text to speech")
        text_to_speech(translated_text_1, output_mp3_1)
        logging.info("Text-to-Speech conversion complete")

        # Return all the results as a dictionarsy
        return {
            "original_text": original_text,
            "source_language": "English" if not is_text else "N/A",  # No source language for text input
            "formality": formality,
            "emotion": emotion,
            "sentiment": sentiment,
            "summary": summary,
            "translated_text_1": translated_text_1,
            "output_mp3_1": output_mp3_1,        
        }

    except Exception as e:
        logging.error(f"Error in processing input data: {e}")
        raise



if __name__ == "__main__":
    ...