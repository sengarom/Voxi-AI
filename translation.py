"""
Voxi ASR Translation Script

This script translates ASR output segments into English using Hugging Face MarianMT models.
It preserves speaker labels, timestamps, and original text, and is easy to extend for more languages.
Optimized to load each model only once per language for faster execution.
Automatically detects language using langdetect.
"""

from transformers import MarianMTModel, MarianTokenizer
from langdetect import detect  # Language detection library

# Mapping ISO language codes to MarianMT model names
LANG_CODE_TO_MODEL = {
    "hi": "Helsinki-NLP/opus-mt-hi-en",  # Hindi to English
    "fr": "Helsinki-NLP/opus-mt-fr-en",  # French to English
    # Add more language mappings here as needed
}

# Cache for loaded models and tokenizers
MODEL_CACHE = {}

def get_model_and_tokenizer(src_lang):
    """
    Load and cache MarianMT model and tokenizer for a given language.
    """
    if src_lang not in MODEL_CACHE:
        model_name = LANG_CODE_TO_MODEL.get(src_lang)
        if not model_name:
            raise ValueError(f"No translation model found for language: {src_lang}")
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        MODEL_CACHE[src_lang] = (model, tokenizer)
    return MODEL_CACHE[src_lang]

def translate_text(text, src_lang):
    """
    Translate text from src_lang to English using MarianMT.
    If src_lang is English, return text as-is.
    """
    if src_lang == "en":
        return text  # No translation needed

    model, tokenizer = get_model_and_tokenizer(src_lang)
    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
    tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    return tgt_text[0]

def translate_asr_segments(segments):
    """
    Detect language and translate each ASR segment's text to English, preserving metadata.
    Adds a 'translation' key to each segment.
    """
    for seg in segments:
        # Detect language automatically if not provided or is None/empty
        lang = seg.get("language")
        if not lang or lang == "":
            lang = detect(seg["text"])
            seg["language"] = lang  # Optionally update the segment with detected language
        seg["translation"] = translate_text(seg["text"], lang)
    return segments

# Example input: list of ASR segments (language field can be omitted or empty)
example_segments = [
    {"speaker": "SPEAKER_01", "start": 0.0, "end": 4.2, "text": "Ravi de vous rencontrer", "language": ""},
    {"speaker": "SPEAKER_02", "start": 4.3, "end": 7.5, "text": "I am fine, thank you.", "language": ""}
]

# Translate and print results for verification
if __name__ == "__main__":
    translated_segments = translate_asr_segments(example_segments)
    for seg in translated_segments:
        print(f"Speaker: {seg['speaker']}, Start: {seg['start']}, End: {seg['end']}")
        print(f"Detected Language: {seg['language']}")
        print(f"Original Text: {seg['text']}")
        print(f"Translated Text: {seg['translation']}")