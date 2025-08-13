"""
Voxi ASR Translation Script

This script translates ASR output segments into English using Hugging Face MarianMT models.
It preserves speaker labels, timestamps, and original text, and is easy to extend for more languages.
Optimized to load each model only once per language for faster execution.
Automatically detects language using langdetect.
"""

from transformers import MarianMTModel, MarianTokenizer
from langdetect import detect  # Language detection library
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Mapping ISO language codes to MarianMT model names
LANG_CODE_TO_MODEL = {
    "hi": "Helsinki-NLP/opus-mt-hi-en",  # Hindi to English
    "fr": "Helsinki-NLP/opus-mt-fr-en",  # French to English
    # Add more language mappings here as needed
}

# Cache for loaded models and tokenizers (keyed by model_name)
MODEL_CACHE = {}

def _resolve_model_name(src_lang: str) -> Optional[str]:
    """Resolve a MarianMT model name for a given source language code.

    Tries explicit mapping first, then falls back to the generic
    "Helsinki-NLP/opus-mt-<src>-en" naming convention used by many languages.
    Returns None if src_lang is falsy.
    """
    if not src_lang:
        return None
    src = src_lang.lower()
    # Prefer explicit mapping if present
    if src in LANG_CODE_TO_MODEL:
        return LANG_CODE_TO_MODEL[src]
    # Generic fallback (works for many languages, e.g., es, de, it, ru, zh, ja)
    return f"Helsinki-NLP/opus-mt-{src}-en"

def get_model_and_tokenizer(src_lang: str):
    """Load and cache MarianMT model and tokenizer for a given language code.

    Uses explicit mapping when available, otherwise falls back to
    "Helsinki-NLP/opus-mt-<src>-en". Caches by model_name.
    """
    model_name = _resolve_model_name(src_lang)
    if not model_name:
        raise ValueError("Source language not provided for translation")

    if model_name not in MODEL_CACHE:
        try:
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            MODEL_CACHE[model_name] = (model, tokenizer)
            logger.info("Loaded translation model: %s", model_name)
        except Exception as exc:
            # Surface a clearer error to callers
            raise RuntimeError(f"Failed to load translation model '{model_name}': {exc}") from exc

    return MODEL_CACHE[model_name]

def translate_text(text: str, src_lang: str) -> str:
    """
    Translate text from src_lang to English using MarianMT.
    If src_lang is English, return text as-is.
    """
    if not text:
        return ""

    if (src_lang or "").lower() == "en":
        return text  # No translation needed

    model, tokenizer = get_model_and_tokenizer(src_lang)
    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True, truncation=True))
    tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    return tgt_text[0]

def translate_to_english(text: str, source_lang: Optional[str]) -> str:
    """Backward-compatible API expected by main.py.

    - If source_lang is 'en', returns text.
    - If source_lang is falsy or 'unknown', tries to auto-detect from text.
    - On any error (no model, download failure), returns original text.
    """
    try:
        lang = (source_lang or "").lower()
        if lang in ("", "unknown"):
            try:
                lang = detect(text) if text else "en"
            except Exception:
                lang = "en"

        if lang == "en":
            return text

        return translate_text(text, lang)
    except Exception as exc:
        logger.warning("Translation failed (%s); returning original text", exc)
        return text

def translate_asr_segments(segments):
    """
    Detect language and translate each ASR segment's text to English, preserving metadata.
    - If a segment lacks 'text' but has 'transcription', that value is used as the source text.
    - Adds a 'translation' key to each segment.
    """
    for seg in segments:
        # Prefer 'text'; fall back to 'transcription' if 'text' missing/empty
        source_text = seg.get("text")
        if not source_text:
            source_text = seg.get("transcription") or ""

        # Detect language automatically if not provided or is None/empty
        lang = seg.get("language")
        if not lang or lang == "":
            try:
                lang = detect(source_text) if source_text else "en"
            except Exception:
                lang = "en"
            seg["language"] = lang  # Optionally update the segment with detected language

        # Always attach translation result
        seg["translation"] = translate_to_english(source_text, lang)
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