# Helsinki-NLP Translation Implementation (Optimized for Batching)
import logging
from typing import Optional, List, Dict
import torch
from transformers import MarianMTModel, MarianTokenizer
import re

logger = logging.getLogger("VoxiAPI")

# Dictionary of language codes to Helsinki-NLP model names
HELSINKI_MODELS = {
    "hi_en": "Helsinki-NLP/opus-mt-hi-en",
    "ur_en": "Helsinki-NLP/opus-mt-ur-en",
    "es_en": "Helsinki-NLP/opus-mt-es-en",
    "fr_en": "Helsinki-NLP/opus-mt-fr-en",
    "de_en": "Helsinki-NLP/opus-mt-de-en",
    "ar_en": "Helsinki-NLP/opus-mt-ar-en",
    "bn_en": "Helsinki-NLP/opus-mt-bn-en",
    # Multilingual model as a fallback
    "mul_en": "Helsinki-NLP/opus-mt-mul-en",
}

# Cache for loaded models and tokenizers
_translation_cache = {}

def _get_model_for_language_pair(source_lang: str, target_lang: str = "en"):
    """
    Lazily loads and caches the appropriate translation model and tokenizer.
    Falls back to a multilingual model if a specific one isn't found.
    """
    model_key = f"{source_lang}_{target_lang}"
    
    if model_key in _translation_cache:
        return _translation_cache[model_key]

    # Find the appropriate model name, with a fallback to multilingual
    model_name = HELSINKI_MODELS.get(model_key, HELSINKI_MODELS["mul_en"])
    if model_key not in HELSINKI_MODELS:
        logger.warning(f"No specific model for {model_key}, falling back to {model_name}.")

    try:
        logger.info(f"Loading Helsinki-NLP model: {model_name}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name).to(device)
        
        _translation_cache[model_key] = (model, tokenizer)
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load translation model {model_name}: {e}")
        _translation_cache[model_key] = (None, None) # Cache failure to avoid retrying
        return None, None

def _translate_batch(texts: List[str], source_lang: str, target_lang: str = "en") -> List[str]:
    """
    Translates a list of texts from a single source language to a target language.
    """
    model, tokenizer = _get_model_for_language_pair(source_lang, target_lang)
    if not model or not tokenizer:
        logger.warning(f"Cannot translate from {source_lang}; model not available.")
        return texts  # Return original texts if model fails to load

    try:
        device = model.device
        batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        translated_ids = model.generate(**batch)
        translated_texts = tokenizer.batch_decode(translated_ids, skip_special_tokens=True)
        return translated_texts
    except Exception as e:
        logger.error(f"Error during batch translation from {source_lang}: {e}")
        return texts # Return original texts on error

def translate_segments_to_english(segments: List[Dict]) -> None:
    """
    Translates a list of transcribed segments in-place.
    Groups segments by language and translates each group as a batch.
    """
    # Group segments by source language
    lang_groups = {}
    for i, seg in enumerate(segments):
        lang = seg.get("language")
        # Only translate non-English text with a valid transcript
        if lang and lang != "en" and seg.get("transcript"):
            if lang not in lang_groups:
                lang_groups[lang] = []
            lang_groups[lang].append({"index": i, "text": seg["transcript"]})

    logger.info(f"Found text to translate in languages: {list(lang_groups.keys())}")

    # Translate each language group in a batch
    for lang, items in lang_groups.items():
        texts_to_translate = [item["text"] for item in items]
        translated_texts = _translate_batch(texts_to_translate, lang, "en")

        # Place translated texts back into the original segments list
        for i, item in enumerate(items):
            original_index = item["index"]
            segments[original_index]["translation"] = translated_texts[i]

def translate_to_english(text: str, source_lang: Optional[str]) -> str:
    """
    Translates a single block of text to English. Used for the full transcript.
    """
    if not text or not source_lang or source_lang == "en" or source_lang == "unknown":
        return text

    # The _translate_batch function expects a list
    translated_chunks = _translate_batch([text], source_lang)
    return translated_chunks[0] if translated_chunks else text
