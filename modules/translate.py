"""
Translation module using MarianMT.
"""
from typing import Optional
from transformers import MarianMTModel, MarianTokenizer

def translate_to_english(text: str, source_lang: str) -> Optional[str]:
    """
    Translate text to English using MarianMT.
    """
    model_name = f"Helsinki-NLP/opus-mt-{source_lang}-en"
    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
        tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
        return tgt_text[0]
    except Exception:
        return None
