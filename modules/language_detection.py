"""
Language detection module using langdetect or fasttext.
"""
def detect_language_from_text(text: str) -> str:

from typing import Any
import numpy as np
import whisper
import torch
from pydub import AudioSegment

def detect_language_from_audio(segment_audio: np.ndarray, sample_rate: int) -> str:
    """
    Detect language from audio using Whisper's language detection.
    segment_audio: np.ndarray (mono or stereo)
    sample_rate: int
    Returns: ISO 639-1 language code (e.g., 'en', 'fr', ...)
    """
    model = whisper.load_model("base")
    # Convert to mono if needed
    if segment_audio.ndim > 1:
        audio_mono = np.mean(segment_audio, axis=0)
    else:
        audio_mono = segment_audio
    # Resample to 16kHz if needed
    if sample_rate != 16000:
        import librosa
        audio_16k = librosa.resample(audio_mono, orig_sr=sample_rate, target_sr=16000)
    else:
        audio_16k = audio_mono
    audio_16k = audio_16k.astype(np.float32)
    # Whisper expects a torch tensor
    audio_tensor = torch.from_numpy(audio_16k)
    # Use Whisper's language detection
    _, probs = model.detect_language(audio_tensor)
    lang = max(probs, key=probs.get)
    return lang
