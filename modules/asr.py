"""
ASR module using OpenAI Whisper.
"""
from typing import Any
import numpy as np

def run_asr_on_segment(segment_audio: np.ndarray, sample_rate: int, language: str = None) -> str:
    """
    Run ASR on audio segment using Whisper.
    """
    # Lazy import to avoid import-time crashes when Whisper isn't available
    import whisper
    # Whisper expects mono, 16kHz float32 numpy array
    import librosa
    # Ensure mono
    if segment_audio.ndim == 2:
        mono = segment_audio.mean(axis=0)
    else:
        mono = segment_audio
    audio_16k = librosa.resample(mono, orig_sr=sample_rate, target_sr=16000)
    model = whisper.load_model("base")
    result = model.transcribe(audio_16k, language=language)
    return result['text']
