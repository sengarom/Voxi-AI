"""
ASR module using OpenAI Whisper.
"""
from typing import Any
import numpy as np
import whisper

def run_asr_on_segment(segment_audio: np.ndarray, sample_rate: int, language: str = None) -> str:
    """
    Run ASR on audio segment using Whisper.
    """
    model = whisper.load_model("base")
    # Whisper expects mono, 16kHz float32 numpy array
    import librosa
    audio_16k = librosa.resample(segment_audio, orig_sr=sample_rate, target_sr=16000)
    result = model.transcribe(audio_16k, language=language)
    return result['text']
