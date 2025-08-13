"""
Language detection module using Whisper to detect spoken language from audio.
"""
import logging
import numpy as np

def detect_language_from_audio(segment_audio: np.ndarray, sample_rate: int) -> str:
    """
    Detect language from audio using Whisper's built-in language detection.
    Parameters:
        segment_audio (np.ndarray): Audio samples. Shape (channels, samples) or (samples,).
        sample_rate (int): Sampling rate of the input audio.
    Returns:
        str: ISO 639-1 language code (e.g., 'en', 'fr'). Returns 'unknown' on failure.
    """
    try:
        # Lazy import to prevent module import-time crashes when Whisper isn't available
        import whisper
        import torch
        model = whisper.load_model("base")
        # Ensure mono
        if segment_audio.ndim == 2:
            audio_mono = segment_audio.mean(axis=0)
        else:
            audio_mono = segment_audio
        # Resample to 16 kHz if needed
        if sample_rate != 16000:
            import librosa
            audio_16k = librosa.resample(audio_mono, orig_sr=sample_rate, target_sr=16000)
        else:
            audio_16k = audio_mono
        audio_16k = audio_16k.astype(np.float32)
        audio_tensor = torch.from_numpy(audio_16k)
        _, probs = model.detect_language(audio_tensor)
        return max(probs, key=probs.get)
    except Exception as e:
        logging.warning(f"detect_language_from_audio: Whisper language detection failed: {e}")
        return "unknown"
