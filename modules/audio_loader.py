"""
Audio loading and normalization module.
"""
from typing import Dict, Any
from pydub import AudioSegment
import numpy as np

def load_audio_file(file_path: str) -> Dict[str, Any]:
    """
    Load an audio file (wav, mp3, etc.) and return a dict with:
      - 'audio': np.ndarray (shape: [channels, samples])
      - 'sample_rate': int
      - 'channels': int
      - 'duration': float (seconds)
    Audio is normalized to -1.0 to 1.0 float32.
    """
    audio = AudioSegment.from_file(file_path)
    change_in_dBFS = -1.0 - audio.max_dBFS
    audio = audio.apply_gain(change_in_dBFS)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    if audio.channels == 2:
        samples = samples.reshape((-1, 2)).T
    else:
        samples = samples[np.newaxis, :]
    samples /= np.iinfo(audio.array_type).max
    return {
        'audio': samples,
        'sample_rate': audio.frame_rate,
        'channels': audio.channels,
        'duration': len(audio) / 1000.0
    }
