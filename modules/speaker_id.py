"""
Speaker identification module using SpeechBrain ECAPA.
"""
from typing import Dict, Any
import numpy as np
from speechbrain.pretrained import EncoderClassifier
from scipy.spatial.distance import cosine

REFERENCE_SPEAKERS = {
    # Fill with real speaker names and their embedding vectors (np.ndarray)
}

_sb_classifier = None

def get_speechbrain_classifier():
    global _sb_classifier
    if _sb_classifier is None:
        _sb_classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    return _sb_classifier

def identify_speaker(segment_audio: np.ndarray) -> str:
    classifier = get_speechbrain_classifier()
    import torch
    if segment_audio.shape[0] > 1:
        audio_mono = np.mean(segment_audio, axis=0)
    else:
        audio_mono = segment_audio[0]
    waveform = torch.tensor(audio_mono).unsqueeze(0)
    embedding = classifier.encode_batch(waveform).detach().cpu().numpy()[0, 0, :]
    best_name = "Unknown"
    best_score = 1.0
    for name, ref_emb in REFERENCE_SPEAKERS.items():
        score = cosine(embedding, ref_emb)
        if score < best_score:
            best_score = score
            best_name = name
    if best_score > 0.6:
        return "Unknown"
    return best_name
