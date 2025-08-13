"""
Speaker diarization module using pyannote.audio.
Provides run_speaker_diarization(audio_data) -> List[segments].
"""
from typing import List, Dict, Any
import os
import logging
import numpy as np
import torch
from pyannote.audio import Pipeline
import warnings

# Suppress torchaudio backend deprecation warning emitted by dependencies
warnings.filterwarnings(
    "ignore",
    message=r".*torchaudio\._backend\.list_audio_backends has been deprecated.*",
    category=UserWarning,
)

_pipeline = None

def _get_hf_token() -> str:
    """
    Get Hugging Face token from environment or local files.
    Checks env vars: HUGGINGFACE_TOKEN, HF_TOKEN, PYANNOTE_TOKEN.
    Then tries project-root files: hf_token.txt, .hf_token (single-line token).
    """
    token = (
        os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("PYANNOTE_TOKEN")
    )
    if token:
        return token.strip()
    # Try local files
    base_dir = os.path.dirname(os.path.dirname(__file__))
    for fname in ("hf_token.txt", ".hf_token"):
        fpath = os.path.join(base_dir, fname)
        if os.path.exists(fpath):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        return content
            except Exception as e:
                logging.warning(f"_get_hf_token: Failed reading {fpath}: {e}")
    return ""

def _get_pipeline() -> Pipeline:
    """Lazily load the pyannote diarization pipeline."""
    global _pipeline
    if _pipeline is None:
        token = _get_hf_token()
        try:
            if not token:
                msg = (
                    "Missing Hugging Face token for pyannote. Set HUGGINGFACE_TOKEN/HF_TOKEN/PYANNOTE_TOKEN "
                    "or create 'hf_token.txt' in the project root with your token."
                )
                logging.error(f"_get_pipeline: {msg}")
                raise RuntimeError(msg)
            _pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=token)
        except Exception as e:
            logging.error(f"run_speaker_diarization: Failed to load pyannote pipeline: {e}")
            raise
    return _pipeline

def run_speaker_diarization(audio_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Perform speaker diarization on provided audio data and return segments.

    Args:
        audio_data (dict): {
            'audio': np.ndarray (shape [channels, samples]),
            'sample_rate': int,
            'channels': int,
            'duration': float
        }

    Returns:
        List[dict]: Each dict has keys: 'speaker' (str), 'start' (float), 'end' (float), 'audio' (np.ndarray segment)
    """
    try:
        audio = audio_data["audio"]
        sr: int = int(audio_data["sample_rate"]) 
    except Exception as e:
        logging.error(f"run_speaker_diarization: Invalid audio_data input: {e}")
        raise

    # Normalize to torch tensor [channels, time]
    if isinstance(audio, np.ndarray):
        audio_t = torch.from_numpy(audio)
    elif isinstance(audio, torch.Tensor):
        audio_t = audio
    else:
        raise ValueError("audio_data['audio'] must be numpy array or torch.Tensor")

    if audio_t.ndim == 1:
        # [time] -> [1, time]
        audio_t = audio_t.unsqueeze(0)
    elif audio_t.ndim == 2:
        # Heuristic: if shape looks like [time, channels] (i.e., much longer first dim
        # and second dim small), transpose to [channels, time]
        if audio_t.shape[0] > audio_t.shape[1] and audio_t.shape[1] <= 8:
            audio_t = audio_t.transpose(0, 1)
    else:
        raise ValueError("audio tensor must be 1D or 2D")

    # Ensure float32 and contiguous
    waveform = audio_t.to(torch.float32).contiguous()
    logging.info(f"run_speaker_diarization: waveform shape={tuple(waveform.shape)}, dtype={waveform.dtype}, sr={sr}")
    # Keep a numpy copy in channel-first for slicing
    samples_cf = waveform.detach().cpu().numpy()  # [channels, time]

    pipeline = _get_pipeline()
    try:
        annotation = pipeline({"waveform": waveform, "sample_rate": sr})
    except Exception as e:
        logging.error(f"run_speaker_diarization: Pipeline execution failed: {e}")
        raise

    segments: List[Dict[str, Any]] = []
    # Iterate segments with labels if available
    try:
        for turn, _, label in annotation.itertracks(yield_label=True):
            start = float(turn.start)
            end = float(turn.end)
            if end <= start:
                continue
            start_idx = max(0, int(start * sr))
            end_idx = max(start_idx, int(end * sr))
            end_idx = min(samples_cf.shape[1], end_idx)

            # --- ASR compatibility: always output mono float32 [1, samples] ---
            # If input is multi-channel, convert to mono by averaging channels.
            seg_audio = samples_cf[:, start_idx:end_idx].astype(np.float32)
            if seg_audio.ndim == 1:
                seg_audio = seg_audio[np.newaxis, :]
            if seg_audio.shape[0] > 1:
                seg_audio = np.mean(seg_audio, axis=0, keepdims=True)

            # seg_audio is now always shape [1, samples], float32, mono

            segments.append({
                "speaker": label or "UNK",
                "start": start,  # float seconds
                "end": end,      # float seconds
                "audio": seg_audio,  # mono float32 [1, samples] for ASR
            })
    except Exception as e:
        logging.error(f"run_speaker_diarization: Failed to parse annotation: {e}")
        raise

    return segments