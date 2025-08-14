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

logger = logging.getLogger("VoxiAPI")
_pipeline = None

def _get_hf_token() -> str:
    """Gets the Hugging Face token from environment variables or local files."""
    token = (
        os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("PYANNOTE_TOKEN")
    )
    if token:
        return token.strip()
    
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
    """Lazily loads the pyannote diarization pipeline."""
    global _pipeline
    if _pipeline is None:
        token = _get_hf_token()
        if not token:
            msg = (
                "Missing Hugging Face token for pyannote. Set HUGGINGFACE_TOKEN, HF_TOKEN, or PYANNOTE_TOKEN "
                "or create 'hf_token.txt' in the project root with your token."
            )
            logging.error(f"_get_pipeline: {msg}")
            raise RuntimeError(msg)
        try:
            _pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
            # Move pipeline to GPU if available
            if torch.cuda.is_available():
                _pipeline.to(torch.device("cuda"))
                logging.info("Moved pyannote pipeline to GPU.")
        except Exception as e:
            logging.error(f"Failed to load pyannote pipeline: {e}")
            raise
    return _pipeline

def run_speaker_diarization(audio_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Performs speaker diarization on provided audio data and returns segments.
    """
    try:
        # **FIX:** Changed key from "audio" to "waveform" to match what app.py provides.
        waveform = audio_data["waveform"]
        sr: int = int(audio_data["sample_rate"])
    except KeyError as e:
        logging.error(f"run_speaker_diarization: Invalid audio_data input. Missing key: {e}")
        raise

    # The pyannote pipeline expects a dictionary with these keys
    pipeline_input = {"waveform": waveform, "sample_rate": sr}
    
    pipeline = _get_pipeline()
    try:
        logging.info(f"Running pyannote pipeline on waveform with shape: {waveform.shape}")
        diarization_result = pipeline(pipeline_input)
    except Exception as e:
        logging.error(f"run_speaker_diarization: Pipeline execution failed: {e}")
        raise

    segments: List[Dict[str, Any]] = []
    # Keep a numpy copy for efficient slicing
    samples_np = waveform.squeeze(0).detach().cpu().numpy()

    for turn, _, speaker_label in diarization_result.itertracks(yield_label=True):
        start = float(turn.start)
        end = float(turn.end)
        
        if end <= start:
            continue
            
        start_idx = int(start * sr)
        end_idx = int(end * sr)
        
        # Slice the numpy array to get the audio for this segment
        segment_audio_np = samples_np[start_idx:end_idx]

        segments.append({
            "speaker": speaker_label,
            "start": start,
            "end": end,
            # The ASR module expects a numpy array, so we pass that directly.
            # No need to include the audio in the return dictionary if asr.py re-loads it,
            # but it can be useful for other potential modules.
            # "audio": segment_audio_np
        })
        
    return segments