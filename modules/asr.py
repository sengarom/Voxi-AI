# Revised and Fixed asr.py
import os
import logging
import numpy as np
import torch
import whisper
from pydub import AudioSegment
from typing import Dict, Any, Optional

logger = logging.getLogger("VoxiAPI")

# --- Model Management ---
_whisper_model: Optional[whisper.Whisper] = None

def get_whisper_model(model_name: str = "medium") -> whisper.Whisper:
    """
    Loads the Whisper model lazily and caches it.
    Uses the 'medium' model by default for a good balance of accuracy and speed.
    """
    global _whisper_model
    if _whisper_model is None:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading Whisper '{model_name}' model on '{device}' device...")
            _whisper_model = whisper.load_model(model_name, device=device)
            logger.info(f"Whisper '{model_name}' model loaded successfully.")
        except Exception as e:
            logger.error(f"Fatal error: Failed to load Whisper model: {e}")
            raise  # Re-raise the exception to halt execution if the model can't load
    return _whisper_model

# --- Audio Processing ---
def _load_and_prepare_audio(
    audio_path: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> np.ndarray:
    """
    A single, unified function to load, segment, and prepare audio for Whisper.
    Converts audio to mono, 16kHz, and float32 format as required by Whisper.
    """
    try:
        audio = AudioSegment.from_file(audio_path)

        # 1. Extract segment if start and end times are provided
        if start_time is not None and end_time is not None:
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)
            audio = audio[start_ms:end_ms]

        # 2. Resample to 16kHz (Whisper's required sample rate)
        if audio.frame_rate != 16000:
            audio = audio.set_frame_rate(16000)

        # 3. Convert to mono
        if audio.channels > 1:
            audio = audio.set_channels(1)

        # 4. Convert to float32 numpy array and normalize to [-1.0, 1.0]
        # pydub stores samples as signed 16-bit integers (from -32768 to 32767)
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        samples /= 32768.0

        return samples

    except Exception as e:
        logger.error(f"Failed to load or process audio file '{audio_path}': {e}")
        return np.array([], dtype=np.float32) # Return an empty array on failure

# --- Core ASR Functions ---
def transcribe_audio_segment(audio_data: np.ndarray) -> Dict[str, Any]:
    """
    Transcribes a single audio segment (as a numpy array) using Whisper.
    This function now expects a pre-processed numpy array.
    """
    if not isinstance(audio_data, np.ndarray) or audio_data.size == 0:
        logger.warning("Received empty or invalid audio data for transcription.")
        return {"text": "", "language": "unknown", "avg_logprob": -1.0}

    try:
        model = get_whisper_model()

        # Set transcription options.
        # verbose=None will show progress bars for long files.
        transcribe_options = {
            "task": "transcribe",
            "fp16": torch.cuda.is_available(),
            "verbose": None,
            "word_timestamps": False  # Set to True if you need word-level timings
        }

        # Let Whisper handle audio length; no need to manually truncate.
        result = model.transcribe(audio_data, **transcribe_options)

        # Calculate an average confidence score (log probability)
        # Note: This is a pseudo-confidence score. Higher is better.
        avg_logprob = result.get("avg_logprob", -1.0)
        if "segments" in result and result["segments"]:
            logprobs = [s['avg_logprob'] for s in result['segments'] if 'avg_logprob' in s]
            if logprobs:
                avg_logprob = sum(logprobs) / len(logprobs)

        return {
            "text": result.get("text", "").strip(),
            "language": result.get("language", "unknown"),
            "avg_logprob": avg_logprob,
        }
    except Exception as e:
        logger.error(f"Whisper transcription failed for a segment: {e}")
        return {"text": "[Transcription Error]", "language": "unknown", "avg_logprob": -1.0}


def transcribe_diarized_segments(audio_path: str, diarization_output: list) -> list:
    """
    Transcribes a list of diarized segments from an audio file.
    """
    results = []
    logger.info(f"Starting Whisper transcription for {len(diarization_output)} diarized segments...")

    for i, segment_info in enumerate(diarization_output):
        start = segment_info.get("start")
        end = segment_info.get("end")
        speaker = segment_info.get("speaker", "UNK")

        # Load and prepare just the audio for this specific segment
        audio_segment_data = _load_and_prepare_audio(audio_path, start_time=start, end_time=end)

        if audio_segment_data.size == 0:
            logger.warning(f"Skipping empty audio segment {i+1} for speaker {speaker}")
            continue

        # Perform transcription on the prepared segment
        asr_result = transcribe_audio_segment(audio_segment_data)

        enriched_segment = {
            "speaker": speaker,
            "start_time": start,
            "end_time": end,
            "transcription": asr_result["text"],
            "language": asr_result["language"],
            "confidence": asr_result["avg_logprob"], # Using avg_logprob as confidence
        }
        results.append(enriched_segment)
        logger.info(f"Processed segment {i+1}/{len(diarization_output)}: Speaker {speaker} -> '{asr_result['text'][:60]}...'")

    logger.info("Whisper transcription for all diarized segments completed.")
    return results


def translate_audio_to_english(audio_path: str) -> str:
    """
    Translates the entire audio content of a file directly to English using Whisper.
    """
    try:
        model = get_whisper_model()
        logger.info(f"Starting translation for audio file: {audio_path}")

        # Use the unified function to load the entire audio file
        # Whisper will process it in 30-second chunks automatically
        audio_data = _load_and_prepare_audio(audio_path)

        if audio_data.size == 0:
            return "[Translation failed: Could not load audio]"

        translate_options = {
            "task": "translate",
            "fp16": torch.cuda.is_available(),
        }

        result = model.transcribe(audio_data, **translate_options)
        translation = result.get("text", "").strip()
        logger.info(f"Translation successful: '{translation[:80]}...'")
        return translation

    except Exception as e:
        logger.error(f"Audio translation failed: {e}")
        return f"[Translation Error: {e}]"