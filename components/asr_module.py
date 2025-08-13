# asr_module.py

import whisper
import torch
import librosa
import numpy as np

# --- Model Loading ---
# This part runs only once when the module is first imported.
print("Loading ASR model...")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ASR_MODEL = whisper.load_model("base", device=DEVICE)
print(f"ASR model loaded on device: {DEVICE}")

def transcribe_diarized_segments(audio_path: str, diarization_output: list) -> list:
    """
    Transcribes diarized audio segments using Whisper.

    Args:
        audio_path (str): The full path to the input audio file.
        diarization_output (list): A list of segment dictionaries from the diarization module.

    Returns:
        list: An enriched list of dictionaries with transcription data.
    """
    try:
        # librosa.load handles opening, decoding, and resampling to 16kHz all in one step.
        # This is much more robust than the previous method.
        audio_data, _ = librosa.load(audio_path, sr=16000, mono=True)
    except Exception as e:
        print(f"Error loading audio file with librosa: {e}")
        return []

    transcribed_segments = []

    print(f"Starting transcription for {len(diarization_output)} segments...")
    for i, segment in enumerate(diarization_output):
        # Convert start/end times from seconds to sample indices
        start_sample = int(float(segment['start_time']) * 16000)
        end_sample = int(float(segment['end_time']) * 16000)
        
        # Slice the audio data using numpy slicing
        segment_audio = audio_data[start_sample:end_sample]

        # The data is already in the correct format (NumPy array), so we pass it directly
        result = ASR_MODEL.transcribe(
            segment_audio,
            fp16=torch.cuda.is_available()
        )
        
        transcribed_text = result.get('text', '').strip()
        
        if not transcribed_text:
            final_text = "[unintelligible]"
            language = "unknown"
            confidence = -1.0
        else:
            final_text = transcribed_text
            language = result.get('language', 'unknown')
            confidence = result.get('avg_logprob', -1.0)

        enriched_segment = {
            "speaker": segment['speaker'],
            "start_time": segment['start_time'],
            "end_time": segment['end_time'],
            "language_code": language,
            "transcription": final_text,
            "confidence": round(confidence, 3)
        }
        transcribed_segments.append(enriched_segment)
        print(f"  Processed segment {i+1}/{len(diarization_output)}: Speaker {segment['speaker']} -> '{final_text[:50]}...'")

    print("Transcription complete.")
    return transcribed_segments