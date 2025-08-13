# asr_module.py
# Author: Kushaan Aggarwal
# Handles Language Identification (LID) and ASR for the Voxi project.

import whisper
import torch
from pydub import AudioSegment
import numpy as np
from pprint import pprint

# --- Model Loading ---
# Load the model once to be efficient. We'll use the 'base' model for a good
# balance of speed and accuracy. The device is set to GPU if available.
print("Loading ASR model...")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ASR_MODEL = whisper.load_model("base", device=DEVICE)
print(f"ASR model loaded on device: {DEVICE}")

def transcribe_diarized_segments(audio_path: str, diarization_output: list) -> list:
    """
    Transcribes diarized audio segments using Whisper.

    This function takes the path to an audio file and a list of diarized segments,
    then returns an enriched list containing the transcription and language for each segment.

    Args:
        audio_path (str): The full path to the input audio file (e.g., 'path/to/audio.wav').
        diarization_output (list): A list of dictionaries from the diarization module.
                                  Each dict must have 'speaker', 'start_time', and 'end_time'.

    Returns:
        list: A list of dictionaries, with each one enriched with transcription,
              language, and confidence scores. Returns an empty list if an error occurs.
    """
    try:
        # Load the main audio file using pydub. It handles various formats automatically.
        audio = AudioSegment.from_file(audio_path)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return []

    transcribed_segments = []

    print(f"Starting transcription for {len(diarization_output)} segments...")
    for i, segment in enumerate(diarization_output):
        # pydub works in milliseconds, so we convert the times from the diarizer.
        start_ms = float(segment['start_time']) * 1000
        end_ms = float(segment['end_time']) * 1000
        
        # Slice the audio to get the specific segment.
        segment_audio = audio[start_ms:end_ms]

        # Whisper requires a 16kHz mono NumPy array.
        # We'll set frame rate to 16000Hz, channels to 1 (mono).
        segment_audio = segment_audio.set_frame_rate(16000).set_channels(1)
        
        # Convert pydub segment to NumPy array.
        samples = np.array(segment_audio.get_array_of_samples()).astype(np.float32) / 32768.0

        # Run transcription on the audio segment.
        result = ASR_MODEL.transcribe(
            samples,
            fp16=torch.cuda.is_available() # Use fp16 for faster GPU inference
        )

        # Get confidence score (avg_logprob is a good proxy).
        # Fallback for very short segments where it might not be calculated.
        confidence = result.get('avg_logprob', -1.0)
        
        # Apply the low-confidence fallback policy.
        # If confidence is too low, mark as unintelligible.
        if confidence < -1.0:
            final_text = "[unintelligible]"
            language = "unknown"
        else:
            final_text = result.get('text', '').strip()
            language = result.get('language', 'unknown')

        # Create the enriched data object for this segment.
        enriched_segment = {
            "speaker": segment['speaker'],
            "start_time": segment['start_time'],
            "end_time": segment['end_time'],
            "language_code": language,
            "transcription": final_text,
            "confidence": round(confidence, 3) # Round for cleaner output
        }
        transcribed_segments.append(enriched_segment)
        print(f"  Processed segment {i+1}/{len(diarization_output)}: Speaker {segment['speaker']} -> '{final_text[:50]}...'")

    print("Transcription complete.")
    return transcribed_segments


# --- Example Usage ---
# The Django developer can use this block to understand how to call the function.
if __name__ == '__main__':
    # This is a mock output, as would be provided by Soumya's diarization module.
    mock_diarization_data = [
        {'speaker': 'SPK_00', 'start_time': '0.78', 'end_time': '3.45'},
        {'speaker': 'SPK_01', 'start_time': '3.90', 'end_time': '6.82'},
        {'speaker': 'SPK_00', 'start_time': '7.10', 'end_time': '10.25'}
    ]

    # You must have a sample audio file named 'sample_audio.mp3' in the same directory.
    # You can download a sample from here: https://filesamples.com/samples/audio/mp3/sample-audio-15-seconds.mp3
    # and rename it.
    mock_audio_file = 'sample_audio.mp3'

    print("--- Running ASR Module Standalone Test ---")
    
    # This is the function call the Django app will make.
    final_output = transcribe_diarized_segments(
        audio_path=mock_audio_file,
        diarization_output=mock_diarization_data
    )

    print("\n--- Final Enriched Output ---")
    if final_output:
        pprint(final_output)
    else:
        print("Processing failed. Please check for errors above.")