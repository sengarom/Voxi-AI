# combined_asr_script.py

import whisper
import torch
import librosa
import numpy as np
from pprint import pprint

# --- Model Loading ---
# This part runs only once when the module is first imported or the script is run.
print("Loading ASR model...")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ASR_MODEL = whisper.load_model("base", device=DEVICE)
print(f"ASR model loaded on device: {DEVICE}")

def detect_language(audio_16k: np.ndarray) -> tuple[str, float]:
    """
    Detect language for a mono, 16 kHz float32 waveform using Whisper's detector.
    Uses log-mel spectrogram as input (expected by Whisper) for robust behavior.
    Returns (ISO 639-1 code, confidence) or ("unknown", 0.0) on failure.
    """
    try:
        # Ensure mono float32 16k 1D
        audio_np = np.asarray(audio_16k)
        if audio_np.ndim != 1:
            audio_np = audio_np.reshape(-1)
        # Replace NaN/Inf, ensure float32
        audio_np = np.nan_to_num(audio_np, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        # Guard: need some audio samples
        if audio_np.size < int(0.25 * 16000):  # < 0.25s is too short for detection
            return "unknown", 0.0
        # Pad/trim to 30s for stable mel shape (as Whisper expects)
        audio_t = torch.from_numpy(audio_np)
        audio_t = whisper.pad_or_trim(audio_t)
        # Build log-mel spectrogram expected by Whisper
        mel = whisper.log_mel_spectrogram(audio_t)
        # Run language detection
        detected_lang, probs = ASR_MODEL.detect_language(mel.to(DEVICE))
        # 'probs' might be a dict (most versions) or a sequence; handle both
        if isinstance(probs, dict):
            lang = max(probs, key=probs.get)
            conf = float(probs.get(lang, 0.0))
        else:
            # Fallback: use detected_lang and best-effort confidence
            lang = detected_lang if isinstance(detected_lang, str) else "unknown"
            try:
                conf = float(max(probs)) if hasattr(probs, '__iter__') else 0.0
            except Exception:
                conf = 0.0
        return lang, conf
    except Exception as e:
        print(f"Language detection failed: {e}")
        return "unknown", 0.0

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

    # Detect language from a longer context (up to first 60s) as a robust fallback
    try:
        max_ctx_seconds = 60
        ctx_samples = min(len(audio_data), int(16000 * max_ctx_seconds))
        fallback_lang, fallback_conf = detect_language(audio_data[:ctx_samples])
    except Exception:
        fallback_lang, fallback_conf = ("unknown", 0.0)

    print(f"Starting transcription for {len(diarization_output)} segments...")
    for i, segment in enumerate(diarization_output):
        # Convert start/end times from seconds to sample indices
        # Clamp boundaries and ensure valid ordering
        start_sample = max(0, int(float(segment['start_time']) * 16000))
        end_sample = max(start_sample + 1, int(float(segment['end_time']) * 16000))

        # Slice the audio data using numpy slicing
        segment_audio = audio_data[start_sample:end_sample]
        if segment_audio.size == 0:
            # Fallback: extend a tiny window if end == start
            end_sample = min(len(audio_data), start_sample + int(0.5 * 16000))
            segment_audio = audio_data[start_sample:end_sample]
        # Ensure float32 mono at 16k (librosa.load already mono,16k)
        segment_audio = segment_audio.astype(np.float32, copy=False)

        # NOTE: For very short segments, language detection may be unreliable.
        # Skip detection for segments <0.6s and use fallback
        if (end_sample - start_sample) < int(0.6 * 16000):
            lang_code, lang_conf = ("unknown", 0.0)
        else:
            lang_code, lang_conf = detect_language(segment_audio)

        # Always pass raw 1D float32 audio to Whisper; it will compute mel internally
        input_for_whisper = segment_audio

        # Choose a stable language hint: prefer segment detection if confident, else file-level
        if lang_code != 'unknown' and lang_conf >= 0.5:
            chosen_lang = lang_code
        elif fallback_lang != 'unknown' and fallback_conf >= 0.5:
            chosen_lang = fallback_lang
        else:
            # As last resort, keep the segment guess even if low
            chosen_lang = lang_code if lang_code != 'unknown' else fallback_lang

        # Build kwargs, only pass language if we have a confident hint; otherwise let Whisper autodetect
        transcribe_kwargs = {
            'fp16': torch.cuda.is_available()
        }
        if chosen_lang and chosen_lang != 'unknown':
            # Prefer segment detection when confident
            if lang_code != 'unknown' and lang_conf >= 0.5:
                transcribe_kwargs['language'] = lang_code
            # Otherwise use file-level fallback if confident
            elif fallback_lang != 'unknown' and fallback_conf >= 0.5:
                transcribe_kwargs['language'] = fallback_lang
            # else: do not pass language and allow autodetection

        result = ASR_MODEL.transcribe(
            input_for_whisper,
            task="transcribe",
            **transcribe_kwargs,
        )

        transcribed_text = result.get('text', '').strip()

        # Decide final language for this segment
        # Prefer confident segment detection; else confident file-level; else Whisper's autodetected language
        language = None
        if lang_code != 'unknown' and lang_conf >= 0.5:
            language = lang_code
        elif fallback_lang != 'unknown' and fallback_conf >= 0.5:
            language = fallback_lang
        else:
            language = result.get('language', 'unknown')

        if not transcribed_text:
            final_text = "[unintelligible]"
            language = "unknown"
            confidence = -1.0
        else:
            final_text = transcribed_text
            # Prefer explicitly detected language; fallback to ASR result if needed
            if not language or language == 'unknown':
                language = result.get('language', 'unknown')
            # Whisper result object doesn't always contain 'avg_logprob'.
            # It's more reliable to check for segment-level confidence if available,
            # but for a whole-segment transcription, we'll stick with this.
            confidence = result.get('avg_logprob', -1.0)

        enriched_segment = {
            "speaker": segment['speaker'],
            "start_time": segment['start_time'],
            "end_time": segment['end_time'],
            "language_code": language,
            "language": language,
            "transcription": final_text,
            "confidence": round(confidence, 3) if confidence is not None else -1.0
        }
        transcribed_segments.append(enriched_segment)
        print(f"  Processed segment {i+1}/{len(diarization_output)}: Speaker {segment['speaker']} -> '{final_text[:50]}...'")

    print("Transcription complete.")
    return transcribed_segments


def run_test():
    """
    A simple function to test our ASR module with mock data.
    """
    print("\n--- Starting ASR Module Standalone Test ---")

    # This is mock output, like the data that would be provided by the diarization module.
    mock_diarization_data = [
        {'speaker': 'SPEAKER_00', 'start_time': '0.00', 'end_time': '10.00'},
        # You could add more segments here to test further
        # {'speaker': 'SPEAKER_01', 'start_time': '10.50', 'end_time': '15.00'},
    ]

    # You must have a sample audio file named 'sample_audio.mp3' in the same directory.
    # Create a dummy one if you don't have it for testing purposes.
    mock_audio_file = 'sample_audio.mp3'
    
    try:
        # Check if the mock audio file exists before running
        with open(mock_audio_file, 'rb') as f:
            pass
        
        # This is the function call the Django app will eventually make.
        final_output = transcribe_diarized_segments(
            audio_path=mock_audio_file,
            diarization_output=mock_diarization_data
        )

        print("\n--- Final Enriched Output ---")
        if final_output:
            pprint(final_output)
        else:
            print("Processing failed. Please check for errors above.")
            
    except FileNotFoundError:
        print(f"\nERROR: The test audio file '{mock_audio_file}' was not found.")
        print("Please place a valid audio file with this name in the same directory to run the test.")


# This line ensures the test runs only when you execute this file directly.
if __name__ == '__main__':
    run_test()
