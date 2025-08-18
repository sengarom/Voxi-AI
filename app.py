
import os
import time
import logging
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from collections import Counter
import torch
import numpy as np
from pydub import AudioSegment

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VoxiAPI")

# --- Module Imports ---
# Assuming these modules exist in a 'modules' directory
from modules import diarization
from modules import asr
from modules import translate

# --- Flask App Configuration ---
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a', 'webm'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB

# Optional CORS support
try:
    from flask_cors import CORS
    CORS(app)
    logger.info("CORS support enabled.")
except ImportError:
    logger.warning("CORS not enabled: `pip install flask_cors` to enable.")

def allowed_file(filename: str) -> bool:
    """Checks if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def _merge_speaker_segments(speakers_data: list) -> list:
    """
    Merges consecutive segments from the same speaker into a single segment.
    """
    if not speakers_data:
        return []

    merged_data = []
    current_segment = speakers_data[0].copy()

    for i in range(1, len(speakers_data)):
        next_segment = speakers_data[i]
        if next_segment['speaker'] == current_segment['speaker']:
            current_segment['end'] = next_segment['end']
            current_segment['transcript'] += ' ' + next_segment['transcript']
        else:
            merged_data.append(current_segment)
            current_segment = next_segment.copy()

    merged_data.append(current_segment)
    return merged_data


@app.route('/', methods=['GET'])
def index():
    """Renders the main UI from the templates folder."""
    return render_template('index.html')

# --- Main Audio Processing Endpoint ---
@app.route('/process_audio', methods=['POST'])
def process_audio_api():
    """
    Main API endpoint to handle audio file upload, diarization, transcription, and translation.
    """
    logger.info("Received request for /process_audio")

    # 1. --- File Validation and Upload ---
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    filename = secure_filename(file.filename)
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        file.save(audio_path)
        logger.info(f"File saved successfully to {audio_path}")
    except Exception as e:
        logger.error(f"Failed to save file: {e}")
        return jsonify({"error": "Failed to save file on server."}), 500

    # 2. --- Speaker Diarization ---
    try:
        logger.info("Loading audio for diarization module...")
        audio_segment = AudioSegment.from_file(audio_path)
        waveform_tensor = torch.from_numpy(
            np.array(audio_segment.get_array_of_samples()).astype(np.float32) / 32768.0
        ).unsqueeze(0)

        audio_for_diarization = {
            'waveform': waveform_tensor,
            'sample_rate': audio_segment.frame_rate
        }

        logger.info("Starting speaker diarization...")
        t0 = time.perf_counter()
        diarized_segments = diarization.run_speaker_diarization(audio_for_diarization)
        logger.info(f"Diarization complete in {time.perf_counter() - t0:.2f}s, found {len(diarized_segments)} segments.")
    except Exception as e:
        logger.error(f"Diarization process failed: {e}")
        os.remove(audio_path)
        return jsonify({"error": f"Diarization failed: {e}"}), 500

    # 3. --- ASR with Whisper ---
    try:
        logger.info("Starting ASR with Whisper...")
        t1 = time.perf_counter()
        asr_results = asr.transcribe_diarized_segments(audio_path, diarized_segments)
        logger.info(f"ASR complete in {time.perf_counter() - t1:.2f}s.")
    except Exception as e:
        logger.error(f"ASR process failed: {e}")
        os.remove(audio_path)
        return jsonify({"error": f"ASR failed: {e}"}), 500

    # 4. --- Process and Combine Initial Results ---
    speakers_data = []
    speaker_label_map = {}
    next_label_ord = ord('A')

    for asr_seg in asr_results:
        raw_speaker_id = asr_seg.get("speaker", "UNK")
        if raw_speaker_id not in speaker_label_map:
            speaker_label_map[raw_speaker_id] = f"Speaker {chr(next_label_ord)}"
            if next_label_ord < ord('Z'):
                next_label_ord += 1

        speakers_data.append({
            "speaker": speaker_label_map[raw_speaker_id],
            "start": asr_seg.get("start_time"),
            "end": asr_seg.get("end_time"),
            "transcript": asr_seg.get("transcription", "").strip(),
            "language": asr_seg.get("language", "unknown"),
            "confidence": asr_seg.get("confidence", -99)
        })

    # 4.5 --- Merge consecutive segments from the same speaker ---
    logger.info(f"Merging {len(speakers_data)} segments into consolidated speaker turns...")
    merged_speakers_data = _merge_speaker_segments(speakers_data)
    logger.info(f"Merged down to {len(merged_speakers_data)} segments.")
    
    # 5. --- Build Full Transcript & Detect Language from Merged Data ---
    full_transcript = " ".join([seg['transcript'] for seg in merged_speakers_data]).strip()
    
    detected_languages = [seg['language'] for seg in merged_speakers_data if seg.get('language') != 'unknown']
    main_language = Counter(detected_languages).most_common(1)[0][0] if detected_languages else "unknown"
    logger.info(f"Determined main language of audio: {main_language}")

    # 6. --- Translation (Optimized) ---
    logger.info("Starting batch translation of segments...")
    t2 = time.perf_counter()
    try:
        # Batch translate all segments in-place using the optimized function
        translate.translate_segments_to_english(merged_speakers_data)
        
        # Translate the full transcript separately
        full_translation = translate.translate_to_english(full_transcript, main_language)
        
        logger.info(f"Translation complete in {time.perf_counter() - t2:.2f}s.")
    except Exception as e:
        logger.error(f"Translation process failed: {e}")
        full_translation = f"[Translation failed: {e}]"

    # 7. --- Final Response ---
    response = {
        "speakers": merged_speakers_data,
        "transcript": full_transcript,
        "language": main_language,
        "translation": full_translation
    }
    
    try:
        os.remove(audio_path)
        logger.info(f"Cleaned up temporary file: {audio_path}")
    except OSError as e:
        logger.error(f"Error removing file {audio_path}: {e}")

    logger.info("Request processing complete. Sending response.")
    return jsonify(response)

@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(e: RequestEntityTooLarge):
    """Handles file uploads that exceed the configured size limit."""
    logger.warning(f"Upload rejected: file too large (limit: {app.config['MAX_CONTENT_LENGTH']} bytes).")
    return jsonify({'error': f'File is too large. Maximum size is {app.config["MAX_CONTENT_LENGTH"] // 1024 // 1024} MB.'}), 413

if __name__ == '__main__':
    # Setting debug=False is recommended for any production/public-facing use
    app.run(debug=False, host='0.0.0.0', port=5000)
