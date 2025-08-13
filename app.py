

import os
import time
import logging
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("VoxiAPI")

# Optional CORS support
try:
    from flask_cors import CORS
except ImportError:
    CORS = None
    logging.warning('CORS not enabled: flask_cors is not installed.')

from modules import audio_loader, diarization, asr, translate

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB
if CORS:
    CORS(app)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Minimal HTML upload form for manual testing
@app.route('/', methods=['GET'])
def index():
    return '''
    <html>
    <body>
        <h2>Upload audio file for diarization, ASR, and translation</h2>
        <form action="/process_audio" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="audio/*" required>
            <button type="submit">Upload</button>
        </form>
    </body>
    </html>
    '''

# --- /process_audio API endpoint ---
@app.route('/process_audio', methods=['POST'])
def process_audio_api():
    logger.info("Received /process_audio request")
    if 'file' not in request.files:
        logger.warning("No file part in request")
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        logger.warning("No selected file")
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        logger.warning(f"Unsupported file type: {file.filename}")
        return jsonify({"error": "Unsupported file type"}), 400

    filename = secure_filename(file.filename)
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(temp_path)
    logger.info(f"Saved file to {temp_path}")

    # 1. Load audio
    audio_data = audio_loader.load_audio_file(temp_path)
    logger.info("Audio loaded")

    # --- Ensure audio is a (channels, samples) torch.Tensor ---
    import numpy as np
    import torch
    audio_arr = audio_data['audio']
    if isinstance(audio_arr, np.ndarray):
        if audio_arr.ndim == 1:
            audio_arr = np.expand_dims(audio_arr, axis=0)
        audio_arr = torch.from_numpy(audio_arr)
    if isinstance(audio_arr, torch.Tensor) and audio_arr.ndim == 1:
        audio_arr = audio_arr.unsqueeze(0)
    # Do not transpose to time-first here; diarization handles layout
    audio_data['audio'] = audio_arr
    # -------------------------------------------------------------

    # 2. Speaker diarization
    logger.info("Starting diarization...")
    _t0 = time.perf_counter()
    diarized_segments = diarization.run_speaker_diarization(audio_data)
    logger.info(f"Diarization complete: {len(diarized_segments)} segments in {time.perf_counter()-_t0:.2f}s")

    # --- Ensure each segment's audio is mono 1D torch.Tensor for ASR ---
    for seg in diarized_segments:
        audio = seg['audio']
        if isinstance(audio, torch.Tensor):
            arr = audio
        else:
            arr = torch.from_numpy(audio) if isinstance(audio, np.ndarray) else audio
        # If stereo, average to mono
        if arr.ndim == 2 and arr.shape[0] > 1:
            arr = arr.mean(dim=0, keepdim=True)
        # If shape is [samples], add channel dim
        if arr.ndim == 1:
            arr = arr.unsqueeze(0)
        # Flatten to 1D for ASR
        arr = arr.flatten()
        seg['audio'] = arr

                # 3. ASR
    asr_input = [
        {
            'speaker': seg['speaker'],
            'start_time': seg['start'],
            'end_time': seg['end']
        } for seg in diarized_segments
    ]
    logger.info("Starting ASR...")
    _t1 = time.perf_counter()
    asr_results = asr.transcribe_diarized_segments(temp_path, asr_input)
    logger.info(f"ASR complete in {time.perf_counter()-_t1:.2f}s; type={type(asr_results)} len={len(asr_results) if hasattr(asr_results, '__len__') else 'NA'}")

    # Normalize ASR results to a list of dicts
    normalized_asr = []
    for item in asr_results or []:
        if isinstance(item, dict):
            normalized_asr.append(item)
        elif isinstance(item, list) and item:
            normalized_asr.append(item[0] if isinstance(item[0], dict) else {})
        else:
            normalized_asr.append({})

                # 4. Build speakers list and full transcript
    speakers = []
    full_transcript = []
    for seg, asr_seg in zip(diarized_segments, normalized_asr):
        speakers.append({
            'speaker': seg['speaker'],
            'start': float(seg['start']),
            'end': float(seg['end']),
            'transcript': (asr_seg.get('transcription') if isinstance(asr_seg, dict) else '') or ''
        })
        full_transcript.append((asr_seg.get('transcription') if isinstance(asr_seg, dict) else '') or '')

    transcript = ' '.join(full_transcript).strip()

    # 5. Language selection from ASR segment votes
    from collections import Counter
    langs = [seg.get('language_code') for seg in normalized_asr if isinstance(seg, dict)]
    langs = [l for l in langs if l and l != 'unknown']
    lang = Counter(langs).most_common(1)[0][0] if langs else 'unknown'
    logger.info(f"Selected language: {lang}")

    # 6. Translation
    try:
        translation = translate.translate_to_english(transcript, lang)
    except Exception as e:
        logger.warning(f"Translation failed: {e}")
        translation = transcript
    logger.info("Translation complete")

    # 7. Build response
    response = {
        "speakers": speakers,
        "transcript": transcript,
        "language": lang,
        "translation": translation
    }
    logger.info("Returning response")
    return jsonify(response)


# Handle uploads larger than MAX_CONTENT_LENGTH gracefully
@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(e):
    logging.warning('Upload rejected: file too large')
    return jsonify({'error': 'File too large'}), 413

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=False)