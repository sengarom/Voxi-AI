# Flask backend for audio upload and processing
# Place your real frontend files in templates/ and static/ folders
import os
import json
import logging
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from main import process_audio_file
# Do not remove these imports (used in main.py):
from modules import audio_loader, diarization, speaker_id, language_detection, asr, translate

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a'}


app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB
CORS(app)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    # This serves the mock frontend. Replace index.html with your real frontend later.
    return render_template('index.html')

@app.route('/process-audio', methods=['POST'])
def process_audio():
    if 'audiofile' not in request.files:
        logging.warning('No file part in request')
        return jsonify({'error': 'No file part'}), 400
    file = request.files['audiofile']
    if file.filename == '':
        logging.warning('No selected file')
        return jsonify({'error': 'No selected file'}), 400
    if not allowed_file(file.filename):
        logging.warning(f'Unsupported file type: {file.filename}')
        return jsonify({'error': 'Unsupported file type'}), 400
    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        file.save(save_path)
        logging.info(f'Received and saved file: {filename}')
    except Exception as e:
        logging.error(f'Failed to save file: {e}')
        return jsonify({'error': f'Failed to save file: {e}'}), 500

    try:
        logging.info(f'Processing file: {filename}')
        results = process_audio_file(save_path)
        logging.info(f'Processing complete for file: {filename}')
    except Exception as e:
        logging.error(f'Processing failed for {filename}: {e}')
        os.remove(save_path)
        return jsonify({'error': f'Processing failed: {e}'}), 500
    os.remove(save_path)
    if not results:
        logging.warning(f'Processing returned no results for file: {filename}')
        return jsonify({'error': 'No speech detected'}), 200
    return jsonify(results)

# Static files (CSS/JS) are served automatically from /static/
# Place your real frontend assets in the static/ folder

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
