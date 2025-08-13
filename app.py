
# Flask backend for audio upload and processing
# Place your real frontend files in templates/ and static/ folders
import os
import json
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from main import process_audio_file

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a'}

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    # This serves the mock frontend. Replace index.html with your real frontend later.
    return render_template('index.html')

@app.route('/process-audio', methods=['POST'])
def process_audio():
    if 'audiofile' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['audiofile']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file type'}), 400
    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        file.save(save_path)
    except Exception as e:
        return jsonify({'error': f'Failed to save file: {e}'}), 500
    try:
        results = process_audio_file(save_path)
    except Exception as e:
        os.remove(save_path)
        return jsonify({'error': f'Processing failed: {e}'}), 500
    os.remove(save_path)
    return jsonify(results)

# Static files (CSS/JS) are served automatically from /static/
# Place your real frontend assets in the static/ folder

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
