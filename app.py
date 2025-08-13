from flask import Flask, request, render_template_string, send_file, jsonify
from flask_cors import CORS
import tempfile
import os
import json
from main import process_audio_file  # You must refactor main.py to expose this function

app = Flask(__name__)
CORS(app)

UPLOAD_FORM = '''
<!doctype html>
<title>Audio Processing Upload</title>
<h1>Upload an audio file</h1>
<form method=post enctype=multipart/form-data>
  <input type=file name=audiofile>
  <input type=submit value=Upload>
</form>
'''

RESULT_TEMPLATE = '''
<!doctype html>
<title>Audio Processing Results</title>
<h1>Results</h1>
<table border=1>
<tr><th>Speaker</th><th>Start (s)</th><th>End (s)</th><th>Language</th><th>Transcript</th><th>Translation</th></tr>
{% for r in results %}
<tr>
  <td>{{ r['speaker'] }}</td>
  <td>{{ r['start'] }}</td>
  <td>{{ r['end'] }}</td>
  <td>{{ r['language'] }}</td>
  <td>{{ r['transcript'] }}</td>
  <td>{{ r['translation'] }}</td>
</tr>
{% endfor %}
</table>
<a href="/download/{{ jsonfile }}">Download JSON</a>
'''

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'audiofile' not in request.files:
            return 'No file part', 400
        file = request.files['audiofile']
        if file.filename == '':
            return 'No selected file', 400
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        try:
            results = process_audio_file(tmp_path)
            json_path = tmp_path + '.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            os.remove(tmp_path)
            return render_template_string(RESULT_TEMPLATE, results=results, jsonfile=os.path.basename(json_path))
        except Exception as e:
            os.remove(tmp_path)
            return f'Processing failed: {e}', 500
    return UPLOAD_FORM

@app.route('/download/<filename>')
def download_json(filename):
    json_path = os.path.join(tempfile.gettempdir(), filename)
    if not os.path.exists(json_path):
        return 'File not found', 404
    return send_file(json_path, as_attachment=True)

@app.route('/api/process', methods=['POST'])
def api_process():
    if 'audiofile' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['audiofile']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name
    try:
        results = process_audio_file(tmp_path)
        os.remove(tmp_path)
        return jsonify(results)
    except Exception as e:
        os.remove(tmp_path)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
