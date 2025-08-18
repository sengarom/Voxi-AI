# Voxi-AI: Intelligent Multilingual Audio Processing

Voxi-AI is a cutting-edge audio processing platform that seamlessly handles speaker diarization, language detection, automatic speech recognition (ASR), and translation. Built on robust open-source frameworks like **SpeechBrain** and **PyDub**, Voxi-AI empowers developers, researchers, and organizations to extract actionable insights from audio content with ease.

---

## 🌟 Features

* **Speaker Diarization**: Accurately identifies multiple speakers in an audio file.
* **Language Detection**: Detects the spoken language of each segment.
* **Automatic Speech Recognition (ASR)**: Converts speech to text efficiently.
* **Translation**: Translates non-English audio into English seamlessly.
* **Audio Preprocessing**: Converts uploaded audio to WAV format, 16kHz, mono for consistent processing.
* **Flexible Backend**: Powered by Flask, ready to integrate with custom frontends.
* **Multi-Format Support**: Accepts `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a` audio files.

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/voxi-ai.git
cd voxi-ai
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **Note:** If `flask_cors` or `ffmpeg` is missing, install them manually:

```bash
pip install flask_cors
# and ensure ffmpeg is installed and available in PATH
```

### 3. Run the Flask Server

```bash
python app.py
```

Visit `http://127.0.0.1:5000` in your browser.

### 4. Upload and Process Audio

* Drag and drop audio files into the web interface (when frontend is ready).
* Supported formats: `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`.
* The backend processes the audio and returns:

  * Speaker-wise segments
  * Language
  * Transcript
  * English Translation

---

## 🛠 Project Structure

```
voxi-ai/
│
├── app.py                # Flask backend server
├── main.py               # Core audio processing pipeline
├── modules/              # Processing modules (ASR, diarization, translation, etc.)
├── uploads/              # Temporary storage for uploaded files
├── processed/            # Preprocessed audio storage
├── templates/            # HTML templates (frontend placeholder)
├── static/               # Static assets (CSS/JS for frontend)
├── requirements.txt      # Python dependencies
└── README.md
```

---

## 💡 Usage Example (CLI)

```bash
python main.py --file sample_audio.mp3 --output result.json
```

Output:

* `result.json`: JSON file with processed segments
* `result.txt`: Human-readable transcript with translations

---

## 📦 Dependencies

* Flask
* Flask-CORS (optional for cross-origin requests)
* PyDub
* SpeechBrain
* Torch & torchaudio
* Other standard Python libraries (`argparse`, `logging`, `json`, etc.)

---

## 🔧 Best Practices

* Keep uploaded audio under **50 MB** for smooth processing.
* Ensure `ffmpeg` is installed for PyDub audio conversions.
* Test with multiple speakers and languages to evaluate accuracy.
* For production, consider adding authentication and secure file handling.

---

## 🌐 Future Enhancements

* **Real-time streaming processing**
* **Advanced translation models**
* **Interactive web frontend**
* **Enhanced speaker identification with profiles**
* **Support for more audio formats and sampling rates**

---

## 🤝 Contributing

We welcome contributions from developers and researchers!

1. Fork the repository
2. Create a new branch for your feature: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add new feature'`
4. Push to the branch: `git push origin feature-name`
5. Open a pull request

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

*Transform your audio into actionable insights with Voxi-AI!*
