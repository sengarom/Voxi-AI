<div align="center">
  <img src="https://your-logo-url.com/voxi-ai-logo.png" alt="Voxi AI Logo" width="150"/>
  <h1>Voxi AI</h1>
  <p><strong>Intelligent Multilingual Audio Processing</strong></p>
  <p>
    <a href="https://github.com/sengarom/Voxi-AI/stargazers"><img src="https://img.shields.io/github/stars/sengarom/Voxi-AI?style=for-the-badge&logo=github&color=00ff41" alt="Stars Badge"/></a>
    <a href="https://github.com/sengarom/Voxi-AI/network/members"><img src="https://img.shields.io/github/forks/sengarom/Voxi-AI?style=for-the-badge&logo=github&color=00ff41" alt="Forks Badge"/></a>
    <a href="https://github.com/sengarom/Voxi-AI/issues"><img src="https://img.shields.io/github/issues/sengarom/Voxi-AI?style=for-the-badge&logo=github&color=ffd700" alt="Issues Badge"/></a>
    <a href="https://github.com/sengarom/Voxi-AI/blob/main/LICENSE"><img src="https://img.shields.io/github/license/sengarom/Voxi-AI?style=for-the-badge&color=39ff14" alt="License Badge"/></a>
  </p>
</div>

---

## 🚀 Overview

Voxi AI is a powerful, open-source platform designed for advanced audio analysis. It seamlessly integrates state-of-the-art AI models to provide a comprehensive suite of tools for processing multilingual audio content. Whether you're a developer, researcher, or content creator, Voxi AI empowers you to unlock valuable insights from your audio files with ease.

Our platform offers a user-friendly web interface to access its core functionalities, including speaker diarization, automatic speech recognition (ASR), language detection, and translation.

## ✨ Key Features

- **🗣️ Speaker Diarization**: Accurately identifies and separates different speakers in an audio file.
- **🎙️ Automatic Speech Recognition (ASR)**: Transcribes spoken words into text with high accuracy using OpenAI's Whisper model.
- **🌍 Language Detection**: Automatically detects the language being spoken in each audio segment.
- **🔄 Translation**: Translates transcribed text from various languages into English using high-quality Helsinki-NLP models.
- **🖥️ Web-Based UI**: An intuitive and responsive interface for easy file uploads and results visualization.
- **⚙️ Flexible Processing**: Supports a wide range of audio formats, including `.wav`, `.mp3`, `.flac`, `.ogg`, and `.m4a`.

## 🛠️ Technology Stack

Voxi AI is built with a robust and scalable technology stack:

- **Backend**: `Python` with `Flask`
- **ASR**: `OpenAI Whisper`
- **Translation**: `Helsinki-NLP`
- **Speaker Diarization**: `pyannote.audio`
- **Audio Manipulation**: `pydub`
- **Frontend**: `HTML5`, `CSS3`, `JavaScript`

## 🏁 Getting Started

Follow these steps to get Voxi AI up and running on your local machine.

### Prerequisites

- Python 3.8 or higher
- `pip` for package management

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sengarom/Voxi-AI.git
   cd Voxi-AI
   ```

2. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python run.py
   ```

4. **Access the web interface:**
   Open your browser and navigate to `http://127.0.0.1:5000`.

## 📖 How to Use

1. **Upload an Audio File**: Drag and drop or browse to select an audio file.
2. **Select Options**: Choose the processing tasks you want to perform (e.g., speaker diarization, translation).
3. **Process**: Click the "Process Audio" button to start the analysis.
4. **View Results**: The results will be displayed in organized tabs for the transcript, speakers, languages, and translation.

## 🏛️ Architectural Overview

The Voxi AI processing pipeline is as follows:

1. **File Upload**: The user uploads an audio file through the web interface.
2. **Audio Conversion**: The uploaded file is converted to a standardized WAV format.
3. **Speaker Diarization**: `pyannote.audio` analyzes the audio to identify speaker segments.
4. **ASR & Language Detection**: Each segment is transcribed and its language is identified by `Whisper`.
5. **Translation**: Non-English segments are translated to English using `Helsinki-NLP` models.
6. **Results Display**: The processed information is sent to the frontend and displayed to the user.

For a more detailed breakdown, please see our [System Architecture Document](SYSTEM_ARCHITECTURE.md).

## 🤝 How to Contribute

We welcome contributions from the community! If you'd like to help improve Voxi AI, please follow these steps:

1. **Fork the repository.**
2. **Create a new branch** for your feature or bug fix: `git checkout -b feature-name`.
3. **Make your changes** and commit them with a clear message.
4. **Push your changes** to your fork.
5. **Create a pull request** to the `main` branch of this repository.

Please read our [Contributing Guidelines](CONTRIBUTING.md) for more information.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 📧 Contact

If you have any questions, suggestions, or feedback, please feel free to reach out to us by opening an issue on GitHub.

---

<div align="center">
  <p>Made with ❤️ by the Voxi AI Team</p>
</div>
