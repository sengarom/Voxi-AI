# Voxi-AI: Multilingual Speaker Identification and Transcription System

## Overview

Voxi-AI is a web-based system that performs language-agnostic speaker identification, diarization, transcription, and translation from short audio clips (1-3 minutes). The system processes audio files to produce structured, per-speaker transcripts with language labels and English translations.

## Key Features

- **Audio Preprocessing**: Normalizes input audio (resampling, mono conversion) and applies noise reduction
- **Speaker Diarization**: Identifies "who spoke when" by segmenting audio by speaker
- **Speaker Identification**: Matches voice segments to known speaker profiles
- **Language Identification**: Detects the language spoken in each segment
- **Automatic Speech Recognition (ASR)**: Transcribes speech in the detected language
- **Neural Machine Translation (NMT)**: Translates non-English transcripts to English

## Technology Stack

- **Frontend**: React.js with Material-UI for a responsive interface
- **Backend**: Flask (Python) for API endpoints and processing
- **Audio Processing**: PyAnnote for diarization, SpeechBrain for speaker/language ID
- **Transcription**: OpenAI Whisper for multilingual ASR
- **Translation**: HuggingFace's MarianMT or M2M-100 for translation
- **Deployment**: Docker for containerization, ready for cloud deployment

## Getting Started

### Prerequisites

- Python 3.8+
- Node.js 14+
- FFmpeg (for audio processing)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/voxi-ai.git
cd voxi-ai

# Set up backend
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Set up frontend
cd ../frontend
npm install
```

### Running the Application

```bash
# Start backend (from the backend directory)
flask run --debug

# Start frontend (from the frontend directory)
npm start
```

Visit `http://localhost:3000` to access the application.

## Project Structure

```
voxi-ai/
├── backend/                 # Flask backend
│   ├── app.py               # Main application file
│   ├── audio_processor.py   # Audio preprocessing module
│   ├── diarization.py       # Speaker diarization module
│   ├── speaker_id.py        # Speaker identification module
│   ├── language_id.py       # Language identification module
│   ├── transcription.py     # ASR module using Whisper
│   ├── translation.py       # Translation module
│   └── requirements.txt     # Python dependencies
├── frontend/                # React frontend
│   ├── public/              # Static files
│   ├── src/                 # Source code
│   │   ├── components/      # React components
│   │   ├── services/        # API services
│   │   └── App.js           # Main application component
│   └── package.json         # Node.js dependencies
└── README.md               # Project documentation
```

## Implementation Roadmap

1. **Setup Project Structure**: Initialize backend and frontend frameworks
2. **Audio Preprocessing**: Implement audio normalization and validation
3. **Speaker Diarization**: Integrate PyAnnote for speaker segmentation
4. **Speaker Identification**: Implement voice embedding comparison
5. **Language Identification**: Add SpeechBrain's language detection
6. **Transcription**: Integrate Whisper for multilingual ASR
7. **Translation**: Add translation capabilities for non-English segments
8. **UI Development**: Create intuitive interface for audio upload and transcript display
9. **Testing & Optimization**: Ensure accuracy and performance
10. **Deployment**: Containerize and prepare for deployment

## License

MIT