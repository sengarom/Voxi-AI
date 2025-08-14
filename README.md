# VOXI AI - Multilingual Audio Processing Platform

VOXI AI is a comprehensive audio processing platform that combines speaker diarization, speech recognition, language detection, and translation capabilities to provide complete audio understanding.

## Features

- **Speaker Diarization**: Identifies different speakers in audio recordings
- **Speech Recognition**: Transcribes speech to text using OpenAI Whisper
- **Language Detection**: Automatically identifies spoken languages
- **Translation**: Translates non-English text to English using Helsinki-NLP models
- **Web Interface**: Easy-to-use interface for processing audio files

## Technology Stack

- **Backend**: Python, Flask
- **ASR**: OpenAI Whisper (local processing)
- **Translation**: Helsinki-NLP Machine Translation Models
- **Speaker Diarization**: PyAnnote.audio
- **Frontend**: HTML, CSS, JavaScript

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/voxi-ai.git
cd voxi-ai
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python run.py
```

4. Open your browser and navigate to `http://localhost:5000`

## Usage

1. Upload an audio file (.wav, .mp3, .flac, .ogg, .m4a)
2. Select processing options (speaker diarization, language detection, etc.)
3. Click "Process Audio"
4. View the results in different tabs:
   - **Transcript**: Shows the transcribed text
   - **Speakers**: Shows the identified speakers and their segments
   - **Languages**: Shows the detected languages
   - **Translation**: Shows the English translation

## System Architecture

The processing flow consists of the following steps:

1. Audio file is uploaded and loaded
2. Speaker diarization is performed using PyAnnote
3. Speech recognition is performed on each segment using Whisper
4. Language detection is performed for each segment
5. Non-English segments are translated to English using Helsinki-NLP models
6. Results are displayed in the web interface

For more details, see [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md).

## Test Scripts

You can test the ASR and translation components individually:

```bash
python tools/test_asr_translation.py --audio your_audio_file.mp3 --language hi
```

## Performance Considerations

- First-time startup may be slow due to model downloads
- Processing time depends on audio length and complexity
- Translation models are cached to improve performance
- GPU acceleration is used if available

## License

[MIT License](LICENSE)
