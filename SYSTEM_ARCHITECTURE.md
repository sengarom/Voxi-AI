# System Architecture

## Overview

VOXI AI is an advanced multilingual audio processing platform that combines speaker diarization, speech recognition, and translation capabilities to provide comprehensive audio understanding.

## Processing Flow

1. **Audio Upload and Loading**
   - The system accepts audio files (.wav, .mp3, .flac, .ogg, .m4a)
   - The audio is loaded using PyDub and normalized to ensure consistent processing

2. **Speaker Diarization with pyannote.audio**
   - The audio is processed by the PyAnnote speaker diarization model
   - This identifies different speakers and separates the audio into segments
   - Each segment contains: speaker ID, start time, end time, and audio data

3. **Speech Recognition with OpenAI Whisper**
   - Each diarized segment is processed by the Whisper ASR model
   - Whisper transcribes the speech to text with high accuracy
   - Whisper also detects the language being spoken in each segment
   - Transcription results include: text, language, confidence score

4. **Language Detection**
   - Whisper provides language detection for each segment
   - The system determines the overall language based on the most common language across segments

5. **Translation with Helsinki-NLP**
   - For non-English segments, the Helsinki-NLP translation models translate the text to English
   - The system uses different models based on the source language
   - Translations are provided both at the segment level and for the complete transcript

6. **Results Processing**
   - The system combines all results into a structured format
   - Speaker labels are normalized (A, B, C, etc.)
   - Results include timestamps, transcriptions, and translations

7. **Frontend Display**
   - Results are displayed in the web interface with tabs for different views
   - Users can view the transcript, speakers, languages, and translations
   - Results can be downloaded for further use

## Components

### Backend (Flask)
- **app.py**: Main application controller
- **modules/audio_loader.py**: Handles audio file loading and preprocessing
- **modules/diarization.py**: Speaker diarization using pyannote.audio
- **modules/asr_new.py**: Speech recognition using OpenAI Whisper
- **modules/translate_new.py**: Translation using Helsinki-NLP models

### Frontend
- **templates/index.html**: Main user interface
- **static/style.css**: Styling for the web interface
- **static/script.js**: Client-side functionality and API integration

## Technology Stack

- **Python 3**: Core backend language
- **Flask**: Web framework
- **PyTorch**: Deep learning framework for models
- **OpenAI Whisper**: State-of-the-art ASR model
- **Helsinki-NLP Models**: Advanced neural machine translation
- **PyAnnote.audio**: Speaker diarization
- **PyDub & Librosa**: Audio processing
- **HTML/CSS/JavaScript**: Frontend interface

## Performance Considerations

- Whisper models are loaded lazily to minimize resource usage when not in use
- Helsinki-NLP models are cached to avoid reloading the same model multiple times
- Text is split into chunks for translation to handle API limits
- Audio processing is optimized for memory efficiency

## Future Improvements

1. Add real-time processing capabilities
2. Implement more language pairs for translation
3. Add support for custom vocabulary and domain-specific terminology
4. Improve speaker identification with voice profiles
5. Add emotion/sentiment detection capabilities
