# Voxi AI Diarization Pipeline Setup

## 1. Install dependencies

Open your terminal in the project folder and run:

```
pip install -r requirements.txt
```

You also need [ffmpeg](https://ffmpeg.org/download.html) installed on your system for audio processing.

## 2. Set your Hugging Face token

Before running the script, set your Hugging Face token as an environment variable.

**Windows Command Prompt:**
```
set HF_AUTH_TOKEN=your_token_here
python diarization_pipeline.py
```

**Windows PowerShell:**
```
$env:HF_AUTH_TOKEN="your_token_here"
python diarization_pipeline.py
```

Replace `your_token_here` with your actual Hugging Face token.

## 3. Run the script

After setting the environment variable, run:

```
python diarization_pipeline.py
```

## 4. Notes

- The demo uses silent audio files for structure demonstration. Replace them with real speech audio for meaningful results.
- For web integration, wrap the `VoxiDiarizationPipeline` class in your web framework (Flask, FastAPI, etc.).



# Requirements

- Python 3.8+
- torch
- pyannote.audio
- pydub
- scipy
- numpy
- ffmpeg (system dependency, must be installed and in PATH)