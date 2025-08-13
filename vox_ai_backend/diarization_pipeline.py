import os
import subprocess
import torch
import torch.nn.functional as F
import torchaudio
from pyannote.audio import Pipeline
from speechbrain.inference import SpeakerRecognition

# ----------------------------
# CONFIG
# ----------------------------
AUDIO_FILE = r"C:\Users\Somya Raj\Codewithme\Voxi-AI\sample.mp3"
PROCESSED_FILE = "processed_audio.wav"
TARGET_SAMPLE_RATE = 16000

# Reference files for speakers
REFERENCES = {
    "Om": r"C:\Users\Somya Raj\Codewithme\Voxi-AI\reference_speakers\om.wav",
    "Somya": r"C:\Users\Somya Raj\Codewithme\Voxi-AI\reference_speakers\somya.wav"
}

HF_TOKEN = "hf_key_1234567890abcdef1234567890abcdef"  # Replace with your Hugging Face token

# ----------------------------
# FUNCTION: Convert to WAV (16 kHz mono)
# ----------------------------
def convert_to_wav(input_path, output_path, sample_rate=16000):
    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ac", "1",
        "-ar", str(sample_rate),
        output_path
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_path

# ----------------------------
# LOAD AUDIO FILE
# ----------------------------
def load_audio(path):
    signal, fs = torchaudio.load(path)
    if fs != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=TARGET_SAMPLE_RATE)
        signal = resampler(signal)
    return signal.squeeze(0)  # 1D tensor

# ----------------------------
# PREPROCESS AUDIO
# ----------------------------
processed_audio = convert_to_wav(AUDIO_FILE, PROCESSED_FILE)

# ----------------------------
# LOAD PIPELINES
# ----------------------------
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization", use_auth_token=HF_TOKEN
)

spk_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models"
)

# ----------------------------
# ENCODE REFERENCE SPEAKERS
# ----------------------------
references = {}
for name, path in REFERENCES.items():
    signal = load_audio(path)             # load waveform tensor
    emb = spk_model.encode_batch(signal)  # returns (frames,192)
    emb = emb.mean(dim=0).squeeze()       # 1D [192]
    references[name] = emb

# ----------------------------
# RUN DIARIZATION
# ----------------------------
diarization = diarization_pipeline({"audio": processed_audio})

# ----------------------------
# MATCH DIARIZED SEGMENTS TO SPEAKERS
# ----------------------------
results = []
for turn, _, speaker_label in diarization.itertracks(yield_label=True):
    segment_signal, _ = torchaudio.load(PROCESSED_FILE, frame_offset=int(turn.start*TARGET_SAMPLE_RATE),
                                        num_frames=int((turn.end-turn.start)*TARGET_SAMPLE_RATE))
    segment_signal = segment_signal.squeeze(0)  # 1D
    segment_emb = spk_model.encode_batch(segment_signal)
    segment_emb = segment_emb.mean(dim=0).squeeze()

    # cosine similarity
    scores = {}
    for name, ref in references.items():
        sim = F.cosine_similarity(segment_emb, ref, dim=0)
        scores[name] = sim.item()

    matched_speaker = max(scores, key=scores.get)
    results.append((turn.start, turn.end, matched_speaker))

# ----------------------------
# PRINT RESULTS
# ----------------------------
print("\n--- Speaker Diarization Results ---")
for start, end, speaker in results:
    print(f"{start:.1f}s - {end:.1f}s  Speaker: {speaker}")

# ----------------------------
# CLEANUP
# ----------------------------
# os.remove(PROCESSED_FILE)
