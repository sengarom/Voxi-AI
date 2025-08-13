import os
import sys
from pyannote.audio import Pipeline
import json

# ------------------------------
# CONFIGURATION
# ------------------------------
# Path to the locally downloaded pyannote model (must contain config.yml)
MODEL_PATH = r"C:/Users/Somya Raj/Codewithme/Voxi-AI/models/diarization"

# Audio file to process (can be mp3, wav, m4a)
AUDIO_FILE = r"C:/Users/Somya Raj/Codewithme/Voxi-AI/OM.wav"

# Output JSON file
OUTPUT_JSON = r"C:/Users/Somya Raj/Codewithme/Voxi-AI/diarization_output.json"
# ------------------------------

def check_model_exists(model_path):
    """Check if model folder exists and has config.yml"""
    config_file = os.path.join(model_path, "config.yml")
    return os.path.exists(config_file)


def main():
    # 1️⃣ Check model folder
    if not check_model_exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}. Make sure you downloaded it and it contains config.yml.")
        sys.exit(1)

    print(f"✅ Using model from: {MODEL_PATH}")

    # 2️⃣ Load pyannote pipeline from local folder
    pipeline = Pipeline.from_pretrained(MODEL_PATH, use_auth_token=None)  # offline

    # 3️⃣ Run diarization
    print(f"🎤 Processing audio file: {AUDIO_FILE} ...")
    diarization = pipeline(AUDIO_FILE)

    # 4️⃣ Prepare results
    results = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        results.append({
            "start_time": round(turn.start, 3),
            "end_time": round(turn.end, 3),
            "speaker_label": speaker
        })

    # 5️⃣ Print results
    print("\n--- Speaker Diarization Results ---")
    for segment in results:
        print(f"{segment['start_time']}s - {segment['end_time']}s | {segment['speaker_label']}")

    # 6️⃣ Save results to JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()