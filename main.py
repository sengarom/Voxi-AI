import argparse
import logging
import json
import os
from datetime import datetime
from modules.audio_loader import load_audio_file
from modules.diarization import run_speaker_diarization
from modules.speaker_id import identify_speaker
from modules.language_detection import detect_language_from_audio
from modules.asr import run_asr_on_segment
from modules.translate import translate_to_english

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler()]
    )

def save_results(results, output_path):
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.error(f"save_results: Failed to write JSON to {output_path}: {e}")

def save_txt(results, output_path):
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for r in results:
                f.write(f"[{r['speaker']}] ({r['start']}-{r['end']}s, {r['language']}): {r['transcript']}\n")
                f.write(f"    [EN]: {r['translation']}\n")
    except Exception as e:
        logging.error(f"save_txt: Failed to write TXT to {output_path}: {e}")

def preprocess_audio_file(audio_file_path: str) -> str:
    try:
        from pydub import AudioSegment
        base_dir = os.path.dirname(__file__)
        processed_dir = os.path.join(base_dir, 'processed')
        os.makedirs(processed_dir, exist_ok=True)
        name, _ = os.path.splitext(os.path.basename(audio_file_path))
        out_path = os.path.join(processed_dir, f"{name}_16k_mono.wav")

        audio = AudioSegment.from_file(audio_file_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(out_path, format='wav')
        logging.info(f"preprocess_audio_file: Saved preprocessed audio to {out_path}")
        return out_path
    except Exception as e:
        logging.error(f"preprocess_audio_file: Failed to preprocess '{audio_file_path}': {e}")
        return audio_file_path

def process_audio_file(audio_file_path: str):
    setup_logger()
    if not os.path.isfile(audio_file_path) or not os.access(audio_file_path, os.R_OK):
        logging.error(f"process_audio_file: File does not exist or is not readable: {audio_file_path}")
        return []

    preprocessed_path = preprocess_audio_file(audio_file_path)
    if preprocessed_path != audio_file_path:
        logging.info(f"process_audio_file: Using preprocessed audio: {preprocessed_path}")
    else:
        logging.info(f"process_audio_file: Proceeding with original audio: {audio_file_path}")

    try:
        audio_data = load_audio_file(preprocessed_path)
    except Exception as e:
        logging.error(f"process_audio_file: Audio loading failed for '{preprocessed_path}': {e}")
        return []

    try:
        segments = run_speaker_diarization(audio_data)
    except Exception as e:
        logging.error(f"process_audio_file: Speaker diarization failed: {e}")
        return []

    results = []
    for i, segment in enumerate(segments):
        logging.info(f"process_audio_file: Processing segment {i+1}/{len(segments)}: {segment.get('start',0)}-{segment.get('end',0)}s")
        try:
            speaker_label = identify_speaker(segment["audio"])
        except Exception as e:
            logging.warning(f"Speaker identification failed: {e}")
            speaker_label = "Unknown"

        try:
            language = detect_language_from_audio(segment["audio"], audio_data['sample_rate'])
        except Exception as e:
            logging.warning(f"Language detection failed: {e}")
            language = "unknown"

        try:
            transcript = run_asr_on_segment(segment["audio"], audio_data['sample_rate'], language)
        except Exception as e:
            logging.warning(f"ASR failed: {e}")
            transcript = ""

        try:
            translation = translate_to_english(transcript, language) if language != 'en' else transcript
        except Exception as e:
            logging.warning(f"Translation failed: {e}")
            translation = transcript

        results.append({
            "speaker": speaker_label,
            "start": segment.get("start", 0),
            "end": segment.get("end", 0),
            "language": language,
            "transcript": transcript,
            "translation": translation
        })
    return results

def main():
    parser = argparse.ArgumentParser(description="Audio processing pipeline")
    parser.add_argument('--file', required=True, help='Input audio file')
    parser.add_argument('--output', required=False, help='Output JSON file')
    args = parser.parse_args()

    results = process_audio_file(args.file)

    if not results:
        logging.warning("No results to save. Skipping output file writing.")
        return

    for r in results:
        print(f"[{r['speaker']}] ({r['start']}-{r['end']}s, {r['language']}): {r['transcript']}")
        print(f"    [EN]: {r['translation']}")

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_json = args.output or f"results_{ts}.json"
    out_txt = out_json.replace('.json', '.txt')
    save_results(results, out_json)
    save_txt(results, out_txt)
    logging.info(f"Results saved to {out_json} and {out_txt}")

if __name__ == "__main__":
    main()
