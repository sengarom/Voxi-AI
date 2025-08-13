import argparse
import logging
import json
from datetime import datetime
from modules.audio_loader import load_audio_file
from modules.diarization import run_speaker_diarization
from modules.speaker_id import identify_speaker
from modules.language_detection import detect_language_from_audio
from modules.asr import run_asr_on_segment
from modules.translate import translate_to_english

def setup_logger():
    """
    Set up the logging configuration for the application.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler()]
    )

def save_results(results, output_path):
    """
    Save the results list as a JSON file.
    Args:
        results (list): List of result dictionaries.
        output_path (str): Path to save the JSON file.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

def save_txt(results, output_path):
    """
    Save the results list as a TXT file with speaker turns and translations.
    Args:
        results (list): List of result dictionaries.
        output_path (str): Path to save the TXT file.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(f"[{r['speaker']}] ({r['start']}-{r['end']}s, {r['language']}): {r['transcript']}\n")
            f.write(f"    [EN]: {r['translation']}\n")


def process_audio_file(audio_file_path: str):
    """
    Run the full audio processing pipeline on the given audio file.
    Args:
        audio_file_path (str): Path to the audio file to process.
    Returns:
        list: List of result dictionaries for each speaker segment.
    """
    setup_logger()
    if not os.path.isfile(audio_file_path) or not os.access(audio_file_path, os.R_OK):
        logging.error(f"File does not exist or is not readable: {audio_file_path}")
        return []
    logging.info(f"Loading audio: {audio_file_path}")
    try:
        audio_data = load_audio_file(audio_file_path)
    except Exception as e:
        logging.error(f"Audio loading failed (load_audio_file): {e}")
        return []

    try:
        segments = run_speaker_diarization(audio_data)
    except Exception as e:
        logging.error(f"Speaker diarization failed (run_speaker_diarization): {e}")
        return []

    results = []
    for i, segment in enumerate(segments):
        logging.info(f"Processing segment {i+1}/{len(segments)}: {segment.get('start', 0)}-{segment.get('end', 0)}s")
        try:
            speaker_label = identify_speaker(segment["audio"])
        except Exception as e:
            logging.warning(f"Speaker identification failed (identify_speaker): {e}")
            speaker_label = "Unknown"

        try:
            language = detect_language_from_audio(segment["audio"], audio_data['sample_rate'])
        except Exception as e:
            logging.warning(f"Language detection failed (detect_language_from_audio): {e}")
            language = "unknown"

        try:
            transcript = run_asr_on_segment(segment["audio"], audio_data['sample_rate'], language)
        except Exception as e:
            logging.warning(f"ASR failed (run_asr_on_segment): {e}")
            transcript = ""

        try:
            translation = translate_to_english(transcript, language) if language != 'en' else transcript
        except Exception as e:
            logging.warning(f"Translation failed (translate_to_english): {e}")
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
    """
    Command-line entry point for the audio processing pipeline.
    Parses arguments, runs processing, and saves results if available.
    """
    parser = argparse.ArgumentParser(description="Audio processing pipeline")
    parser.add_argument('--file', required=True, help='Input audio file')
    parser.add_argument('--output', required=False, help='Output JSON file')
    args = parser.parse_args()

    results = process_audio_file(args.file)

    if not results:
        logging.warning("No results to save. Skipping output file writing.")
        return

    # Print results
    for r in results:
        print(f"[{r['speaker']}] ({r['start']}-{r['end']}s, {r['language']}): {r['transcript']}")
        print(f"    [EN]: {r['translation']}")

    # Save results
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_json = args.output or f"results_{ts}.json"
    out_txt = out_json.replace('.json', '.txt')
    save_results(results, out_json)
    save_txt(results, out_txt)
    logging.info(f"Results saved to {out_json} and {out_txt}")

if __name__ == "__main__":
    main()
