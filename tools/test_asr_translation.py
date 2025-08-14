# Test script for ASR and Translation
import os
import sys
import time
import argparse
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("VOXI-Test")

# Add the parent directory to the path so we can import modules
sys.path.append(str(Path(__file__).parent.parent))

# Import our modules
from modules import asr
from modules import translate
from modules import audio_loader

def test_asr(audio_path, language=None):
    """
    Test the ASR module with a single audio file
    """
    logger.info(f"Testing ASR with file: {audio_path}")
    logger.info(f"Language hint: {language or 'auto-detect'}")
    
    # Load the audio
    audio_data = audio_loader.load_audio_file(audio_path)
    logger.info(f"Loaded audio: {audio_data['duration']:.2f}s, {audio_data['sample_rate']}Hz")
    
    # Create a simple segment with the entire audio
    segment = {
        'audio': audio_data['audio'],
        'sample_rate': audio_data['sample_rate'],
        'speaker': 'TEST',
        'start': 0,
        'end': audio_data['duration']
    }
    
    # Run ASR
    start_time = time.time()
    result = asr.transcribe_audio_segment(segment['audio'], language)
    elapsed = time.time() - start_time
    
    # Print results
    logger.info(f"ASR completed in {elapsed:.2f}s")
    logger.info(f"Detected language: {result.get('language', 'unknown')}")
    logger.info(f"Confidence: {result.get('confidence', 0):.2f}")
    logger.info(f"Transcript: {result.get('text', '')}")
    
    return result

def test_translation(text, language):
    """
    Test the translation module with a text sample
    """
    logger.info(f"Testing translation for language: {language}")
    logger.info(f"Text: {text[:50]}...")
    
    # Run translation
    start_time = time.time()
    result = translate.translate_to_english(text, language)
    elapsed = time.time() - start_time
    
    # Print results
    logger.info(f"Translation completed in {elapsed:.2f}s")
    logger.info(f"Translation: {result}")
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Test VOXI AI ASR and Translation")
    parser.add_argument("--audio", required=True, help="Path to audio file for testing")
    parser.add_argument("--language", default=None, help="Language hint (e.g., 'hi', 'ur', 'en')")
    args = parser.parse_args()
    
    if not os.path.exists(args.audio):
        logger.error(f"Audio file not found: {args.audio}")
        return 1
    
    # Test ASR
    logger.info("=== Testing ASR ===")
    asr_result = test_asr(args.audio, args.language)
    
    # Test Translation
    if asr_result and asr_result.get('text'):
        logger.info("\n=== Testing Translation ===")
        detected_lang = asr_result.get('language')
        translation_result = test_translation(asr_result.get('text'), detected_lang)
    
    logger.info("Testing completed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
