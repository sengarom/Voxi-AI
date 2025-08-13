# test_asr.py

from asr_module import transcribe_diarized_segments
from pprint import pprint

def run_test():
    """
    A simple function to test our ASR module with mock data.
    """
    print("--- Starting ASR Module Standalone Test ---")

    # This is mock output, like the data that would be provided by the diarization module.
    mock_diarization_data = [
        {'speaker': 'SPEAKER_00', 'start_time': '0.00', 'end_time': '10.00'},
        
    ]

    # You must have a sample audio file in the same directory.
    mock_audio_file = 'sample_audio.mp3'

    # This is the function call the Django app will eventually make.
    final_output = transcribe_diarized_segments(
        audio_path=mock_audio_file,
        diarization_output=mock_diarization_data
    )

    print("\n--- Final Enriched Output ---")
    if final_output:
        pprint(final_output)
    else:
        print("Processing failed. Please check for errors above.")

# This line ensures the test runs only when you execute this file directly.
if __name__ == '__main__':
    run_test()