# Voxi AI: Diarization & Speaker Identification Pipeline
# This script provides the core logic for processing an audio file to determine
# "who spoke when" and identify speakers against an enrolled database.

# Requirements
# - Python 3.8+
# - torch
# - pyannote.audio
# - pydub
# - scipy
# - numpy
# - ffmpeg (system dependency, must be installed and in PATH)

import os
import torch
from pydub import AudioSegment
from pyannote.audio import Pipeline as DiarizationPipeline
from pyannote.audio import Model
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.core import Annotation
from scipy.spatial.distance import cdist
import numpy as np
import warnings
import tempfile
import logging
import shutil

# Suppress a specific warning from pyannote.audio about ONNX runtime
warnings.filterwarnings("ignore", message="Could not find the optimal ONNXRuntime backend.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def check_ffmpeg():
    if not shutil.which("ffmpeg"):
        logger.error("ffmpeg is not installed or not found in system PATH.")
        raise EnvironmentError(
            "ffmpeg is required but not found. Please install ffmpeg and ensure it is in your system PATH."
        )

# Call this at the top of your script, before any audio processing
check_ffmpeg()

class VoxiDiarizationPipeline:
    """
    A class to encapsulate the speaker diarization and identification process.
    """

    # --- Constants ---
    # The similarity threshold for identifying a speaker. A lower value means a stricter match.
    # This value may need tuning based on testing with your specific data.
    SIMILARITY_THRESHOLD = 0.5  # Cosine distance, lower is more similar.

    def __init__(self, auth_token):
        """
        Initializes the pipeline by loading the necessary pyannote.audio models.

        Args:
            auth_token (str): A Hugging Face authentication token is required to download
                              the pre-trained models from Hugging Face Hub.
        """
        if not auth_token:
            raise ValueError("A Hugging Face authentication token is required. "
                             "Please visit https://huggingface.co/settings/tokens")

        # Use GPU if available, otherwise fallback to CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 1. Load the pre-trained speaker diarization pipeline
        # This pipeline will handle segmentation, embedding, and clustering.
        print("Loading Speaker Diarization pipeline...")
        self.diarization_pipeline = DiarizationPipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=auth_token
        ).to(self.device)

        # 2. Load the embedding model separately for speaker identification
        # This gives us direct access to creating voiceprints.
        print("Loading Speaker Embedding model...")
        self.embedding_model = Model.from_pretrained(
            "pyannote/embedding",
            use_auth_token=auth_token
        ).to(self.device)

        # This dictionary will store the voiceprints of enrolled speakers
        self.enrolled_speakers = {}

    def _is_safe_path(self, base_dir, target_path):
        # Returns True if target_path is inside base_dir
        base_dir = os.path.abspath(base_dir)
        target_path = os.path.abspath(target_path)
        return os.path.commonpath([base_dir]) == os.path.commonpath([base_dir, target_path])

    def _preprocess_audio(self, input_path):
        """
        Converts any input audio file to the format required by our models:
        - 16kHz sample rate
        - Mono (single channel)
        - WAV format

        This step is crucial for robustness across different audio inputs.

        Args:
            input_path (str): Path to the input audio file.

        Returns:
            str: Path to the processed temporary WAV file.
        """
        logger.info(f"Preprocessing audio file: {input_path}")

        # Directory traversal protection
        allowed_base_dir = os.getcwd()  # or set to your allowed data directory
        if not self._is_safe_path(allowed_base_dir, input_path):
            logger.error("Attempted access outside allowed directory.")
            raise ValueError("Invalid file path: directory traversal detected.")

        allowed_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
        max_size_mb = 100

        if not os.path.isfile(input_path):
            logger.error("File does not exist.")
            raise FileNotFoundError(f"File not found: {input_path}")

        ext = os.path.splitext(input_path)[1].lower()
        if ext not in allowed_extensions:
            logger.error(f"Unsupported file type '{ext}'. Allowed types: {allowed_extensions}")
            raise ValueError(f"Unsupported file type: {ext}")

        size_mb = os.path.getsize(input_path) / (1024 * 1024)
        if size_mb > max_size_mb:
            logger.error(f"File size {size_mb:.2f} MB exceeds limit of {max_size_mb} MB.")
            raise ValueError(f"File size exceeds limit: {size_mb:.2f} MB")

        if size_mb == 0:
            logger.error("File is empty.")
            raise ValueError("File is empty.")

        try:
            audio = AudioSegment.from_file(input_path)
            audio = audio.set_frame_rate(16000).set_channels(1)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                audio.export(temp_file.name, format="wav")
                logger.info(f"Audio preprocessed and saved to: {temp_file.name}")
                return temp_file.name
        except Exception as e:
            logger.exception(f"Error during audio preprocessing: {e}")
            raise

    def process_enrollment_data(self, enrollment_dir):
        """
        Processes all audio files in the enrollment directory to create
        a voiceprint for each speaker.

        Args:
            enrollment_dir (str): Path to the directory containing speaker folders.
                                  Expected structure: enrollment_dir/{speaker_name}/{sample}.wav
        """
        logger.info(f"Processing enrollment data from: {enrollment_dir}")

        allowed_base_dir = os.getcwd()
        if not self._is_safe_path(allowed_base_dir, enrollment_dir):
            logger.error("Attempted access outside allowed directory.")
            return

        if not os.path.isdir(enrollment_dir):
            logger.error(f"Enrollment directory not found: {enrollment_dir}")
            return

        for speaker_name in os.listdir(enrollment_dir):
            speaker_dir = os.path.join(enrollment_dir, speaker_name)
            if os.path.isdir(speaker_dir):
                embeddings = []
                for sample_file in os.listdir(speaker_dir):
                    sample_path = os.path.join(speaker_dir, sample_file)
                    processed_path = self._preprocess_audio(sample_path)
                    if processed_path:
                        try:
                            embedding = self.embedding_model(
                                {"uri": sample_file, "audio": processed_path}
                            )
                            embeddings.append(embedding.squeeze())
                        except Exception as e:
                            logger.error(f"Could not create embedding for {sample_path}: {e}")
                        finally:
                            if os.path.exists(processed_path):
                                os.remove(processed_path)

                if embeddings:
                    avg_embedding = np.mean(np.array([e.cpu().numpy() for e in embeddings]), axis=0)
                    self.enrolled_speakers[speaker_name] = avg_embedding
                    logger.info(f"Enrolled speaker '{speaker_name}' with a voiceprint.")
        logger.info("Enrollment processing complete.")


    def _identify_speaker(self, target_embedding):
        """
        Compares a target embedding to all enrolled speaker voiceprints.

        Args:
            target_embedding (np.ndarray): The embedding of the speaker to identify.

        Returns:
            tuple: (speaker_name, confidence_score)
                   'Unknown' if no match is found above the threshold.
                   Confidence is the cosine similarity (lower is better).
        """
        if not self.enrolled_speakers:
            return "Unknown", 1.0

        enrolled_names = list(self.enrolled_speakers.keys())
        enrolled_embeddings = np.array(list(self.enrolled_speakers.values()))

        # Calculate cosine distance between the target and all enrolled embeddings
        distances = cdist(np.expand_dims(target_embedding, axis=0), enrolled_embeddings, metric='cosine')[0]

        best_match_index = np.argmin(distances)
        best_match_score = distances[best_match_index]

        # Check if the best match is below our similarity threshold
        if best_match_score < self.SIMILARITY_THRESHOLD:
            # We have a confident match
            return enrolled_names[best_match_index], best_match_score
        else:
            # No match was close enough
            return "Unknown", best_match_score


    def run(self, audio_path, num_speakers=None, min_speakers=None, max_speakers=None):
        """
        The main execution function. It runs the full diarization and
        identification pipeline on a single audio file.

        Args:
            audio_path (str): Path to the audio file to be processed.
            num_speakers (int, optional): The exact number of speakers in the audio.
            min_speakers (int, optional): The minimum number of speakers.
            max_speakers (int, optional): The maximum number of speakers.            MOCK_TEST_FILE = "mock_test_audio.wav"
            ...
            final_result = voxi_pipeline.run(MOCK_TEST_FILE)

        Returns:
            list: A list of dictionaries, where each dictionary represents a
                  speaker segment. Returns an empty list on failure.
        """
        # 1. Preprocess the input audio
        processed_audio_path = self._preprocess_audio(audio_path)
        if not processed_audio_path:
            return []

        # 2. Run Speaker Diarization
        logger.info("Running speaker diarization...")
        try:
            diarization_result = self.diarization_pipeline(
                processed_audio_path,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers
            )
        except Exception as e:
            logger.error(f"Error during diarization: {e}")
            if os.path.exists(processed_audio_path):
                os.remove(processed_audio_path)
            return []

        # 3. Process results and perform Speaker Identification
        output_segments = []
        # The whole_speech=True parameter ensures we get a single embedding for the entire file's speech content per speaker
        speaker_embeddings = self.embedding_model.crop(processed_audio_path, diarization_result, mode="whole")

        speaker_map = {}
        for (speaker_label, _), embedding in speaker_embeddings:
            # Identify the speaker using the extracted embedding
            identified_name, confidence = self._identify_speaker(embedding.cpu().numpy())
            speaker_map[speaker_label] = {
                "name": identified_name,
                "confidence": 1 - confidence # Convert distance to similarity score (higher is better)
            }
            logger.info(f"Mapped diarization label '{speaker_label}' to '{identified_name}' with confidence {speaker_map[speaker_label]['confidence']:.2f}")

        # 4. Format the final output
        for segment, _, speaker_label in diarization_result.itertracks(yield_label=True):
            mapped_info = speaker_map.get(speaker_label, {"name": "Unknown", "confidence": 0.0})
            output_segments.append({
                "speaker_tag": speaker_label,
                "identified_speaker": mapped_info["name"],
                "start_time": round(segment.start, 3),
                "end_time": round(segment.end, 3),
                "identification_confidence": round(mapped_info['confidence'], 3)
            })

        # 5. Clean up the temporary audio file
        if os.path.exists(processed_audio_path):
            os.remove(processed_audio_path)

        logger.info("Pipeline run completed.")
        return output_segments

# --- Example Usage ---
import os
HF_AUTH_TOKEN = os.environ.get("HF_AUTH_TOKEN")
if not HF_AUTH_TOKEN:
    raise ValueError("Please set the HF_AUTH_TOKEN environment variable with your Hugging Face token.")

# --- Setup a mock environment for demonstration ---
logger.info("--- Setting up mock environment for demonstration ---")

MOCK_ENROLL_DIR = "mock_enrollment_data"
os.makedirs(os.path.join(MOCK_ENROLL_DIR, "Alice"), exist_ok=True)
os.makedirs(os.path.join(MOCK_ENROLL_DIR, "Bob"), exist_ok=True)

try:
    AudioSegment.silent(duration=5000).export(os.path.join(MOCK_ENROLL_DIR, "Alice", "alice_sample1.wav"), format="wav")
    AudioSegment.silent(duration=5000).export(os.path.join(MOCK_ENROLL_DIR, "Bob", "bob_sample1.wav"), format="wav")
    MOCK_TEST_FILE = "mock_test_audio.wav"
    AudioSegment.silent(duration=10000).export(MOCK_TEST_FILE, format="wav")
    logger.info("Mock files created successfully. (Note: These are silent files for structure demonstration).")
    logger.info("For a real test, replace these with actual audio files.")

    logger.info("--- Initializing and running the Voxi AI Pipeline ---")
    voxi_pipeline = VoxiDiarizationPipeline(auth_token=HF_AUTH_TOKEN)

    voxi_pipeline.process_enrollment_data(MOCK_ENROLL_DIR)

    logger.info(f"--- Processing test file: {MOCK_TEST_FILE} ---")
    final_result = voxi_pipeline.run(MOCK_TEST_FILE)

    import json
    logger.info("--- Final Result ---")
    logger.info(json.dumps(final_result, indent=2))

except Exception as e:
    logger.exception("An error occurred during the example run.")
    logger.error("Please ensure you have 'ffmpeg' installed on your system for audio creation.")
    logger.error("For a real test, you can manually create the directory structure and use your own audio files.")

# --- Diarize your own audio file ---
YOUR_AUDIO_FILE = "my_audio.wav"  # <-- Change to your actual file name

logger.info(f"--- Processing your audio file: {YOUR_AUDIO_FILE} ---")
final_result = voxi_pipeline.run(YOUR_AUDIO_FILE)

import json
logger.info("--- Final Result ---")
logger.info(json.dumps(final_result, indent=2))