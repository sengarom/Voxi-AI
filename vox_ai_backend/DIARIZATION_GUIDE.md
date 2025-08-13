# Voxi AI: Diarization & Speaker Identification Pipeline
# This script provides the core logic for processing an audio file to determine
# "who spoke when" and identify speakers against an enrolled database.

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

# Suppress a specific warning from pyannote.audio about ONNX runtime
warnings.filterwarnings("ignore", message="Could not find the optimal ONNXRuntime backend.")

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
        print(f"Preprocessing audio file: {input_path}")
        try:
            audio = AudioSegment.from_file(input_path)
            audio = audio.set_frame_rate(16000).set_channels(1)

            # Create a temporary file for the processed audio
            temp_output_path = f"temp_{os.path.basename(input_path)}.wav"
            audio.export(temp_output_path, format="wav")
            print(f"Audio preprocessed and saved to: {temp_output_path}")
            return temp_output_path
        except Exception as e:
            print(f"Error during audio preprocessing: {e}")
            return None

    def process_enrollment_data(self, enrollment_dir):
        """
        Processes all audio files in the enrollment directory to create
        a voiceprint for each speaker.

        Args:
            enrollment_dir (str): Path to the directory containing speaker folders.
                                  Expected structure: enrollment_dir/{speaker_name}/{sample}.wav
        """
        print(f"Processing enrollment data from: {enrollment_dir}")
        if not os.path.isdir(enrollment_dir):
            print(f"Enrollment directory not found: {enrollment_dir}")
            return

        for speaker_name in os.listdir(enrollment_dir):
            speaker_dir = os.path.join(enrollment_dir, speaker_name)
            if os.path.isdir(speaker_dir):
                embeddings = []
                for sample_file in os.listdir(speaker_dir):
                    sample_path = os.path.join(speaker_dir, sample_file)
                    # Preprocess each sample to ensure correct format
                    processed_path = self._preprocess_audio(sample_path)
                    if processed_path:
                        # Extract embedding from the entire sample file
                        try:
                            embedding = self.embedding_model(
                                {"uri": sample_file, "audio": processed_path}
                            )
                            # The model returns a (1, 1, D) tensor, we squeeze it
                            embeddings.append(embedding.squeeze())
                        except Exception as e:
                            print(f"Could not create embedding for {sample_path}: {e}")
                        finally:
                            # Clean up the temporary file
                            if os.path.exists(processed_path):
                                os.remove(processed_path)

                if embeddings:
                    # Average the embeddings to create a single, robust voiceprint
                    avg_embedding = np.mean(np.array([e.cpu().numpy() for e in embeddings]), axis=0)
                    self.enrolled_speakers[speaker_name] = avg_embedding
                    print(f"Enrolled speaker '{speaker_name}' with a voiceprint.")
        print("Enrollment processing complete.")


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
            max_speakers (int, optional): The maximum number of speakers.

        Returns:
            list: A list of dictionaries, where each dictionary represents a
                  speaker segment. Returns an empty list on failure.
        """
        # 1. Preprocess the input audio
        processed_audio_path = self._preprocess_audio(audio_path)
        if not processed_audio_path:
            return []

        # 2. Run Speaker Diarization
        print("Running speaker diarization...")
        try:
            diarization_result = self.diarization_pipeline(
                processed_audio_path,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers
            )
        except Exception as e:
            print(f"Error during diarization: {e}")
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
            print(f"Mapped diarization label '{speaker_label}' to '{identified_name}' with confidence {speaker_map[speaker_label]['confidence']:.2f}")


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

        print("Pipeline run completed.")
        return output_segments

# --- Example Usage ---
if __name__ == '__main__':
    # IMPORTANT: You need a Hugging Face auth token to run this.
    # 1. Go to https://huggingface.co/settings/tokens
    # 2. Create a new token with "read" permissions.
    # 3. Paste the token here or set it as an environment variable.
    HF_AUTH_TOKEN = os.environ.get("HF_AUTH_TOKEN", "YOUR_HUGGING_FACE_TOKEN_HERE")

    if HF_AUTH_TOKEN == "YOUR_HUGGING_FACE_TOKEN_HERE":
        print("\n!!! WARNING: Please provide a Hugging Face authentication token. !!!\n")
    else:
        # --- Setup a mock environment for demonstration ---
        print("\n--- Setting up mock environment for demonstration ---")

        # Create a mock enrollment directory
        MOCK_ENROLL_DIR = "mock_enrollment_data"
        os.makedirs(os.path.join(MOCK_ENROLL_DIR, "Alice"), exist_ok=True)
        os.makedirs(os.path.join(MOCK_ENROLL_DIR, "Bob"), exist_ok=True)

        # Create dummy audio files for enrollment (requires ffmpeg to be installed)
        # In a real scenario, these would be actual audio recordings.
        try:
            AudioSegment.silent(duration=5000).export(os.path.join(MOCK_ENROLL_DIR, "Alice", "alice_sample1.wav"), format="wav")
            AudioSegment.silent(duration=5000).export(os.path.join(MOCK_ENROLL_DIR, "Bob", "bob_sample1.wav"), format="wav")
            # Create a dummy test file to process
            MOCK_TEST_FILE = "mock_test_audio.wav"
            AudioSegment.silent(duration=10000).export(MOCK_TEST_FILE, format="wav")
            print("Mock files created successfully. (Note: These are silent files for structure demonstration).")
            print("For a real test, replace these with actual audio files.")

            # --- Run the pipeline ---
            print("\n--- Initializing and running the Voxi AI Pipeline ---")
            voxi_pipeline = VoxiDiarizationPipeline(auth_token=HF_AUTH_TOKEN)

            # 1. Process enrollment data first
            voxi_pipeline.process_enrollment_data(MOCK_ENROLL_DIR)

            # 2. Run the full pipeline on a test audio file
            # Replace MOCK_TEST_FILE with a real audio file path for a meaningful result.
            print(f"\n--- Processing test file: {MOCK_TEST_FILE} ---")
            final_result = voxi_pipeline.run(MOCK_TEST_FILE)

            # 3. Print the final, structured result
            import json
            print("\n--- Final Result ---")
            print(json.dumps(final_result, indent=2))

        except Exception as e:
            print(f"\n--- DEMO FAILED ---")
            print(f"An error occurred during the example run: {e}")
            print("Please ensure you have 'ffmpeg' installed on your system for audio creation.")
            print("For a real test, you can manually create the directory structure and use your own audio files.")