import pytest
from vox_ai_backend.diarization_pipeline import VoxiDiarizationPipeline

# Use a dummy token for testing initialization
DUMMY_TOKEN = "dummy"

def test_invalid_file_type(tmp_path):
    pipeline = VoxiDiarizationPipeline(auth_token=DUMMY_TOKEN)
    # Create a dummy text file
    invalid_file = tmp_path / "invalid.txt"
    invalid_file.write_text("not audio")
    with pytest.raises(ValueError):
        pipeline._preprocess_audio(str(invalid_file))

def test_nonexistent_file():
    pipeline = VoxiDiarizationPipeline(auth_token=DUMMY_TOKEN)
    with pytest.raises(FileNotFoundError):
        pipeline._preprocess_audio("does_not_exist.wav")

def test_empty_file(tmp_path):
    pipeline = VoxiDiarizationPipeline(auth_token=DUMMY_TOKEN)
    empty_file = tmp_path / "empty.wav"
    empty_file.write_bytes(b"")
    with pytest.raises(ValueError):
        pipeline._preprocess_audio(str(empty_file))