import sys
import os
import pytest
import torch
import torch.nn as nn
import math

# Add the src directory to the Python path to enable module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from banda.utils.inference import OverlapAddProcessor

# Mock model for testing
class MockSeparator(nn.Module):
    def __init__(self, expected_output_stems: list):
        super().__init__()
        self.expected_output_stems = expected_output_stems

    def forward(self, audio_chunk: torch.Tensor, **kwargs) -> dict:
        # Simulate separation: return a dictionary of tensors with the same shape as input
        # For simplicity, we'll just return the input chunk for each expected stem
        separated_stems = {stem: audio_chunk.clone() for stem in self.expected_output_stems}
        return separated_stems

@pytest.fixture
def sample_audio():
    # Stereo audio, 44100 samples (1 second)
    return torch.randn(2, 44100)

@pytest.fixture
def overlap_add_processor():
    # chunk_size_seconds, hop_size_seconds, fs
    return OverlapAddProcessor(chunk_size_seconds=0.5, hop_size_seconds=0.25, fs=44100)

def test_overlap_add_processor_chunking(overlap_add_processor, sample_audio):
    chunked_audio, padded_length, original_length = overlap_add_processor._chunk_audio(sample_audio)

    # Assertions for chunking
    assert original_length == sample_audio.shape[1]
    assert chunked_audio.ndim == 3 # (n_chunks, channels, chunk_size_samples)
    assert chunked_audio.shape[1] == sample_audio.shape[0] # channels
    assert chunked_audio.shape[2] == overlap_add_processor.chunk_size_samples # chunk_size_samples

    # Check if the total length of chunks covers the padded length
    expected_n_chunks = math.ceil((padded_length - overlap_add_processor.chunk_size_samples) / overlap_add_processor.hop_size_samples) + 1
    assert chunked_audio.shape[0] == expected_n_chunks

def test_overlap_add_processor_reconstruction(overlap_add_processor, sample_audio):
    chunked_audio, padded_length, original_length = overlap_add_processor._chunk_audio(sample_audio)

    # Simulate processed chunks (e.g., from a model)
    # For simplicity, we'll use the chunked_audio itself as processed_chunks
    processed_chunks_dict = {"vocals": chunked_audio, "bass": chunked_audio}

    reconstructed_stems = overlap_add_processor._reconstruct_audio(
        processed_chunks_dict, padded_length, original_length
    )

    # Assertions for reconstruction
    assert isinstance(reconstructed_stems, dict)
    assert "vocals" in reconstructed_stems
    assert "bass" in reconstructed_stems

    for stem, audio in reconstructed_stems.items():
        assert audio.ndim == 2 # (channels, samples)
        assert audio.shape[0] == sample_audio.shape[0] # channels
        assert audio.shape[1] == original_length # original length

def test_overlap_add_processor_process(overlap_add_processor, sample_audio):
    mock_model = MockSeparator(expected_output_stems=["vocals", "bass"])
    
    # Move audio to CPU for initial processing, then it will be moved to device by processor
    separated_audio = overlap_add_processor.process(
        mock_model, sample_audio.cpu(), batch_size=2, device=torch.device("cpu")
    )

    # Assertions for end-to-end process
    assert isinstance(separated_audio, dict)
    assert "vocals" in separated_audio
    assert "bass" in separated_audio

    for stem, audio in separated_audio.items():
        assert audio.ndim == 2
        assert audio.shape[0] == sample_audio.shape[0]
        assert audio.shape[1] == sample_audio.shape[1]

    # Test with a different device (e.g., mps if available, otherwise still cpu)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    separated_audio_device = overlap_add_processor.process(
        mock_model, sample_audio.to(device), batch_size=2, device=device
    )
    for stem, audio in separated_audio_device.items():
        assert audio.device == torch.device("cpu") # Reconstructed audio is moved back to CPU