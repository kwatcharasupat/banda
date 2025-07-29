import torch
import torch.nn.functional as F
import math
from typing import Dict, Any, Optional, Tuple

class OverlapAddProcessor:
    def __init__(self, chunk_size_seconds: float, hop_size_seconds: float, fs: int):
        self.chunk_size_samples = int(chunk_size_seconds * fs)
        self.hop_size_samples = int(hop_size_seconds * fs)
        self.fs = fs

        self.overlap_samples = self.chunk_size_samples - self.hop_size_samples
        
        # Pre-compute Hann window and scaler
        self.window = torch.hann_window(self.chunk_size_samples)
        self.scaler = self.chunk_size_samples / (2 * self.hop_size_samples)

    def _chunk_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Chunks a single audio signal into overlapping segments.

        Args:
            audio (torch.Tensor): The input audio tensor of shape (channels, samples).

        Returns:
            Tuple[torch.Tensor, int, int]: A tuple containing:
                - chunked_audio (torch.Tensor): Tensor of shape (n_chunks, channels, chunk_size_samples).
                - padded_length (int): The length of the audio after padding.
                - original_length (int): The original length of the audio.
        """
        channels, original_length = audio.shape

        # Calculate padding
        # The coda example uses 2 * overlap at the start and 2 * overlap + pad at the end
        # This ensures that the first and last chunks are fully processed
        n_chunks_needed = int(math.ceil((original_length - self.chunk_size_samples) / self.hop_size_samples)) + 1
        
        # Calculate total length required for unfolding
        required_length = (n_chunks_needed - 1) * self.hop_size_samples + self.chunk_size_samples
        
        # Calculate padding needed at the end
        pad_end = required_length - original_length

        # Apply symmetric padding at the beginning and calculated padding at the end
        # The 2 * overlap at the start and end is from the coda example
        audio = F.pad(audio, (2 * self.overlap_samples, 2 * self.overlap_samples + pad_end))
        padded_length = audio.shape[-1]

        # Reshape for unfold: (batch_size, channels, 1, samples)
        audio = audio.reshape(1, channels, 1, padded_length)

        # Unfold into chunks: (1, channels * chunk_size_samples, n_chunks)
        chunked_audio_unfolded = F.unfold(
            audio, kernel_size=(1, self.chunk_size_samples), stride=(1, self.hop_size_samples)
        )

        # Reshape to (n_chunks, channels, chunk_size_samples)
        n_chunks = chunked_audio_unfolded.shape[-1]
        chunked_audio = chunked_audio_unfolded.reshape(channels, self.chunk_size_samples, n_chunks).permute(2, 0, 1)

        return chunked_audio, padded_length, original_length

    def _reconstruct_audio(
        self,
        processed_chunks: Dict[str, torch.Tensor],
        padded_length: int,
        original_length: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Reconstructs full audio signals from processed chunks using overlap-add.

        Args:
            processed_chunks (Dict[str, torch.Tensor]): Dictionary where keys are stem names
                and values are tensors of processed chunks for that stem,
                shape (total_processed_chunks, channels, chunk_size_samples).
            padded_length (int): The length of the audio after padding during chunking.
            original_length (int): The original length of the audio before padding.

        Returns:
            Dict[str, torch.Tensor]: Dictionary where keys are stem names and values are
                the full-length reconstructed audio tensors of shape (channels, original_samples).
        """
        reconstructed_stems = {}
        for stem, chunks in processed_chunks.items():
            # Apply window and scale
            windowed_chunks = chunks * self.window.to(chunks.device) / self.scaler

            # Reshape for fold: (channels * chunk_size_samples, n_chunks)
            channels = windowed_chunks.shape[1]
            windowed_chunks_folded = windowed_chunks.permute(1, 2, 0).reshape(channels * self.chunk_size_samples, -1)

            # Fold back into full audio
            reconstructed_audio_padded = F.fold(
                windowed_chunks_folded,
                output_size=(channels, padded_length),
                kernel_size=(channels, self.chunk_size_samples),
                stride=(channels, self.hop_size_samples),
            )
            
            # Remove extra dimensions and crop padding
            reconstructed_audio = reconstructed_audio_padded.squeeze(0)[:, 2 * self.overlap_samples : 2 * self.overlap_samples + original_length]
            reconstructed_stems[stem] = reconstructed_audio
        
        return reconstructed_stems

    @torch.inference_mode()
    def process(
        self,
        model: torch.nn.Module,
        audio: torch.Tensor,
        batch_size: int,
        device: torch.device,
        **model_kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Performs overlap-add inference on a given audio signal using the provided model.

        Args:
            model (torch.nn.Module): The separation model to use for inference.
            audio (torch.Tensor): The input audio tensor (mixture) of shape (channels, samples).
            batch_size (int): The batch size for processing chunks.
            device (torch.device): The device to perform computations on.
            **model_kwargs: Additional keyword arguments to pass to the model's forward method.

        Returns:
            Dict[str, torch.Tensor]: Dictionary where keys are stem names and values are
                the full-length separated audio tensors.
        """
        model.eval()
        model.to(device)
        self.window = self.window.to(device) # Ensure window is on the correct device

        chunked_audio, padded_length, original_length = self._chunk_audio(audio.to(device))
        n_chunks = chunked_audio.shape[0]

        # Use defaultdict to accumulate chunks for each stem
        processed_chunks_by_stem = {}

        for i in range(0, n_chunks, batch_size):
            start = i
            end = min(i + batch_size, n_chunks)
            
            batch_of_chunks = chunked_audio[start:end]

            # Pass model_kwargs to the model's forward method
            # Assuming model.forward can handle these kwargs
            separated_chunks_batch = model(batch_of_chunks, **model_kwargs)

            for stem, chunks in separated_chunks_batch.items():
                if stem not in processed_chunks_by_stem:
                    processed_chunks_by_stem[stem] = []
                processed_chunks_by_stem[stem].append(chunks.cpu()) # Move to CPU to avoid OOM during accumulation

        # Concatenate all chunks for each stem
        final_processed_chunks = {
            stem: torch.cat(chunks_list, dim=0)
            for stem, chunks_list in processed_chunks_by_stem.items()
        }

        reconstructed_stems = self._reconstruct_audio(
            final_processed_chunks, padded_length, original_length
        )
        
        return reconstructed_stems