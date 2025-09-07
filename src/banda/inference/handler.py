from banda.data.item import SourceSeparationBatch

import torch
from torch.nn import functional as F
import math
from typing import Any, List, Tuple

from torch import nn

from banda.utils import BaseConfig


class InferenceHandlerParams(BaseConfig):
    fs: int
    chunk_size_seconds: float
    inference_batch_size: int
    hop_size_seconds: float = 1.0


class InferenceHandler(nn.Module):
    """
    Handles inference-related operations such as chunked processing and audio saving.

    Args:
        fs (int): Sampling frequency.
        chunk_size_seconds (float): Chunk size in seconds.
        hop_size_seconds (float): Hop size in seconds.
        batch_size (int): Batch size for inference.
        _verbose (bool): If True, enables verbose logging.
    """

    def __init__(
        self,
        *,
        config: InferenceHandlerParams,
    ) -> None:
        super().__init__()

        self.config = config

        self.fs = config.fs
        self.chunk_size_seconds = config.chunk_size_seconds
        self.hop_size_seconds = config.hop_size_seconds

        self.chunk_size_samples = int(self.chunk_size_seconds * self.fs)
        self.hop_size_samples = int(self.hop_size_seconds * self.fs)
        self.overlap_samples = self.chunk_size_samples - self.hop_size_samples

        scaler = self.chunk_size_samples / (2 * self.hop_size_samples)
        window = torch.hann_window(
            self.chunk_size_samples,
        ).reshape(1, 1, self.chunk_size_samples)

        self.register_buffer("window", window * scaler)

        self.inference_batch_size = self.config.inference_batch_size

    def save_audio(
        self,
        batch: SourceSeparationBatch,
        log_dir: str,
    ) -> None:
        """
        Save audio to disk, either single-stem or multi-stem.

        Args:
            batch (BatchedInputOutput): Batch containing audio data.
            log_dir (str): Directory to save audio files.
            batch_idx (Optional[int]): Batch index (used for single-stem audio).
            single_stem (bool): If True, save single-stem audio; otherwise, save multi-stem audio.
        """
        batch_size = batch["mixture"]["audio"].shape[0]
        assert batch_size == 1, "Batch size must be 1 for inference"
        raise NotImplementedError("Audio saving is not implemented yet.")

    def _chunk_audio(
        self,
        audio: torch.Tensor,
    ) -> torch.Tensor:
        _, n_channels, n_samples = audio.shape

        n_chunks = (
            int(
                math.ceil(
                    (n_samples + 4 * self.overlap_samples - self.chunk_size_samples)
                    / self.hop_size_samples
                )
            )
            + 1
        )
        pad = (
            (n_chunks - 1) * self.hop_size_samples + self.chunk_size_samples - n_samples
        )

        audio = F.pad(
            audio, pad=(2 * self.overlap_samples, 2 * self.overlap_samples + pad)
        )  # _, n_channels, n_samples + pad + 4 * overlap

        audio = audio.reshape(
            n_channels, 1, -1, 1
        )  # _, n_channels, n_samples + pad + 4 * overlap, 1

        chunked_audio = F.unfold(
            audio,
            kernel_size=(self.chunk_size_samples, 1),
            stride=(self.hop_size_samples, 1),
        )  # _, n_channels * chunk_size, n_chunks

        chunked_audio = chunked_audio.permute(2, 0, 1).reshape(
            -1, n_channels, self.chunk_size_samples
        )  # n_chunks, n_channels, chunk_size

        return chunked_audio

    def chunk_batch(self, batch: SourceSeparationBatch) -> List[SourceSeparationBatch]:
        """
        Prepare chunked audio for inference.

        Args:
            batch (SourceSeparationBatch): Input batch.

        Returns:
            Tuple containing chunked audio and related parameters.
        """
        audio = batch.mixture.audio
        batch_size, _, n_samples = audio.shape
        assert batch_size == 1, "Batch size must be 1 for inference"
        assert n_samples == batch.n_samples.item(), "Number of samples must match"

        chunked_audio = self._chunk_audio(audio.cpu())
        n_chunks = chunked_audio.shape[0]

        n_inference_batches = int(math.ceil(n_chunks / self.inference_batch_size))

        batches = []

        for i in range(n_inference_batches):
            start_idx = i * self.inference_batch_size
            end_idx = min((i + 1) * self.inference_batch_size, n_chunks)
            batches.append(
                SourceSeparationBatch(
                    mixture={"audio": chunked_audio[start_idx:end_idx]},
                    sources={},
                    estimates={},
                    n_samples=torch.tensor(n_samples, dtype=torch.int64),
                )
            )

        return batches

    def chunked_reconstruct(
        self,
        batches: List[SourceSeparationBatch],
    ) -> SourceSeparationBatch:
        # check that all batches have the same sources
        for batch in batches:
            assert batch.estimates.keys() == batches[0].estimates.keys()

        reconstructed_estimates = {}

        for source in batches[0].estimates:
            chunked_estimate = torch.cat(
                [batch["estimates"][source]["audio"] for batch in batches], dim=0
            )
            reconstructed_estimate = self._overlap_add(
                chunked_estimate, n_samples=batches[0].n_samples.item()
            )
            reconstructed_estimates[source] = {"audio": reconstructed_estimate}

        reconstructed_batch = batch.model_copy()
        reconstructed_batch["estimates"] = reconstructed_estimates
        return reconstructed_batch

    def _overlap_add(
        self, chunked_outputs: torch.Tensor, n_samples: int
    ) -> SourceSeparationBatch:
        output = torch.cat(chunked_outputs, dim=0)
        output = output * self.window.to(
            output.device
        )  # n_chunks, n_channels, chunk_size

        n_chunks, _, chunk_size = output.shape
        assert chunk_size == self.chunk_size_samples

        output = torch.permute(output, (1, 2, 0))

        padded_length = (n_chunks - 1) * self.hop_size_samples + self.chunk_size_samples

        output = F.fold(
            output,
            output_size=(padded_length, 1),
            kernel_size=(self.chunk_size_samples, 1),
            stride=(self.hop_size_samples, 1),
        )
        return output[
            None,
            :,
            0,
            2 * self.overlap_samples : n_samples + 2 * self.overlap_samples,
            0,
        ]
