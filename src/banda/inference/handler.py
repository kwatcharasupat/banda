from banda.data.item import SourceSeparationBatch

import torch
from torch.nn import functional as F
import math
from typing import List

from torch import nn

from banda.utils import BaseConfig
from tqdm import tqdm
import torchaudio as ta

import structlog
logger = structlog.get_logger(__name__)

class InferenceHandlerParams(BaseConfig):
    fs: int
    chunk_size_seconds: float
    inference_batch_size: int | None = None
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
        self.window = torch.hann_window(
            self.chunk_size_samples,
        ).reshape(1, 1, self.chunk_size_samples) / scaler

        inference_batch_size = self.config.inference_batch_size
        if inference_batch_size is None:
            device_properties = torch.cuda.get_device_properties()
            total_memory_bytes = device_properties.total_memory
            total_memory_gb = total_memory_bytes / (1024**3)
            inference_batch_size = total_memory_gb * 0.75 # make a guess
            inference_batch_size = int(inference_batch_size // 4) * 4

        self.inference_batch_size = inference_batch_size

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
                    (n_samples - self.chunk_size_samples)
                    / self.hop_size_samples
                )
            )
            + 1
        )
        pad = (
            (n_chunks - 1) * self.hop_size_samples + self.chunk_size_samples - n_samples
        )

        if n_samples >= pad + 2 * self.overlap_samples:
            audio = F.pad(
                audio, pad=(2 * self.overlap_samples, 2 * self.overlap_samples + pad),
                mode="reflect"
            )  # _, n_channels, n_samples + pad + 4 * overlap
        else:
            logger.warn("Using multiple reflect due to insufficient audio length")
            n_reflect_front = 2 * self.overlap_samples // n_samples
            n_reflect_front_res = 2 * self.overlap_samples - n_samples * n_reflect_front
            n_reflect_back = (2 * self.overlap_samples + pad)// n_samples
            n_reflect_back_res = 2 * self.overlap_samples - n_samples * n_reflect_back

            for _ in range(n_reflect_front):
                audio = F.pad(
                    audio, pad=(n_samples, 0),
                    mode="reflect"
                )  
            
            for _ in range(n_reflect_back):
                audio = F.pad(
                    audio, pad=(0, n_samples),
                    mode="reflect"
                )  

            audio = F.pad(
                    audio, pad=(n_reflect_front_res, n_reflect_back_res),
                    mode="reflect"
                )  

        padded_samples = audio.shape[-1]
        assert padded_samples == (n_chunks - 1) * self.hop_size_samples + self.chunk_size_samples + 4 * self.overlap_samples
        audio = audio.reshape(
            n_channels, 1, -1, 1
        )  # _, n_channels, n_samples + pad + 4 * overlap, 1

        chunked_audio = F.unfold(
            audio,
            kernel_size=(self.chunk_size_samples, 1),
            stride=(self.hop_size_samples, 1),
        )  # n_channels, chunk_size, n_chunks

        chunked_audio = chunked_audio.permute(2, 0, 1).reshape(
            -1, n_channels, self.chunk_size_samples
        )  # n_chunks, n_channels, chunk_size

        return chunked_audio, padded_samples

    def chunk_batch(self, batch: SourceSeparationBatch) -> List[SourceSeparationBatch]:
        """
        Prepare chunked audio for inference.

        Args:
            batch (SourceSeparationBatch): Input batch.

        Returns:
            Tuple containing chunked audio and related parameters.
        """
        audio = batch.mixture["audio"]
        batch_size, _, n_samples = audio.shape
        assert batch_size == 1, "Batch size must be 1 for inference"
        assert n_samples == batch.n_samples.item(), f"Number of samples must match, got {n_samples} and {batch.n_samples.item()}"

        chunked_audio = self._chunk_audio(audio)

        return chunked_audio
    
    def to_inference_batch(self, chunked_audio: torch.Tensor, inference_batch_size: int = None):
        n_chunks = chunked_audio.shape[0]

        if inference_batch_size is None:
            inference_batch_size = self.inference_batch_size

        n_inference_batches = int(math.ceil(n_chunks / inference_batch_size))

        for i in tqdm(range(n_inference_batches), position=1):
            start_idx = i * inference_batch_size
            end_idx = min((i + 1) * inference_batch_size, n_chunks)

            yield SourceSeparationBatch(
                    mixture={"audio": chunked_audio[start_idx:end_idx]},
                    sources={},
                    estimates={},
                )
            

    def chunked_reconstruct(
        self,
        batches: List[SourceSeparationBatch],
        original_batch: SourceSeparationBatch,
        padded_samples: int
    ) -> SourceSeparationBatch:
        # check that all batches have the same sources
        for batch in batches:
            print(batch.estimates.keys())
        
        exit()
            # if list(batch.estimates.keys()) == list(batches[0].estimates.keys()):
            #     print("ok")
            # assert list(batch.estimates.keys()) == list(batches[0].estimates.keys()), (list(batch.estimates.keys()), list(batches[0].estimates.keys()))

        reconstructed_estimates = {}

        for source in batches[0].estimates:
            reconstructed_estimate = self._overlap_add(
                [batch.estimates[source]["audio"] for batch in batches], n_samples=original_batch.n_samples, padded_samples=padded_samples
            )
            reconstructed_estimates[source] = {"audio": reconstructed_estimate}

        original_batch.estimates = reconstructed_estimates
        return original_batch

    def _overlap_add(
        self, chunked_outputs: torch.Tensor, n_samples: int, padded_samples: int
    ) -> SourceSeparationBatch:
        output = torch.cat(chunked_outputs, dim=0)
        output = output * self.window.to(
            output.device
        )  # n_chunks, n_channels, chunk_size

        n_chunks, _, chunk_size = output.shape
        assert chunk_size == self.chunk_size_samples

        output = torch.permute(output, (1, 2, 0)) # n_channels, chunk_size, n_chunks

        output = F.fold(
            output,
            output_size=(padded_samples, 1),
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
