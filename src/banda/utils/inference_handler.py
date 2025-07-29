import torch
from typing import Dict, Any
from omegaconf import DictConfig
from banda.utils.inference import OverlapAddProcessor

class InferenceHandler:
    def __init__(self, model: torch.nn.Module, inference_config: DictConfig):
        self.model = model
        self.overlap_add_processor = OverlapAddProcessor(
            chunk_size_seconds=inference_config.chunk_size_seconds,
            hop_size_seconds=inference_config.hop_size_seconds,
            fs=inference_config.fs,
        )
        self.inference_batch_size = inference_config.batch_size

    def perform_inference(
        self,
        full_length_mixture_audio: torch.Tensor,
        query_for_inference: Any,
        device: torch.device,
        batch_size_cut_factor: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """
        Performs inference using the configured model and overlap-add processor.
        Handles OutOfMemoryError by reducing batch size and retrying.
        """
        try:
            separated_audio = self.overlap_add_processor.process(
                self.model,
                full_length_mixture_audio,
                query=query_for_inference,
                batch_size=self.inference_batch_size // batch_size_cut_factor,
                device=device,
            )
            return separated_audio
        except torch.cuda.OutOfMemoryError as e:
            if self.inference_batch_size // batch_size_cut_factor == 1:
                raise e  # Cannot reduce batch size further

            # Reduce batch size and retry
            new_batch_size_cut_factor = batch_size_cut_factor * 2
            torch.cuda.empty_cache()
            return self.perform_inference(
                full_length_mixture_audio,
                query_for_inference,
                device,
                new_batch_size_cut_factor,
            )