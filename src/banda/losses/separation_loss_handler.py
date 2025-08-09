import torch
import torch.nn as nn
from typing import Dict, List, Union
import structlog
from omegaconf import OmegaConf, DictConfig
import hydra.utils


logger = structlog.get_logger(__name__)


from banda.data.batch_types import (
    SeparationBatch,
    FixedStemSeparationBatch,
    BaseSeparationBatch, # Import the new base class
    QueryAudioSeparationBatch,
    QueryClassSeparationBatch,
    AudioSignal, # Import AudioSignal
)
from banda.losses.base import LossHandler
from banda.losses.loss_configs import LossCollectionConfig


class SeparationLossHandler(LossHandler):
    """
    Handles loss calculation for various source separation tasks based on batch type.
    """

    def __init__(self, loss_config: DictConfig): # Changed type hint to DictConfig
        """
        Args:
            loss_config (DictConfig): A Hydra DictConfig containing
                                      all necessary parameters for the loss functions.
        """
        super().__init__(None) # Pass None to base LossHandler
        
        # Explicitly create LossCollectionConfig from the DictConfig
        parsed_loss_config = LossCollectionConfig(**loss_config)

        self.loss_fns = nn.ModuleDict()
        self.loss_weights = {}
        for name, config in parsed_loss_config.losses.items(): # Use parsed_loss_config
            # Manually instantiate the loss function using class_target
            loss_class = hydra.utils.get_class(config.fn.class_target)
            loss_params = {k: v for k, v in config.fn.items() if k != "class_target"}
            self.loss_fns[name] = loss_class(**loss_params)
            self.loss_weights[name] = config.weight


    def calculate_loss(
        self,
        predictions: Dict[str, AudioSignal], # Changed type hint to AudioSignal
        batch: SeparationBatch,
    ) -> torch.Tensor:
        """
        Calculates the loss based on the provided predictions and batch.
        This method is generalized to handle both fixed-stem and query-based separation,
        as both models now return Dict[str, AudioSignal].

        Args:
            predictions (Dict[str, AudioSignal]): A dictionary of predicted source AudioSignal objects.
            batch (SeparationBatch): The input batch containing true sources and other data.

        Returns:
            torch.Tensor: The calculated loss.
        """
        # Initialize total_loss on the correct device
        if not predictions:
            return torch.tensor(0.0) # No predictions, no loss

        # Get device from the first available audio tensor in predictions
        first_prediction_key = next(iter(predictions))
        if predictions[first_prediction_key].audio is not None:
            device = predictions[first_prediction_key].audio.device
        elif predictions[first_prediction_key].spectrogram is not None:
            device = predictions[first_prediction_key].spectrogram.device
        else:
            # Fallback if no audio or spectrogram is present in any prediction
            device = torch.device("cpu") 
        total_loss = torch.tensor(0.0, device=device)

        true_sources = batch.sources # true_sources is Dict[str, AudioSignal]
        num_sources_with_predictions = 0

        # Iterate over predictions and match with true sources
        for source_name, sep_audio_signal in predictions.items(): # sep_audio_signal is AudioSignal
            if source_name in true_sources:
                true_audio_signal = true_sources[source_name] # true_audio_signal is AudioSignal
                num_sources_with_predictions += 1
                for loss_name, loss_fn in self.loss_fns.items():
                    weight = self.loss_weights.get(loss_name, 1.0)
                    
                    # Use audio or spectrogram for loss calculation based on availability
                    if sep_audio_signal.audio is not None and true_audio_signal.audio is not None:
                        loss = loss_fn(sep_audio_signal.audio, true_audio_signal.audio)
                    elif sep_audio_signal.spectrogram is not None and true_audio_signal.spectrogram is not None:
                        loss = loss_fn(sep_audio_signal.spectrogram, true_audio_signal.spectrogram)
                    else:
                        logger.warning(f"Skipping loss for {source_name} due to missing audio/spectrogram in AudioSignal.")
                        continue
                    total_loss += loss * weight
        
        if num_sources_with_predictions > 0:
            return total_loss / num_sources_with_predictions
        else:
            return torch.tensor(0.0, device=device) # Return 0 loss on the correct device if no matching sources

    def to(self, device: torch.device) -> None:
        """
        Moves all internal loss functions to the specified device.

        Args:
            device (torch.device): The target device.
        """
        logger.debug(f"SeparationLossHandler.to: Moving loss functions to device {device}")
        for loss_fn in self.loss_fns.values():
            loss_fn.to(device)