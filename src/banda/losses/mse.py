
import torch
from banda.data.batch_types import AudioSignal, SeparationBatch
from banda.utils.registry import LossRegistry
from torch import nn

@LossRegistry.register()
class SpectralMSELoss(nn.Module):
    """
    Mean Squared Error Loss
    """
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()


    def forward(self, batch: SeparationBatch):

        losses = {}

        for source_name, estimate in batch.estimates.items():
            source_name: str
            estimate: AudioSignal
            target : AudioSignal = batch.sources[source_name]
            loss = self.loss_fn(estimate.spectrogram, target.spectrogram)
            losses[source_name] = loss

        total_loss = sum(losses.values())

        return total_loss, losses