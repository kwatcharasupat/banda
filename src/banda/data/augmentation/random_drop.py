from banda.data.augmentation.base import BaseRegisteredAugmentation
from audiomentations.core.transforms_interface import BaseWaveformTransform
from numpy.typing import NDArray
import numpy as np


class RandomDrop(BaseRegisteredAugmentation, BaseWaveformTransform):
    def __init__(self, *, config):
        super().__init__(config=config)
        self.p = self.config.p

    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int):
        super().randomize_parameters(samples, sample_rate)

    def apply(
        self, samples: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        return np.zeros_like(samples)
