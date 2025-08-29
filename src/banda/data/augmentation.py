"""
This module provides audio augmentation utilities using `torch_audiomentations`.

Classes:
    SmartGain: A gain augmentation with dynamic range adjustment based on dBFS threshold.
    Audiomentations: A wrapper for predefined and custom audio augmentation pipelines.

Constants:
    PREDEFINED_AUGMENTATIONS: Dictionary of predefined augmentation pipelines.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel
import torch
import torch_audiomentations as audiomentations
from torch_audiomentations.utils.dsp import convert_decibels_to_amplitude_ratio
from torch_audiomentations.utils.object_dict import ObjectDict


class SmartGain(audiomentations.Gain):
    """
    Gain augmentation with dynamic range adjustment based on dBFS threshold.

    Args:
        p (float): Probability of applying the transform.
        min_gain_in_db (float): Minimum gain in decibels.
        max_gain_in_db (int): Maximum gain in decibels.
        dbfs_threshold (float): Threshold in dBFS for dynamic adjustment.
    """

    def __init__(
        self,
        p: float = 0.5,
        min_gain_in_db=-6,
        max_gain_in_db: int = 6,
        dbfs_threshold=-48.0,
    ) -> None:
        super().__init__(
            p=p,
            min_gain_in_db=min_gain_in_db,
            max_gain_in_db=max_gain_in_db,
        )
        self.dbfs_threshold = dbfs_threshold

    def randomize_parameters(
        self,
        samples: torch.Tensor,
        sample_rate: Optional[int] = None,
        targets: Optional[torch.Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> None:
        """
        Randomize gain parameters based on the dBFS of the input samples.

        Args:
            samples (Optional[Tensor]): Input audio samples of shape (batch_size, channels, samples).
            sample_rate (Optional[int]): Sample rate of the audio.
            targets (Optional[Tensor]): Target audio samples (unused).
            target_rate (Optional[int]): Target sample rate (unused).
        """
        dbfs = 10 * torch.log10(torch.mean(torch.square(samples)) + 1e-6)

        if dbfs > self.dbfs_threshold:
            low = max(self.min_gain_in_db, self.dbfs_threshold - float(dbfs))
        else:
            low = max(0.0, self.min_gain_in_db)

        distribution = torch.distributions.Uniform(
            low=torch.tensor(low, dtype=torch.float32, device=samples.device),
            high=torch.tensor(
                self.max_gain_in_db, dtype=torch.float32, device=samples.device
            ),
            validate_args=True,
        )
        selected_batch_size = samples.size(0)
        self.transform_parameters["gain_factors"] = (
            convert_decibels_to_amplitude_ratio(
                distribution.sample(sample_shape=(selected_batch_size,))
            )
            .unsqueeze(1)
            .unsqueeze(1)
        )


class AugmentationConfig(BaseModel):
    """
    Configuration for augmentation transforms.

    Args:
        cls (str): Class name of the augmentation transform.
        kwargs (dict): Keyword arguments for the transform.
    """

    cls: str
    kwargs: Dict[str, Any]


class ComposedAugmentationConfig(BaseModel):
    """
    Configuration for composed augmentation transforms.

    Args:
        transforms (List[AugmentationConfig]): List of augmentation configurations.
    """

    transforms: List[AugmentationConfig]


PREDEFINED_AUGMENTATIONS = {
    "gssp": [
        AugmentationConfig(
            cls="SmartGain",
            kwargs={
                "p": 1.0,
                "min_gain_in_db": -6,
                "max_gain_in_db": 6,
                "dbfs_threshold": -48.0,
            },
        ),
        AugmentationConfig(
            cls="TimeStretch",
            kwargs={"p": 1.0, "min_rate": 0.8, "max_rate": 1.2},
        ),
        AugmentationConfig(
            cls="PitchShift",
            kwargs={"p": 1.0, "min_semitones": -4, "max_semitones": 4},
        ),
    ]
}


class Audiomentations(audiomentations.Compose):
    """
    Wrapper for predefined and custom audio augmentation pipelines.

    Args:
        augment (str): Name of the predefined augmentation pipeline.
        fs (int): Sampling rate of the audio.
    """

    def __init__(self, augment: str = "gssp", fs: int = 44100) -> None:
        if isinstance(augment, str):
            if augment in PREDEFINED_AUGMENTATIONS:
                augment_config: List[AugmentationConfig] = PREDEFINED_AUGMENTATIONS[
                    augment
                ]
            else:
                raise ValueError(f"Unknown augmentation preset: {augment}")

        transforms = []

        for transform in augment_config:
            if transform.cls == "Gain":
                transforms.append(SmartGain(**transform.kwargs))
            else:
                transforms.append(
                    getattr(audiomentations, transform.cls)(**transform.kwargs)
                )

        super().__init__(transforms=transforms, shuffle=True)

        self.fs = fs

    def forward(
        self,
        samples: Optional[torch.Tensor] = None,
    ) -> ObjectDict:
        """
        Apply the augmentation pipeline to the input samples.

        Args:
            samples (Optional[torch.Tensor]): Input audio samples of shape (batch_size, channels, samples).

        Returns:
            ObjectDict: Augmented audio samples.
        """
        return super().forward(samples, sample_rate=self.fs)
