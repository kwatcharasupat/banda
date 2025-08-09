import torch
import torch.nn as nn
from typing import List, Tuple
from omegaconf import DictConfig
import hydra.utils

import structlog
import logging

logger = structlog.get_logger(__name__)
logging.getLogger(__name__).setLevel(logging.DEBUG)

from banda.models.common_components.configs.common_configs import BandsplitModuleConfig
from banda.models.common_components.spectral_components.spectral_base import BandsplitSpecification, get_bandsplit_specs_factory
from banda.models.common_components.configs.bandsplit_configs import BandsplitType # Import BandsplitType


class BandsplitModule(nn.Module):
    """
    A module for performing bandsplitting on a spectrogram.
    """
    def __init__(self, config: BandsplitModuleConfig) -> None:
        super().__init__()
        self.config = config

        # Explicitly instantiate band_specs as a Pydantic object
        instantiated_band_specs = hydra.utils.instantiate(config.band_specs, _recursive_=False)

        logger.info(f"bandsplit_type: {BandsplitType.MUSICAL}")
        logger.info(f"n_fft: {config.n_fft}")
        logger.info(f"fs: {config.fs}")
        logger.info(f"config.band_specs type: {type(instantiated_band_specs)}")
        logger.info(f"config.band_specs content: {instantiated_band_specs}")
        logger.info(f"drop_dc_band: {config.drop_dc_band}")
        logger.info(f"require_no_overlap: {config.require_no_overlap}")

        self.band_specs_factory = get_bandsplit_specs_factory(
            bandsplit_type=BandsplitType.MUSICAL, # Explicitly set bandsplit_type
            n_fft=config.n_fft, # Assuming n_fft is passed to BandsplitModuleConfig
            fs=config.fs, # Assuming fs is passed to BandsplitModuleConfig
            config=instantiated_band_specs, # Pass the instantiated Pydantic object
            drop_dc_band=config.drop_dc_band,
            require_no_overlap=config.require_no_overlap
        )
        self.band_specs = self.band_specs_factory.get_band_specs()

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Applies bandsplitting to the input spectrogram.

        Args:
            spectrogram (torch.Tensor): Input spectrogram.
                Shape: (batch_size, channels, freq_bins, time_frames, 2)

        Returns:
            torch.Tensor: Bandsplit features.
                Shape: (batch_size, channels, n_bands, time_frames, 2)
        """
        bandsplit_features = []
        for f_start, f_end in self.band_specs:
            band = spectrogram[..., f_start:f_end, :, :]
            bandsplit_features.append(band)
        
        # Concatenate along a new dimension for bands
        # Resulting shape: (batch_size, channels, n_bands, band_freq_bins, time_frames, 2)
        # This needs to be flattened or handled appropriately by the next module
        # For now, let's assume a simple concatenation and let the TF model handle reshaping
        
        # Pad bands to have the same frequency dimension if necessary, or handle variable length
        # For simplicity, let's assume fixed-size bands for now or handle padding later.
        # A common approach is to pad to the max band size or use a list of tensors.
        
        # For now, let's just return a list of tensors, and the TF model will handle it.
        # If the TF model expects a single tensor, we'll need to adjust this.
        return bandsplit_features # Returning a list of tensors for now
