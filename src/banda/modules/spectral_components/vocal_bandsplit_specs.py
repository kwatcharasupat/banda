import os
import torch
from typing import List, Tuple
from omegaconf import OmegaConf
from banda.models.modules.configs.bandsplit_configs._bandsplit_models import VocalBandsplitSpecsConfig, FixedBandsplitSpecsConfig, BandDefinitionConfig
from banda.models.modules.spectral_components.spectral_base import BandsplitSpecification
from banda.models.modules.spectral_components.fixed_bandsplit_specs import FixedBandsplitSpecs # Import the base class

import structlog
import logging

logger = structlog.get_logger(__name__)
logging.getLogger(__name__).setLevel(logging.DEBUG)

class VocalBandsplitSpecification(FixedBandsplitSpecs):
    def __init__(self, nfft: int, fs: int, config: VocalBandsplitSpecsConfig, version: str = "7", drop_dc_band: bool = False) -> None:
        # Load bands from YAML file
        script_dir = os.path.dirname(__file__)
        yaml_path = os.path.join(script_dir, "..", "configs", "bandsplit_configs", "vocal_bands.yaml")
        vocal_bands_config_data = OmegaConf.load(yaml_path)
        
        # Dynamically select the version's bands from the loaded config
        version_config_data = vocal_bands_config_data.get(f"version{version}")
        if version_config_data is None:
            raise ValueError(f"Configuration for Vocal Bandsplit version {version} not found in {yaml_path}.")
        
        # Convert DictConfig to list of BandDefinitionConfig
        bands_data = [BandDefinitionConfig(**band) for band in version_config_data.bands]

        # Adjust start_hz of the first band if drop_dc_band is True and start_hz is 0.0
        if drop_dc_band and bands_data[0].start_hz == 0.0:
            bands_data[0].start_hz = BandsplitSpecification.index_to_hertz(0, nfft=nfft, fs=fs, drop_dc_band=True)
            logger.info(f"Adjusted first band start_hz to {bands_data[0].start_hz} due to drop_dc_band=True.")

        # Create a FixedBandsplitSpecsConfig with the dynamically loaded bands
        fixed_bands_config = FixedBandsplitSpecsConfig(bands=bands_data)
        
        # Pass this new config to the superclass
        super().__init__(nfft=nfft, fs=fs, config=fixed_bands_config, drop_dc_band=drop_dc_band)
        self.version = version

    def get_band_specs(self) -> List[Tuple[int, int]]:
        # This logic is now handled in __init__ by creating a FixedBandsplitSpecsConfig
        # So, we can just call the superclass method directly.
        bands = super().get_band_specs()
        return bands