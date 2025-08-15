import os
import torch
from typing import List, Tuple
from omegaconf import OmegaConf
import structlog

logger = structlog.get_logger(__name__)

from banda.models.modules.configs.bandsplit_configs._bandsplit_models import (
    FixedBandsplitSpecsConfig,
    VocalBandsplitSpecsConfig,
    BandDefinitionConfig,
    BandsplitType,
)
from banda.models.modules.spectral_components.spectral_base import BandsplitSpecification # Corrected import
from banda.models.modules.spectral_components.bandsplit_registry import bandsplit_registry

class FixedBandsplitSpecs(BandsplitSpecification):
    def __init__(self, nfft: int, fs: int, config: FixedBandsplitSpecsConfig, drop_dc_band: bool = False) -> None:
        super().__init__(nfft=nfft, fs=fs, drop_dc_band=drop_dc_band)
        self.config = config
        self.bands_config = config.bands # Expecting a list of band definitions

    def get_band_specs(self) -> List[Tuple[int, int]]:
        bands = []
        for band_def in self.bands_config:
            start_hz = band_def.start_hz
            end_hz = band_def.end_hz
            bandwidth_hz = band_def.bandwidth_hz

            start_index = self.hertz_to_index(start_hz)
            end_index = self.hertz_to_index(end_hz) if end_hz is not None else self.max_index

            if bandwidth_hz is None:
                # Treat as one subband
                bands.append((start_index, end_index))
            else:
                # Split by bandwidth
                bands.extend(self.get_band_specs_with_bandwidth(start_index, end_index, bandwidth_hz))
        
        self._validate_band_specs(bands)
        return bands

    @property
    def is_overlapping(self) -> bool:
        return False # Fixed bands are typically not overlapping


class VocalBandsplitSpecification(FixedBandsplitSpecs):
    def __init__(self, nfft: int, fs: int, config: VocalBandsplitSpecsConfig, version: str = "7", drop_dc_band: bool = False) -> None:
        script_dir = os.path.dirname(__file__)
        yaml_path = os.path.join(script_dir, "..", "configs", "bandsplit_configs", "vocal_bands.yaml")
        vocal_bands_config_data = OmegaConf.load(yaml_path)
        
        version_config_data = vocal_bands_config_data.get(f"version{version}")
        if version_config_data is None:
            raise ValueError(f"Configuration for Vocal Bandsplit version {version} not found in {yaml_path}.")
        
        bands_data = [BandDefinitionConfig(**band) for band in version_config_data.bands]

        if drop_dc_band and bands_data[0].start_hz == 0.0:
            bands_data[0].start_hz = BandsplitSpecification.index_to_hertz(0, nfft=nfft, fs=fs, drop_dc_band=True)
            logger.info(f"Adjusted first band start_hz to {bands_data[0].start_hz} due to drop_dc_band=True.")

        fixed_bands_config = FixedBandsplitSpecsConfig(bands=bands_data)
        
        super().__init__(nfft=nfft, fs=fs, config=fixed_bands_config, drop_dc_band=drop_dc_band)
        self.version = version

    def get_band_specs(self) -> List[Tuple[int, int]]:
        bands = super().get_band_specs()
        return bands


class OtherBandsplitSpecification(FixedBandsplitSpecs):
    def __init__(self, nfft: int, fs: int, config: FixedBandsplitSpecsConfig, drop_dc_band: bool = False) -> None:
        super().__init__(nfft=nfft, fs=fs, config=config, drop_dc_band=drop_dc_band)

    def get_band_specs(self) -> List[Tuple[int, int]]:
        bands = super().get_band_specs()
        return bands


class BassBandsplitSpecification(FixedBandsplitSpecs):
    def __init__(self, nfft: int, fs: int, config: FixedBandsplitSpecsConfig, drop_dc_band: bool = False) -> None:
        super().__init__(nfft=nfft, fs=fs, config=config, drop_dc_band=drop_dc_band)

    def get_band_specs(self) -> List[Tuple[int, int]]:
        bands = super().get_band_specs()
        return bands


class DrumBandsplitSpecification(FixedBandsplitSpecs):
    def __init__(self, nfft: int, fs: int, config: FixedBandsplitSpecsConfig, drop_dc_band: bool = False) -> None:
        super().__init__(nfft=nfft, fs=fs, config=config, drop_dc_band=drop_dc_band)

    def get_band_specs(self) -> List[Tuple[int, int]]:
        bands = super().get_band_specs()
        return bands

def register_fixed_bandsplit_specs():
    bandsplit_registry.register(BandsplitType.FIXED, FixedBandsplitSpecs)
    bandsplit_registry.register(BandsplitType.VOCAL, VocalBandsplitSpecification)
    bandsplit_registry.register(BandsplitType.VOCAL_V7, VocalBandsplitSpecification)
    bandsplit_registry.register(BandsplitType.OTHER, OtherBandsplitSpecification)
    bandsplit_registry.register(BandsplitType.BASS, BassBandsplitSpecification)
    bandsplit_registry.register(BandsplitType.DRUM, DrumBandsplitSpecification)