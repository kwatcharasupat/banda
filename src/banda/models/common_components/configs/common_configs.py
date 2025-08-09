from pydantic import BaseModel, ConfigDict, Field # Import Field
from typing import Dict, List, Optional, Union, Any

from banda.models.common_components.spectral_components.spectral_base import BandsplitSpecification # Import BandsplitSpecification (now in models folder)
from banda.core.interfaces import BaseTimeFrequencyModel # Import BaseTimeFrequencyModel

# Pydantic models for Separator configuration

class STFTConfig(BaseModel):
    n_fft: int
    win_length: Optional[int] = None
    hop_length: int
    window_fn: str
    wkwargs: Optional[Dict] = None
    power: Optional[int] = None
    center: bool
    normalized: bool
    pad_mode: str
    onesided: bool

class BandsplitModuleConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True) # Allow arbitrary types for BandsplitSpecification
    # band_specs will be an instantiated BandsplitSpecification object
    band_specs: Any # Changed type hint to Any to bypass strict Pydantic validation

    require_no_overlap: bool
    require_no_gap: bool
    normalize_channel_independently: bool
    treat_channel_as_feature: bool
    drop_dc_band: bool = True # Added drop_dc_band to BandsplitModuleConfig

class BaseTFModelConfig(BaseModel):
    # Common parameters for all TF models
    target_: str = Field(alias="_target_") # Add target_ field with alias for _target_
    n_modules: int
    emb_dim: int
    dropout: float
    # Positional Encoding parameters (optional, for Transformer/Mamba)
    max_seq_len_bands: Optional[int] = None
    max_seq_len_time: Optional[int] = None

class RNNTFModelConfig(BaseTFModelConfig):
    rnn_dim: int
    bidirectional: bool
    rnn_type: str
    use_layer_norm: bool = True # Added use_layer_norm with default value

class TransformerTFModelConfig(BaseTFModelConfig):
    nhead: int
    dim_feedforward: int

class MambaTFModelConfig(BaseTFModelConfig):
    d_state: int
    d_conv: int
    expand: int

from banda.models.common_components.configs.mask_estimation_configs import MaskEstimationConfig 

class BaseSeparatorConfig(BaseModel): # New BaseSeparatorConfig
    stems: List[str]
    fs: int
    emb_dim: int
    in_channel: int # This is the input audio channels
    stft: STFTConfig # Renamed from stft_config
    bandsplit: BandsplitModuleConfig # Renamed from bandsplit_config
    tfmodel: Union[RNNTFModelConfig, TransformerTFModelConfig, MambaTFModelConfig] # Renamed from tf_model
    mask_estim: MaskEstimationConfig # Renamed from mask_estim_config

class BanditSeparatorConfig(BaseSeparatorConfig): # Inherit from BaseSeparatorConfig
    pass # No additional fields for Bandit beyond BaseSeparatorConfig

class BanquetSeparatorConfig(BaseSeparatorConfig): # Inherit from BaseSeparatorConfig
    query_features: int
    query_processor_type: str