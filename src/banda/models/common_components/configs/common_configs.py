from pydantic import BaseModel, ConfigDict # Import ConfigDict
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
    band_specs: BandsplitSpecification # This will be instantiated by Hydra

    require_no_overlap: bool
    require_no_gap: bool
    normalize_channel_independently: bool
    treat_channel_as_feature: bool

class TFModelConfig(BaseModel):
    # Common parameters for all TF models
    n_modules: int
    emb_dim: int
    dropout: float

    # RNN-specific parameters (optional)
    rnn_dim: Optional[int] = None
    bidirectional: Optional[bool] = None
    rnn_type: Optional[str] = None

    # Transformer-specific parameters (optional)
    nhead: Optional[int] = None
    dim_feedforward: Optional[int] = None

    # Mamba-specific parameters (optional)
    d_state: Optional[int] = None
    d_conv: Optional[int] = None
    expand: Optional[int] = None

    # Positional Encoding parameters (optional, for Transformer/Mamba)
    max_seq_len_bands: Optional[int] = None
    max_seq_len_time: Optional[int] = None

class MaskEstimationConfig(BaseModel):
    mlp_dim: int
    hidden_activation: str
    hidden_activation_kwargs: Optional[Dict] = None
    complex_mask: bool
    # Reverted: removed num_stems and changed input_channels back to in_channel
    in_channel: int 

class BanditSeparatorConfig(BaseModel):
    stems: List[str]
    fs: int
    emb_dim: int
    in_channel: int # This is the input audio channels
    stft_config: STFTConfig
    bandsplit_config: BandsplitModuleConfig
    tf_model: Any # Changed back to Any for now
    mask_estim_config: MaskEstimationConfig

class BanquetSeparatorConfig(BanditSeparatorConfig):
    query_features: int
    query_processor_type: str