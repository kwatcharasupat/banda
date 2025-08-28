from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Union, Dict, Any, List
from omegaconf import DictConfig # Import DictConfig

# Import all referenced Pydantic configs directly
from banda.data.datamodule import DataSplitConfig, DataModuleConfig # Import DataModuleConfig
from banda.losses.loss_handler import LossConfig
from banda.metrics.metric_handler import MetricsConfig
# OptimizerConfig, LoggerConfig, TrainerConfig are defined in this file, so no import needed for them


class LoggerConfig(BaseModel):
    target_: str = Field(alias="_target_")
    project: str
    log_model: bool # Changed to bool
    save_dir: Optional[str] = None

class TrainerConfig(BaseModel):
    target_: str = Field(alias="_target_")
    accelerator: str
    devices: Any
    max_epochs: int
    precision: int
    log_every_n_steps: int

class OptimizerConfig(BaseModel): # Added OptimizerConfig definition
    target_: str = Field(alias="_target_")
    lr: float

class TrainConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    project_root: str
    experiment_name: str
    seed: int
    log_level: str
    paths: Dict[str, str] # Reverted to Dict[str, str]
    data: DataModuleConfig # Changed to DataModuleConfig
    model: Any # Direct reference
    loss: List[LossConfig] # Changed to List[LossConfig]
    metrics: List[MetricsConfig] # Changed to List[MetricsConfig]
    optimizer: OptimizerConfig # Direct reference
    trainer: TrainerConfig # Direct reference
    logger: LoggerConfig # Direct reference

# Define a main configuration schema for Hydra
class MainConfig(BaseModel):
    project_root: str
    experiment_name: str
    seed: int
    log_level: str
    paths: Dict[str, str] # Reverted to Dict[str, str]
    data: DataModuleConfig # Changed to DataModuleConfig
    model: Any
    loss: List[LossConfig] # Changed to List[LossConfig]
    metrics: List[MetricsConfig] # Changed to List[MetricsConfig]
    optimizer: OptimizerConfig
    trainer: TrainerConfig
    logger: LoggerConfig

# Removed TrainConfig.model_rebuild() from here
