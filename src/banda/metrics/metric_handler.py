#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.
#

from typing import Any, Dict, List, Optional, Union
import torch
from torch import nn
from pydantic import BaseModel, Field, ConfigDict

from banda.utils.registry import MetricRegistry

class MetricsConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    target_: str = Field(alias="_target_")
    name: str
    params: Optional[Dict[str, Any]] = None

class MetricHandler(nn.Module):
    def __init__(self, metric_configs: List[MetricsConfig]):
        super().__init__()
        self.metrics = nn.ModuleDict()

        for metric_cfg in metric_configs:
            metric_class = MetricRegistry.get(metric_cfg.target_) # Use MetricRegistry
            metric_fn = metric_class(**(metric_cfg.params or {}))
            self.metrics[metric_cfg.name] = metric_fn

    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        results = {}
        for metric_name, metric_fn in self.metrics.items():
            # This logic might need to be more sophisticated depending on your metrics
            if metric_name in predictions and metric_name in targets:
                results[metric_name] = metric_fn(predictions[metric_name], targets[metric_name])
            elif "mixture" in predictions and "mixture" in targets: # Example for a common case
                results[metric_name] = metric_fn(predictions["mixture"], targets["mixture"])
            else:
                raise KeyError(f"Metric '{metric_name}' cannot find corresponding predictions/targets.")
        return results

    @classmethod
    def from_config(cls, config: Union[MetricsConfig, List[MetricsConfig]]) -> "MetricHandler":
        if isinstance(config, list):
            return cls(config)
        else:
            return cls([config])