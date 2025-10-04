from typing import Dict, List
from omegaconf import DictConfig
from torch import nn
import torch
from banda.data.item import SourceSeparationBatch

from banda.metrics.base import MetricConfig, MetricRegistry
from banda.utils import BaseConfig

from torchmetrics import MetricCollection


class MetricHandlerParams(BaseConfig):
    metrics: List[MetricConfig]
    stems: List[str] = None

    auto_group: bool = True


class MetricHandler(nn.Module):
    def __init__(self, *, config: DictConfig):
        super().__init__()
        self.config = MetricHandlerParams.model_validate(config)

        assert self.config.stems or self.config.stem_agnostic, (
            "Either `stems` must be provided or `stem_agnostic` must be True."
        )

        self._build_metrics()

    def _build_metrics(self):
        metric_collection = {}
        for stem in self.config.stems:
            metric_collection[stem] = {}
            for metric_config in self.config.metrics:
                cls_str = metric_config.cls
                cls = MetricRegistry.get_registry().get(cls_str, None)
                if cls is None:
                    raise ValueError(
                        f"Metric class '{cls_str}' not found in registry. \n Available classes: {MetricRegistry.get_registry().keys()}"
                    )

                metric = cls(**metric_config.params.model_dump())

                metric_collection[stem][cls_str] = metric

        self.metric_collection = nn.ModuleDict(
            {
                stem: MetricCollection(metrics)
                for stem, metrics in metric_collection.items()
            }
        )

    def reset(self):
        for key in self.metric_collection:
            self.metric_collection[key].reset()

    def compute(self) -> Dict[str, float]:
        with torch.no_grad():
            out = {
                stem: self.metric_collection[stem].compute()
                for stem in self.metric_collection
            }

            out = {
                f"{stem}/{key}": out[stem][key].detach().cpu()
                for stem in out
                for key in out[stem]
            }

        self.reset()
        return out

    def update(self, batch: SourceSeparationBatch):
        with torch.no_grad():
            for key in batch.estimates:
                estimate = batch.estimates[key]["audio"]
                source = batch.sources.get(key, {}).get("audio", None)
                if source is None:
                    source = torch.zeros_like(estimate, requires_grad=False)

                if self.config.auto_group and key not in self.metric_collection:
                    key = key.split("/")[0]

                self.metric_collection[key].update(estimate, source)
