#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.
import os
from rich import print as rprint

import random
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
import pytorch_lightning.callbacks as pl_callbacks
from pytorch_lightning.loggers.wandb import WandbLogger
import torch
import hydra.utils
from torch import nn
import structlog
from pytorch_lightning.utilities.seed import isolate_rng
from banda.data.base import DataConfig, SourceSeparationDataModule
from banda.inference.handler import InferenceHandler, InferenceHandlerParams
from banda.losses.handler import LossHandler, LossHandlerConfig
from banda.metrics.handler import MetricHandler, MetricHandlerParams
from banda.models.base import ModelRegistry
from banda.system.base import SourceSeparationSystem
from banda.utils import BaseConfig, WithClassConfig
from hydra.core.hydra_config import HydraConfig
import time
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder
from banda.data.item import MultiDomainSignal, SourceSeparationBatch


logger = structlog.get_logger(__name__)


torch.set_float32_matmul_precision("high")


class TrainingConfig(BaseConfig):
    seed: int

    data: DataConfig
    model: WithClassConfig[BaseConfig]
    loss: LossHandlerConfig
    metrics: MetricHandlerParams
    inference: InferenceHandlerParams = None

    trainer: BaseConfig

    ckpt_path: str | None = None
    run_training: bool = True
    run_evaluation: bool = False


def _build_model(config: WithClassConfig[BaseConfig]) -> nn.Module:
    cls_str = config.cls

    cls = ModelRegistry.get_registry().get(cls_str, None)

    if cls is None:
        raise ValueError(
            f"Unknown model class: {cls_str}. Available classes are: {list(ModelRegistry.get_registry().keys())}"
        )

    return cls(config=config.params)


@hydra.main(
    config_path="../experiments", version_base="1.3"
)  # Point to the top-level config.yaml
def inference(config: DictConfig) -> None:
    config: TrainingConfig = TrainingConfig.model_validate(config)

    rprint(config)

    pl.seed_everything(config.seed, workers=True)
    stems = config.model.params.stems
    print(stems)
    # return

    model = _build_model(config=config.model)

    inference_handler = InferenceHandler(config=config.inference)

    system = SourceSeparationSystem(
        model=model,
        loss_handler=None,
        metric_handler=None,
        inference_handler=inference_handler,
        optimizer_config=None,
    )

    ckpt_path = config.ckpt_path
    input_path = config.input_path
    # output_path = config.output_path
    input_base = os.path.basename(input_path).rsplit(".", 1)[0]
    input_dir = os.path.dirname(input_path)
    stem_abbrev = "".join([s[0] for s in stems])
    output_path = os.path.join(input_dir, f"{input_base}_{stem_abbrev}_out")
    print(f"Output path: {output_path}")

    # trainer.predict(system, datamodule=None, ckpt_path=ckpt_path)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt['state_dict']
    state_dict = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.')}
    state_dict = {k.replace('pre_tf_seqband.', 'pre_tf_model.seqband.'): v for k, v in state_dict.items()}
    system.model.load_state_dict(state_dict, strict=False)
        

    system = system.to("cuda")
    system.eval()

    mixture = AudioDecoder(input_path,
                           sample_rate=config.inference.fs,
                           ).get_all_samples()

    batch = SourceSeparationBatch(mixture=MultiDomainSignal({'audio': mixture.data[None, ...]}),
                                  sources={},
                                  estimates={},
                                  n_samples=torch.tensor([mixture.data.shape[-1]]),
                                  )
    with torch.no_grad():
        outputs = system.predict_step(batch.model_dump())

    os.makedirs(output_path, exist_ok=True)

    for stem, est_dict in outputs.estimates.items():
        est_signal = est_dict['audio'][0].cpu()
        output_path_stem = os.path.join(output_path, f"{stem}.flac")
        AudioEncoder(
            samples=est_signal,
            sample_rate=config.inference.fs,
        ).to_file(output_path_stem)
        print(f"Written to {output_path_stem}")

if __name__ == "__main__":
    inference()
