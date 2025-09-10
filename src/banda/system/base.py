import gc
import os
from typing import Literal
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch

from banda.data.item import SourceSeparationBatch
from banda.inference.handler import InferenceHandler
from banda.losses.base import LossDict
from banda.losses.handler import LossHandler
from banda.metrics.handler import MetricHandler
from banda.models.base import BaseRegisteredModel
from pytorch_lightning.loggers import WandbLogger
from tqdm import tqdm

import torchaudio as ta
import structlog
import pandas as pd

logger = structlog.get_logger(__name__)


class SourceSeparationSystem(pl.LightningModule):
    def __init__(
        self,
        *,
        model: BaseRegisteredModel,
        loss_handler: LossHandler,
        metric_handler: MetricHandler,
        inference_handler: InferenceHandler | None = None,
        optimizer_config: DictConfig | None = None,
    ):
        super().__init__()
        self.model = model
        self.loss_handler = loss_handler
        self.metric_handler = metric_handler
        self.inference_handler = inference_handler
        self.optimizer_config = optimizer_config

    def on_train_epoch_start(self):
        self.metric_handler.reset()

    def training_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        batch, total_loss = self.common_step(batch, mode="train")

        self.metric_handler.reset()
        self.metric_handler.update(batch)
        metric_dict = self.metric_handler.compute()
        self._log_metric(metric_dict=metric_dict, mode="train")

        return total_loss

    def on_validation_epoch_start(self):
        self.metric_handler.reset()

    def validation_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        batch, total_loss = self.common_step(batch, mode="val")
        self.metric_handler.update(batch)
        return total_loss

    def on_validation_epoch_end(self):
        metric_dict = self.metric_handler.compute()
        self._log_metric(metric_dict=metric_dict, mode="val")

    def on_test_epoch_start(self):
        self.test_metrics = []

    def test_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        reconstructed_batch = self.inference_step(batch, batch_idx, dataloader_idx)

        self.metric_handler.reset()
        self.metric_handler.update(reconstructed_batch)
        metric_dict = self.metric_handler.compute()
        print(metric_dict)

        logger.info("Test metric", metric_dict=metric_dict)

        self.test_metrics.append(
            {
                "batch_idx": batch_idx,
                "full_path": reconstructed_batch.full_path[0],
                **metric_dict,
            }
        )
        
        self.log_dict(
            metric_dict, prog_bar=True, logger=False, on_step=True, on_epoch=False
        )

    def on_test_epoch_end(self):
        df = pd.DataFrame(self.test_metrics)
        df = df.set_index("batch_idx")

        print(df)

        logger: WandbLogger = self.logger
        logger.log_table(key="test/metrics", dataframe=df)

    def predict_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        return self.inference_step(batch, batch_idx, dataloader_idx)

    def _try_inference(
        self, chunked_audio: torch.Tensor, inference_batch_size: int = None
    ):
        chunked_outputs = []

        for chunked_batch in self.inference_handler.to_inference_batch(
            chunked_audio=chunked_audio, inference_batch_size=inference_batch_size
        ):
            chunked_batch.mixture["audio"] = chunked_batch.mixture["audio"].to(
                self.device
            )
            chunked_output = self.forward(chunked_batch)
            chunked_outputs.append(
                SourceSeparationBatch(
                    mixture=None,
                    sources={},
                    estimates={
                        source: {
                            "audio": chunked_output.estimates[source]["audio"].cpu()
                        }
                        for source in chunked_output.estimates
                    },
                )
            )

            chunked_output.estimates = {}
            chunked_output.mixture = {}
            torch.cuda.empty_cache()

        return chunked_outputs

    def inference_step(
        self, batch: dict, batch_idx: int, dataloader_idx: int
    ) -> SourceSeparationBatch:
        batch = SourceSeparationBatch.model_validate(batch)

        assert self.inference_handler is not None, "Inference handler is not set."

        assert str(batch.mixture["audio"].device) == "cpu", (
            f"mixture should be on cpu, but got {batch.mixture['audio'].device}"
        )

        chunked_audio, padded_samples = self.inference_handler.chunk_batch(batch)

        chunked_outputs = None
        inference_batch_size = self.inference_handler.inference_batch_size

        while chunked_outputs is None:
            try:
                chunked_outputs = self._try_inference(
                    chunked_audio, inference_batch_size=inference_batch_size
                )
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()

                if inference_batch_size >= 48:
                    inference_batch_size -= 8
                elif inference_batch_size > 12:
                    inference_batch_size -= 4
                elif inference_batch_size > 4:
                    inference_batch_size -= 2
                else:
                    inference_batch_size -= 1

                if inference_batch_size == 0:
                    raise torch.cuda.OutOfMemoryError()

                logger.warning(
                    f"The current batch size does not fit in memory. Trying {inference_batch_size}"
                )

        self.inference_handler.inference_batch_size = inference_batch_size

        reconstructed_batch = self.inference_handler.chunked_reconstruct(
            chunked_outputs, original_batch=batch, padded_samples=padded_samples
        )

        return reconstructed_batch

    def common_step(
        self, batch: dict, *, mode: Literal["train", "val", "test", "predict"]
    ):
        batch = self.forward(batch)
        loss_dict = self.loss_handler(batch)
        self._log_loss(loss_dict=loss_dict, mode=mode)

        return batch, loss_dict.total_loss

    def forward(self, batch: dict | SourceSeparationBatch) -> SourceSeparationBatch:
        batch = SourceSeparationBatch.model_validate(batch)
        return self.model(batch)

    def configure_optimizers(self):
        # Define the optimizer and learning rate scheduler

        optimizer_config = self.optimizer_config.optimizer
        cls = getattr(torch.optim, optimizer_config.cls)
        params = optimizer_config.params
        optimizer = cls(self.parameters(), **params)

        scheduler_config = self.optimizer_config.scheduler
        scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_config.cls)
        scheduler_kwargs = scheduler_config.params
        scheduler = scheduler_cls(optimizer, **scheduler_kwargs)

        return [optimizer], [scheduler]

    def _log_loss(
        self, loss_dict: LossDict, mode: Literal["train", "val", "test", "predict"]
    ):
        prefix = f"{mode}"
        on_step = mode == "train"
        on_epoch = mode != "train"

        prog_bar_loss = {
            k: v for k, v in loss_dict.loss_contrib.items() if "/" not in k
        }

        self.log_dict(prog_bar_loss, on_step=True, prog_bar=True, logger=False)

        loss_contrib_dict = {
            f"{prefix}/loss/{key}": value
            for key, value in loss_dict.loss_contrib.items()
        }

        self.log_dict(
            loss_contrib_dict, on_step=on_step, on_epoch=on_epoch, prog_bar=False
        )

        self.log(
            name="loss",
            value=loss_dict.total_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=False,
        )
        self.log(
            name=f"{mode}/loss",
            value=loss_dict.total_loss,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=False,
            logger=True,
        )

    def _log_metric(
        self, metric_dict: dict, mode: Literal["train", "val", "test", "predict"]
    ):
        prefix = f"{mode}"
        on_step = mode == "train"
        on_epoch = mode != "train"

        if mode == "train":
            self.log_dict(
                metric_dict, on_step=True, on_epoch=False, prog_bar=True, logger=False
            )

        metric_dict = {
            f"{prefix}/metric/{key}": value for key, value in metric_dict.items()
        }

        self.log_dict(metric_dict, on_step=on_step, on_epoch=on_epoch, prog_bar=False)

    def load_state_dict(self, state_dict, strict: bool = True):
        state_dict = self._handle_split_tf_model_change(state_dict)

        super().load_state_dict(state_dict, strict=strict)

    def _handle_split_tf_model_change(self, state_dict):
        # handle the cases where the model were trained with the unified tf_model.
        # now the same model is split into pre_tf_model and post_tf_model
        # if use_split_tf_model is False, pre_tf_model is just tf_model and post_tf_model is identity

        current_state_dict_keys = set(self.state_dict().keys())

        # check if model.tf_model is in state_dict

        has_old_tf_model = any(
            key.startswith("model.tf_model") for key in state_dict.keys()
        )

        if has_old_tf_model:
            logger.info(
                "The checkpoint was trained with the unified tf_model. Splitting the weights into pre_tf_model and post_tf_model."
            )

            # loop through the current key of the pre_tf_model state dict
            # if there are more keys in the old tf_model, then the rest has to be in post_tf_model

            for k in sorted(current_state_dict_keys):
                if k.startswith("model.post_tf_model"):
                    print(k)

            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("model.tf_model"):
                    new_key = key.replace("model.tf_model", "model.pre_tf_model")

                    if new_key not in current_state_dict_keys:
                        # FIXME: deal with this later
                        new_key = key.replace("model.tf_model", "model.post_tf_model")
                        # update the module numbering
                        seq_band_idx = int(
                            key.replace("model.tf_model.seqband.", "").split(".")[0]
                        )
                        # print(key)
                        print(
                            f"Replacing {seq_band_idx} with {seq_band_idx - self.model.pre_tf_model.config.n_modules}"
                        )
                        new_key = new_key.replace(
                            f"seqband.{seq_band_idx}.",
                            f"seqband.{seq_band_idx - 2 * self.model.pre_tf_model.config.n_modules}.",
                        )
                        assert new_key in current_state_dict_keys, (
                            f"Key {new_key} not found in current state dict."
                        )

                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value

            return new_state_dict

        return state_dict
