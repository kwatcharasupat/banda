import os
from pytorch_lightning.callbacks import ModelCheckpoint


class ModelCheckpointWithAutoRestart(ModelCheckpoint):
    def __init__(self, config_name: str, **kwargs):
        super().__init__(**kwargs)
        self.config_name = config_name

    def on_exception(self, trainer, pl_module, exception):
        super().on_exception(trainer, pl_module, exception)

        last_checkpoint = self._last_checkpoint_saved
        command = f"python scripts/slurm.py make {self.config_name} --ckpt_path={last_checkpoint} --submit"
        os.system(command)
        print(f"Submitted new job with command: {command}")
