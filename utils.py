import os
from typing import Optional, Any
from dataclasses import dataclass
from lightning.pytorch import LightningModule, Trainer, callbacks

from diffusers.schedulers.scheduling_utils import SchedulerMixin


@dataclass
class TrainingOptions:
    scheduler: Optional[SchedulerMixin]
    ema_decay: float = 0.9999
    batch_size: int = 64
    learning_rate: float = 1.e-4


@dataclass
class InferenceOptions:
    scheduler: Optional[SchedulerMixin]
    pipeline_kwargs: Any
    num_samples: int = 1024


class PipelineCheckpoint(callbacks.ModelCheckpoint):

    def on_save_checkpoint(self, trainer: Trainer, pl_module: LightningModule, checkpoint) -> None:
        # only ema parameters (if any) saved in pipeline
        with pl_module.maybe_ema():
            pipe_path = os.path.join(
                os.path.dirname(self.best_model_path),
                f'pipeline-{pl_module.current_epoch}'
            )
            pl_module.save_pretrained(pipe_path)

        return super().on_save_checkpoint(trainer, pl_module, checkpoint)
