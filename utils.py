import os.path as osp

from lightning.pytorch import LightningModule, Trainer, callbacks
from diffusers.pipelines import DiffusionPipeline

class PipelineCheckpoint(callbacks.ModelCheckpoint):

    def on_save_checkpoint(self, trainer: Trainer, pl_module: LightningModule, checkpoint) -> None:
        pipe_path = osp.join(
            osp.dirname(self.best_model_path),
            f'pipeline-{pl_module.current_epoch}'
        )
        pipe: DiffusionPipeline = pl_module.pipe
        pipe.save_pretrained(pipe_path)

        return super().on_save_checkpoint(trainer, pl_module, checkpoint)
