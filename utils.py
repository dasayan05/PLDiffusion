import os.path as osp

from torch_ema import ExponentialMovingAverage as EMA
from lightning.pytorch import LightningModule, Trainer, callbacks
from diffusers.configuration_utils import ConfigMixin

from omegaconf.listconfig import ListConfig
from omegaconf.dictconfig import DictConfig
from diffusers.configuration_utils import FrozenDict


class PipelineCheckpoint(callbacks.ModelCheckpoint):

    def on_save_checkpoint(self, trainer: Trainer, pl_module: LightningModule, checkpoint) -> None:
        # only ema parameters (if any) saved in pipeline
        with pl_module.maybe_ema():
            pipe_path = osp.join(
                osp.dirname(self.best_model_path),
                f'pipeline-{pl_module.current_epoch}'
            )
            pl_module.save_pretrained(pipe_path)

        return super().on_save_checkpoint(trainer, pl_module, checkpoint)


class EMACallback(callbacks.Callback):

    def __init__(self, decay: float = 0.9999) -> None:
        super().__init__()

        # EMA decay hyper parameter
        self.decay = decay

    @property
    def ema_wanted(self):
        return (self.decay != -1)

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if self.ema_wanted:
            pl_module.ema = EMA(pl_module.parameters(), decay=self.decay)
            pl_module.ema.to(pl_module.device)
        else:
            pl_module.ema = None

    def on_save_checkpoint(self, trainer: Trainer, pl_module: LightningModule, checkpoint: dict) -> None:
        if self.ema_wanted:
            checkpoint['ema'] = pl_module.ema.state_dict()

        return super().on_save_checkpoint(trainer, pl_module, checkpoint)

    def on_load_checkpoint(self, trainer: Trainer, pl_module: LightningModule, checkpoint: dict) -> None:
        if self.ema_wanted:
            pl_module.ema.load_state_dict(checkpoint['ema'])

        return super().on_load_checkpoint(trainer, pl_module, checkpoint)

    def on_before_zero_grad(self, trainer: Trainer, pl_module: LightningModule, optimizer) -> None:
        if self.ema_wanted:
            pl_module.ema.update(pl_module.parameters())

        return super().on_before_zero_grad(trainer, pl_module, optimizer)


def _fix_hydra_config_serialization(conf_mixin: ConfigMixin):
    # This is a hack due to incompatibility between hydra and diffusers
    new_internal_dict = {}
    for k, v in conf_mixin._internal_dict.items():
        if isinstance(v, ListConfig):
            new_internal_dict[k] = list(v)
        elif isinstance(v, DictConfig):
            new_internal_dict[k] = dict(v)
        else:
            new_internal_dict[k] = v
    conf_mixin._internal_dict = FrozenDict(new_internal_dict)
