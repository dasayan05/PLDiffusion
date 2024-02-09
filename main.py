import os
import math
from itertools import chain

import hydra as hy
from omegaconf import DictConfig, OmegaConf
from contextlib import contextmanager, nullcontext

import torch as th
from torchvision import transforms, utils as tv_utils
from lightning.pytorch import callbacks, Trainer, LightningModule
from torchmetrics.image.fid import FrechetInceptionDistance

from utils import (
    PipelineCheckpoint,
    EMACallback,
    _fix_hydra_config_serialization,
    ConfigMixin
)
from dm import ImageDatasets

from diffusers import DDPMPipeline
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

# some global stuff necessary for the program
th.set_float32_matmul_precision('medium')
to_tensor = transforms.ToTensor()


class Diffusion(LightningModule):
    def __init__(self,
                 models_cfg: DictConfig,
                 training_cfg: DictConfig,
                 inference_cfg: DictConfig
                 ):
        super().__init__()

        self.training_cfg = training_cfg
        self.inference_cfg = inference_cfg

        self.model = hy.utils.instantiate(models_cfg.unet)
        self.train_scheduler = \
            hy.utils.instantiate(self.training_cfg.scheduler)
        self.infer_scheduler = \
            hy.utils.instantiate(self.inference_cfg.scheduler)

        self.fid = FrechetInceptionDistance(
            normalize=True, reset_real_features=False)

        self.save_hyperparameters()

    def on_fit_start(self) -> None:
        for child in chain(self.children(), vars(self).values()):
            if isinstance(child, ConfigMixin):
                _fix_hydra_config_serialization(child)

    def record_real_data_for_FID(self, batch):
        if self.current_epoch == 0:
            self.fid.update(batch, real=True)

    def training_step(self, batch, batch_idx):
        clean_images = batch['images']

        self.record_real_data_for_FID((clean_images + 1) / 2.)

        noise = th.randn_like(clean_images)
        timesteps = th.randint(
            low=0,
            high=self.train_scheduler.config.num_train_timesteps,
            size=(clean_images.size(0), ), device=self.device
        ).long()
        noisy_images = self.train_scheduler.add_noise(
            clean_images, noise, timesteps)

        # Predict the noise residual
        model_output = self.model(noisy_images, timesteps).sample
        loss = th.nn.functional.mse_loss(model_output, noise)

        log_key = f'{"train" if self.training else "val"}/simple_loss'
        self.log_dict({log_key: loss},
                      prog_bar=True, sync_dist=True,
                      on_step=self.training,
                      on_epoch=not self.training)

        return loss

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

    @contextmanager
    def maybe_ema(self):
        ema = self.ema  # The EMACallback() ensures this
        ctx = nullcontext if ema is None else ema.average_parameters
        yield ctx

    def sample(self, **kwargs: dict):
        kwargs.pop('output_type', None)
        kwargs.pop('return_dict', False)

        pipe = self.pipeline()

        with self.maybe_ema():
            images, = pipe(
                **kwargs,
                output_type="pil",
                return_dict=False
            )
        return images

    def pipeline(self) -> DiffusionPipeline:
        pipe = DDPMPipeline(self.model, self.infer_scheduler).to(
            device=self.device, dtype=self.dtype)  # .to() isn't necessary
        pipe.set_progress_bar_config(disable=True)
        return pipe

    def save_pretrained(self, path: str, push_to_hub: bool = False):
        pipe = self.pipeline()
        pipe.save_pretrained(path, safe_serialization=True,
                             push_to_hub=push_to_hub)

    def on_validation_epoch_end(self) -> None:
        batch_size = self.inference_cfg.pipeline_kwargs.get(
            'batch_size', self.training_cfg.batch_size * 2)

        n_per_rank = math.ceil(
            self.inference_cfg.num_samples / self.trainer.world_size)
        n_batches_per_rank = math.ceil(
            n_per_rank / batch_size)

        # TODO: This may end up accummulating a little more than given 'n_samples'
        for _ in range(n_batches_per_rank):
            pil_images = self.sample(
                **self.inference_cfg.pipeline_kwargs
            )
            images = th.stack([to_tensor(pil_image)
                              for pil_image in pil_images], 0)
            self.fid.update(images.to(dtype=self.dtype, device=self.device),
                            real=False)

        fid = self.fid.compute()
        self.log('FID', fid, prog_bar=True, on_epoch=True, sync_dist=True)
        self.fid.reset()

        if self.global_rank == 0:
            image_grid = tv_utils.make_grid(images,
                                            nrow=math.ceil(batch_size ** 0.5), padding=1)
            try:
                saving_dir = self.logger.experiment.dir  # for wandb
            except AttributeError:
                saving_dir = self.logger.experiment.log_dir  # for TB

            tv_utils.save_image(image_grid,
                                os.path.join(saving_dir, f'samples_epoch_{self.current_epoch}.png'))

    def configure_optimizers(self):
        optim = th.optim.AdamW(
            self.parameters(), lr=self.training_cfg.learning_rate)
        sched = th.optim.lr_scheduler.StepLR(optim, 1, gamma=0.99)
        return {
            'optimizer': optim,
            'lr_scheduler': {'scheduler': sched, 'interval': 'epoch', 'frequency': 1}
        }


@hy.main(version_base=None, config_path='./configs')
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)  # resolve all string interpolation

    system = Diffusion(cfg.models, cfg.training, cfg.inference)
    datamodule = ImageDatasets(cfg.data)

    trainer = Trainer(
        callbacks=[
            callbacks.LearningRateMonitor(
                'epoch', log_momentum=True, log_weight_decay=True),
            PipelineCheckpoint(mode='min', monitor='FID'),
            EMACallback(decay=cfg.training.ema_decay),
            callbacks.RichProgressBar()
        ],
        logger=hy.utils.instantiate(cfg.logger, _recursive_=True),
        **cfg.pl_trainer
    )
    trainer.fit(system, datamodule=datamodule,
                ckpt_path=cfg.resume_from_checkpoint
                )


if __name__ == '__main__':
    main()
