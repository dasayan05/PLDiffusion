import os
import math
import contextlib
from tqdm import tqdm

import torch as th
from torchvision import transforms, utils as tv_utils
from torch_ema import ExponentialMovingAverage as EMA

from lightning.pytorch import LightningModule, cli, callbacks as cb
from diffusers import DDPMPipeline
from diffusers.models.modeling_utils import ModelMixin
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from dm import ImageDatasets
from metrics import Metrics
from utils import (
    PipelineCheckpoint,
    TrainingOptions,
    InferenceOptions
)

# some global stuff necessary for the program
th.set_float32_matmul_precision('medium')
to_tensor = transforms.ToTensor()


class Diffusion(LightningModule):
    def __init__(self,
                 network: ModelMixin,
                 training: TrainingOptions,
                 inference: InferenceOptions,
                 ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.hp = self.hparams  # short hand

        self.model = self.hp.network
        self.train_scheduler = self.hp.training.scheduler
        self.infer_scheduler = self.hp.inference.scheduler or self.hp.training.scheduler

        # For making the UNet aware of the total training timesteps
        # (TODO): clean this up properly; can we do it without this
        self.model.diffusion_steps = self.train_scheduler.config.num_train_timesteps

        self.ema = \
            EMA(self.model.parameters(), decay=self.hp.training.ema_decay) \
            if self.ema_wanted else None

        self.metrics = Metrics(vFID=True)

    @property
    def ema_wanted(self):
        return self.hp.training.ema_decay != -1

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        if self.ema_wanted:
            checkpoint['ema'] = self.ema.state_dict()
        return super().on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        if self.ema_wanted:
            self.ema.load_state_dict(checkpoint['ema'])
        return super().on_load_checkpoint(checkpoint)

    def on_before_zero_grad(self, optimizer) -> None:
        if self.ema_wanted:
            self.ema.update(self.model.parameters())
        return super().on_before_zero_grad(optimizer)

    def to(self, *args, **kwargs):
        if self.ema_wanted:
            self.ema.to(*args, **kwargs)
        self.metrics.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        clean_images = batch['images']

        self.metrics.record_real_data_for_FID((clean_images + 1) / 2.)

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
        clean_images = batch['images']
        self.metrics.record_real_data_for_vFID((clean_images + 1.) / 2.)

        return self.training_step(batch, batch_idx)

    @contextlib.contextmanager
    def maybe_ema(self):
        ema = self.ema  # The EMACallback() ensures this
        ctx = contextlib.nullcontext if ema is None else ema.average_parameters
        yield ctx

    def sample(self, **kwargs: dict):
        kwargs.pop('output_type', None)
        kwargs.pop('return_dict', False)

        pipe = self.pipeline()
        pipe.set_progress_bar_config(disable=True)

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
        return pipe

    def save_pretrained(self, path: str, push_to_hub: bool = False):
        pipe = self.pipeline()
        pipe.save_pretrained(path, safe_serialization=True,
                             push_to_hub=push_to_hub)

    def on_validation_epoch_end(self) -> None:
        batch_size = self.hp.inference.pipeline_kwargs.get('batch_size', 128)

        n_per_rank = math.ceil(
            self.hp.inference.num_samples / self.trainer.world_size)
        n_batches_per_rank = math.ceil(n_per_rank / batch_size)

        sampling_pbar = tqdm(total=n_batches_per_rank, desc='Sampling',
                             disable=self.global_rank != 0)

        # TODO: This may end up accummulating a little more than given 'n_samples'
        with self.metrics.metrics() as m:
            for _ in range(n_batches_per_rank):
                pil_images = self.sample(
                    **self.hp.inference.pipeline_kwargs
                )
                m.record_fake_data(pil_images)
                sampling_pbar.update(1)

            met = {'FID': m.FID.item(), 'vFID': m.vFID.item()}

            self.log_dict(met, prog_bar=True, on_epoch=True, sync_dist=True)

        if self.global_rank == 0:
            images = th.stack([to_tensor(pil_image)
                              for pil_image in pil_images], 0)
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
            self.parameters(), lr=self.hp.training.learning_rate)
        sched = th.optim.lr_scheduler.StepLR(optim, 1, gamma=0.999)
        return {
            'optimizer': optim,
            'lr_scheduler': {'scheduler': sched, 'interval': 'epoch', 'frequency': 1}
        }


if __name__ == '__main__':
    lr_monitor = cb.LearningRateMonitor('epoch',
                                        log_momentum=True,
                                        log_weight_decay=True
                                        )
    model_checkpointing = PipelineCheckpoint(mode='min',
                                             monitor='FID'
                                             )

    cli.LightningCLI(Diffusion, ImageDatasets,
                     parser_kwargs={
                         'parser_mode': 'omegaconf'
                     },
                     trainer_defaults={
                         'callbacks': [
                             lr_monitor,
                             model_checkpointing,
                             cb.TQDMProgressBar()

                         ],
                     }
                     )
