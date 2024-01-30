import os, math
import wandb
import hydra as hy
from omegaconf import DictConfig, OmegaConf

import torch as th
from torchvision import transforms, utils as tv_utils
from lightning.pytorch import loggers, callbacks, Trainer, LightningModule
from torchmetrics.image.fid import FrechetInceptionDistance
from torch_ema import ExponentialMovingAverage as EMA

from utils import PipelineCheckpoint
from dm import ImageDatasets

from diffusers import (
    UNet2DModel,
    DDPMScheduler,
    DDPMPipeline
)

# some global stuff necessary for the program
th.set_float32_matmul_precision('medium')
to_tensor = transforms.ToTensor()


class Diffusion(LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

        self.model = UNet2DModel(**OmegaConf.to_container(self.config.model))
        self.scheduler = DDPMScheduler(**OmegaConf.to_container(self.config.training.scheduler))
        self.ema = EMA(*self.model.parameters(), decay = self.config.training.ema_decay)

        self.fid = FrechetInceptionDistance(normalize=True, reset_real_features=False)

        self.save_hyperparameters(self.config)
        
    def on_fit_start(self) -> None:
        return super().on_fit_start()
    
    def record_real_data_for_FID(self, batch):
        if self.current_epoch == 0:
            self.fid.update(batch, real = True)

    def training_step(self, batch, batch_idx):
        clean_images = batch['images']

        self.record_real_data_for_FID((clean_images + 1) / 2.)

        noise = th.randn_like(clean_images)
        timesteps = th.randint(
            low = 0,
            high = self.scheduler.config.num_train_timesteps,
            size = (clean_images.size(0), ), device=self.device
        ).long()
        noisy_images = self.scheduler.add_noise(clean_images, noise, timesteps)

        # Predict the noise residual
        model_output = self.model(noisy_images, timesteps).sample
        loss = th.nn.functional.mse_loss(model_output, noise)

        if self.training:
            self.log_dict({
                'train/simple_loss': loss
            }, prog_bar=True, sync_dist=True, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        self.log_dict({
            'val/simple_loss': loss
        }, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        return loss
    
    def sample(self, n = 32):
        images, = self.pipe(
            batch_size=n,
            num_inference_steps=self.config.inference.num_inference_steps,
            output_type="numpy",
            return_dict=False
        )
        return th.from_numpy(images).permute(0, 3, 1, 2).to(self.device)
    
    def create_hf_pipeline(self):
        self.pipe = DDPMPipeline(self.model, self.scheduler).to(device=self.device, dtype=self.dtype)
        self.pipe.set_progress_bar_config(disable=True)
    
    def on_validation_epoch_end(self) -> None:
        self.create_hf_pipeline()

        n_per_rank = math.ceil(self.config.inference.num_samples / self.trainer.world_size)
        n_batches_per_rank = math.ceil(n_per_rank / self.config.inference.batch_size)
        # TODO: This may end up accummulating a little more than given 'n_samples'
        for _ in range(n_batches_per_rank):
            images = self.sample(n = self.config.inference.batch_size)
            self.fid.update(images, real = False)
        
        fid = self.fid.compute()
        self.log('FID', fid, prog_bar=True, on_epoch=True, sync_dist=True)
        self.fid.reset()
        
        if self.global_rank == 0:
            image_grid = tv_utils.make_grid(images, nrow=math.ceil(self.config.inference.batch_size ** 0.5), padding=1)
            tv_utils.save_image(image_grid,
                    os.path.join(self.logger.experiment.dir, f'samples_epoch_{self.current_epoch}.png'))
    
    def configure_optimizers(self):
        optim = th.optim.AdamW(self.parameters(), lr=self.config.training.learning_rate)
        sched = th.optim.lr_scheduler.StepLR(optim, 1, gamma=0.99)
        return {
            'optimizer': optim,
            'lr_scheduler': { 'scheduler': sched, 'interval': 'epoch', 'frequency': 1 }
        }
    

@hy.main(version_base=None, config_path='./config')
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg) # resolve all string interpolation

    system = Diffusion(cfg)
    datamodule = ImageDatasets(cfg.data)

    trainer = Trainer(
        accelerator = 'gpu',
        num_nodes = 1,
        benchmark = True,
        strategy = 'ddp',
        num_sanity_val_steps = 0,
        callbacks = [
            callbacks.LearningRateMonitor('epoch', log_momentum=True, log_weight_decay=True),
            PipelineCheckpoint()
        ],
        logger = loggers.WandbLogger(
            settings = wandb.Settings(_disable_stats = True, _disable_meta = True),
            **OmegaConf.to_container(cfg.logging)
        ),
        **OmegaConf.to_container(cfg.pl_trainer)
    )
    trainer.fit(system, datamodule=datamodule,
        ckpt_path = cfg.resume_from_checkpoint
    )

if __name__ == '__main__':
    main()