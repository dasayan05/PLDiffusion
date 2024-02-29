import torch as th
from lightning.pytorch import cli

from dm import ImageDatasets
from core.modules import Diffusion

# some global stuff necessary for the program
th.set_float32_matmul_precision('medium')

if __name__ == '__main__':
    cli.LightningCLI(Diffusion,
                     ImageDatasets,
                     subclass_mode_model=True,
                     parser_kwargs={
                         'parser_mode': 'omegaconf'
                     }
                     )
