from typing import Union
from PIL.Image import Image
from contextlib import contextmanager

import torch as th
from torchvision import transforms

from torchmetrics.image.fid import (
    NoTrainInceptionV3,
    FrechetInceptionDistance as FID
)

to_tensor = transforms.ToTensor()


class Metrics(th.nn.Module):

    def __init__(self,
                 feature: int = 2048,
                 FID: bool = True,
                 vFID: bool = False) -> None:
        super().__init__()
        self.is_FID = FID
        self.is_vFID = vFID

        self.inceptionv3 = NoTrainInceptionV3("inception-v3-compat",
                                              features_list=[str(feature)])

        if self.is_FID:
            self._fid = FID(
                feature=self.inceptionv3,
                reset_real_features=False,
                normalize=True
            )
            self._fid.persistent(True)
            self._fid.requires_grad_(False)

        if self.is_vFID:
            self._vfid = FID(
                feature=self.inceptionv3,
                reset_real_features=False,
                normalize=True
            )
            self._vfid.persistent(True)
            self._vfid.requires_grad_(False)

    def to(self, device: th.device = None, dtype: th.dtype = None):
        self.dtype = dtype
        self.device = device

    @contextmanager
    def metrics(self):
        if self.is_FID:
            self._fid.reset()
        if self.is_vFID:
            self._vfid.reset()

        yield self

        if self.is_FID:
            self._fid.reset()
        if self.is_vFID:
            self._vfid.reset()

    @property
    def FID(self):
        if self.is_FID:
            return self._fid.compute()
        else:
            # meaning it's disabled
            return th.tensor(-1.)

    @property
    def vFID(self):
        if self.is_vFID:
            return self._vfid.compute()
        else:
            # meaning it's disabled
            return th.tensor(-1.)

    @property
    def n_real_FID(self):
        if self.is_FID:
            return self._fid.metric_state['real_features_num_samples']
        else:
            return float('inf')

    @property
    def n_real_vFID(self):
        if self.is_vFID:
            return self._vfid.metric_state['real_features_num_samples']
        else:
            return float('inf')

    def record_data(self,
                    fid: FID,
                    batch: Union[list[Image], th.Tensor],
                    real: bool):
        # batch must be either list of PIL Images, ..
        # .. or, a Tensor of shape (BxCxHxW)
        if isinstance(batch, list):
            batch = th.stack([to_tensor(pil_image)
                              for pil_image in batch], 0)
            # .to() must be called before this
            batch = batch.to(dtype=self.dtype, device=self.device)
        fid.update(batch, real=real)

    def record_fake_data(self, batch):
        if self.is_FID:
            self.record_fake_data_for_FID(batch)
        if self.is_vFID:
            self.record_fake_data_for_vFID(batch)

    def record_fake_data_for_FID(self, batch):
        if self.is_FID:
            self.record_data(self._fid, batch, False)

    def record_real_data_for_FID(self, batch, upto: int = 50000):
        if self.n_real_FID < upto and self.is_FID:
            self.record_data(self._fid, batch, True)

    def record_fake_data_for_vFID(self, batch):
        if self.is_vFID:
            self.record_data(self._vfid, batch, False)

    def record_real_data_for_vFID(self, batch, upto: int = 10000):
        if self.is_vFID and self.n_real_vFID < upto:
            self.record_data(self._vfid, batch, True)
