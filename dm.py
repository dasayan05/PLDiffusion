import os

from torch.utils.data import DataLoader
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)

from lightning import (
    LightningDataModule
)

from datasets import (
    load_from_disk,
    load_dataset,
    DatasetDict,
    Dataset
)


def load_hf_dataset(path_or_reponame: str, **kwargs):
    if os.path.exists(path_or_reponame):
        return load_from_disk(path_or_reponame)
    else:
        return load_dataset(
            path_or_reponame,
            cache_dir=kwargs.get('cache_dir', None)
        )


class ImageDatasets(LightningDataModule):

    def __init__(self,
                 data_dir: str,
                 batch_size: int = 64,
                 image_resolution: int = 32,
                 HF_DATASET_IMAGE_KEY: str = 'img'
                 ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.hp = self.hparams

        # Preprocessing the datasets and DataLoaders creation.
        self.augmentations = Compose(
            [
                Resize(self.hp.image_resolution,
                       interpolation=InterpolationMode.BILINEAR),
                CenterCrop(self.hp.image_resolution),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize([0.5], [0.5]),
            ]
        )

    def setup(self, stage: str) -> None:
        dataset = load_hf_dataset(self.hp.data_dir)
        dataset.set_transform(
            lambda sample: ImageDatasets._transforms(self, sample)
        )

        if isinstance(dataset, DatasetDict):
            self.train_dataset, self.valid_dataset = dataset['train'], dataset['test']
        elif isinstance(dataset, Dataset):
            split_datasets = dataset.train_test_split(0.1, 0.9, seed=42)
            self.train_dataset, self.valid_dataset = split_datasets['train'], split_datasets['test']

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.hp.batch_size,
                          shuffle=True,
                          num_workers=4,
                          pin_memory=True,
                          drop_last=False
                          )

    def val_dataloader(self):
        return DataLoader(self.valid_dataset,
                          batch_size=self.hp.batch_size,
                          shuffle=False,
                          num_workers=4,
                          pin_memory=True,
                          drop_last=False
                          )

    def _transforms(self, sample):
        images = [
            self.augmentations(image.convert("RGB"))
            for image in sample[self.hp.HF_DATASET_IMAGE_KEY]
        ]
        return {"images": images}
