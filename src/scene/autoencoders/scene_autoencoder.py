"""This script contains the Denoising autoencoder used to extract features from the Big Bang Theory dataset 
"""
import os
import sys
import lightning as L
import torch
import torch.nn.functional as F
import torchvision.transforms as tr
import random
import shutil
import wandb

random.seed(69)
torch.manual_seed(69)

from torch import optim
from pathlib import Path
from typing import Any, Union, Iterable
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything

seed_everything(69, workers=True)

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
current = SCRIPT_DIR

while 'src' not in os.listdir(current):
    current = Path(current).parent

PARENT_DIR = str(current)
sys.path.append(str(current))
sys.path.append(os.path.join(current, 'src'))

from torch import nn
from src.scene.classifier.pretrained_conv import load_model, returnTF
import src.utilities.directories_and_files as dirf
import src.utilities.pytorch_utilities as pu


def deconvolution_block(input_channels,
                        output_channels,
                        kernel_size=4,
                        stride=2,
                        padding: int = 1,
                        final_layer: bool = False):
    layers = [
        nn.ConvTranspose2d(in_channels=input_channels,
                           out_channels=output_channels,
                           kernel_size=kernel_size,
                           stride=stride),
    ]

    if final_layer:
        layers.append(nn.Tanh())
    else:
        layers.extend([nn.BatchNorm2d(output_channels), nn.ReLU()])

    return nn.Sequential(*layers)


class SceneDenoiseAE(L.LightningModule):
    _input_shape = (3, 224, 224)

    def __init__(self, convolutional_block: nn.Module, *args: Any, **kwargs: Any):
        # the first step is to load the resnet model pretrained on the place365 dataset
        super().__init__(*args, **kwargs)
        self.encoder = load_model(feature_extractor=True)

        o = self.encoder(torch.randn(1, 3, 224, 224))

        if o.shape != (1, 512, 14, 14):
            raise ValueError(f"Please make sure the encoder is loaded correctly !!")

        self.conv_block = convolutional_block

        x_temp = torch.randn((1, 512, 14, 14))
        y_temp = self.conv_block(x_temp)
        if y_temp.shape != (1, 512, 14, 14):
            raise ValueError(f"The convolutional block is expected to keep the dimensions of the input."
                             f"Expected: {(3, 14, 14)}. Found: {y_temp.shape}")

        # at this point we are sure the model will not silently break.
        # the architecture is hardcoded for the moment.
        self.decoder = nn.Sequential(
            *[deconvolution_block(input_channels=512, output_channels=256, stride=2, kernel_size=9),

              deconvolution_block(input_channels=256, output_channels=128, stride=1, kernel_size=6),

              deconvolution_block(input_channels=128, output_channels=64, stride=1, kernel_size=7),

              deconvolution_block(input_channels=64, output_channels=32, stride=1, kernel_size=7),

              deconvolution_block(input_channels=32, output_channels=16, stride=2, kernel_size=9),

              deconvolution_block(input_channels=16, output_channels=3, stride=2, kernel_size=4),
              ])

        self.save_hyperparameters()

    def _forward_pass(self, batch):
        x = batch
        x_noise = add_noise(x, noise_factor=random.random() * 0.5)
        # make sure both x, and x_noise are of the expected dimensions
        batch_size = x.size(dim=0)

        if tuple(x.shape) != ((batch_size,) + self._input_shape):
            raise ValueError(f"The input is not of the expected shape: expected {self._input_shape}. "
                             f"Found: {x[0].shape}")
        x_r = self.decoder(self.conv_block(self.encoder(x_noise)))
        # the loss is the Mean Squared Error between the constructed image and the original image
        mse_loss = F.mse_loss(x_r, x)
        return mse_loss, x_noise, x_r

    def training_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        mse_loss, _, _ = self._forward_pass(batch)
        self.log(name='train_loss', value=mse_loss.cpu().item())
        return mse_loss

    def validation_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        mse_loss, _, _ = self._forward_pass(batch)
        self.log(name='val_loss', value=mse_loss.cpu().item())
        return mse_loss

    def configure_optimizers(self):
        # since the encoder is pretrained, we would like to avoid significantly modifying its weights/
        # on the other hand, the rest of the AE should have higher learning rates.
        parameters = [{"params": self.encoder.parameters(), "lr": 10 ** -5},
                      {"params": self.conv_block.parameters(), "lr": 10 ** -2},
                      {"params": self.decoder.parameters(), "lr": 10 ** -3}]

        # add a learning rate scheduler        
        optimizer = optim.Adam(parameters)

        return optimizer

    # this is the loading function used in the DatasetFolder pytorch implementation.


def _load_sample(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class GenerativeDS(Dataset):
    def __init__(self,
                 data_path: Union[str, Path],
                 transformation: tr = None,
                 image_extensions: Iterable[str] = None,
                 ) -> None:
        super().__init__()

        self.data_path = dirf.process_save_path(data_path,
                                                file_ok=False,
                                                condition=lambda _: dirf.all_images(_,
                                                                                    image_extensions=image_extensions))
        self.file_names = sorted(os.listdir(self.data_path))
        self.t = transformation

    def __getitem__(self, index) -> Any:
        sample_path = os.path.join(self.data_path, self.file_names[index])
        sample = _load_sample(sample_path) if self.t is None else self.t(_load_sample(sample_path))
        return sample

    def __len__(self) -> int:
        return len(self.file_names)


def add_noise(x: torch.Tensor, noise_factor: float = 0.3) -> torch.Tensor:
    # the first step is to add Guassian noise
    x_noise = x + noise_factor * torch.randn(*x.shape).to(pu.get_module_device(x))
    # make sure to clip the noise to the range [0, 1] # since the original image has pixel values in that range
    return torch.clip(x_noise, min=0, max=1)


def train_ae(train_dir: Union[str, Path],
             val_dir: Union[str, Path] = None,
             log_dir: Union[str, Path] = None,
             image_extensions: Iterable[str] = None,
             batch_size: int = 32,
             num_epochs: int = 10):
    # first process both directories
    train_dir = dirf.process_save_path(train_dir,
                                       file_ok=False,
                                       condition=lambda _: dirf.all_images(_, image_extensions=image_extensions),
                                       error_message="The directory is expected to have only image files."
                                       )
    # the output directory must be empty
    log_dir = os.path.join(SCRIPT_DIR, 'logs') if log_dir is None else log_dir
    # process the path
    log_dir = dirf.process_save_path(log_dir, file_ok=False)

    # define the dataset 
    model_transformation = returnTF()

    train_dataset = GenerativeDS(data_path=train_dir,
                                 transformation=model_transformation)

    train_dl = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          pin_memory=True)
    if val_dir is not None:
        val_dl = DataLoader(dataset=GenerativeDS(data_path=val_dir, transformation=model_transformation),
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True, )
    else:
        val_dl = None

    # the convolutional block
    conv_block = nn.Sequential(
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(5, 5), stride=1, padding='same'),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding='same'),
        nn.BatchNorm2d(num_features=512),
        nn.ReLU())

    model = SceneDenoiseAE(convolutional_block=conv_block)

    wandb_logger = WandbLogger(project='cntell_auto_encoder',
                               log_model="all", save_dir=log_dir)

    # define the trainer
    trainer = L.Trainer(logger=wandb_logger,
                        default_root_dir=log_dir,
                        max_epochs=num_epochs,
                        deterministic=True)

    # the val_dataloaders have 'None' values as default
    trainer.fit(model=model,
                train_dataloaders=train_dl,
                val_dataloaders=val_dl
                )


if __name__ == '__main__':
    wandb.login(key='36259fe078be47d3ffd8f3b2628a4d773c6e1ce7')

    all_data = os.path.join(PARENT_DIR, 'src', 'scene', 'augmented_data')

    train_dir = os.path.join(PARENT_DIR, 'src', 'scene', 'train_dir')
    val_dir = os.path.join(PARENT_DIR, 'src', 'scene', 'val_dir')

    # let's split the data into train and test splits
    train_data, val_data = train_test_split(os.listdir(all_data), test_size=0.1, random_state=69)

    if not os.path.isdir(train_dir):
        os.makedirs(train_dir)
        for f in train_data:
            shutil.copyfile(os.path.join(all_data, f), os.path.join(train_dir, f))

    if not os.path.isdir(val_dir):
        os.makedirs(val_dir)
        for f in val_data:
            shutil.copyfile(os.path.join(all_data, f), os.path.join(val_dir, f))

    logs = os.path.join(PARENT_DIR, 'src', 'scene', 'autoencoders', 'runs')
    os.makedirs(logs, exist_ok=True)

    train_ae(train_dir=train_dir,
             val_dir=val_dir,
             batch_size=4,
             log_dir=os.path.join(logs, f'exp_{len(os.listdir(logs)) + 1}'), num_epochs=100)
