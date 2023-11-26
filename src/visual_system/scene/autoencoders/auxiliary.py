"""

"""


import os
import sys
import lightning as L
import torch
import torch.nn.functional as F
import torchvision.transforms as tr
import random
import pandas as pd

from pathlib import Path

from typing import Any, Union, Iterable
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything
from torchvision import transforms as trn
from torch import nn
from torch.utils.data import Dataset
from PIL import Image

seed_everything(69, workers=True)


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
current = SCRIPT_DIR

while 'src' not in os.listdir(current):
    current = Path(current).parent

PARENT_DIR = str(current)
sys.path.append(str(current))
sys.path.append(os.path.join(current, 'src'))

import src.utilities.directories_and_files as dirf
import src.utilities.pytorch_utilities as pu


## data augmentation
def augment(x) -> torch.Tensor:
    p = random.random()
    
    # cropping the image
    if p < 0.33:
        return trn.RandomResizedCrop(size=(180, 180), antialias=True)(x)
    
    # flipping the image
    elif p < 0.66:
        return trn.RandomVerticalFlip(p=0.5)(x)
    
    # rotation
    # choose the angle
    return trn.RandomRotation(degrees=15)(x)


def returnTF(add_augment: bool = True):
    # load the image transformer
    tf = trn.Compose([
        lambda x: augment(x), 
        trn.Resize((224, 224)),
        trn.ToTensor(),        
    ]) if add_augment else trn.Compose([trn.Resize((224, 224)), trn.ToTensor()])
    
    return tf


def deconvolution_block(input_channels,
                        output_channels,
                        kernel_size=4,
                        stride=2,
                        padding: int = 0,
                        final_layer: bool = False):
    layers = [
        nn.ConvTranspose2d(in_channels=input_channels,
                           out_channels=output_channels,
                           kernel_size=kernel_size,
                           stride=stride, 
                           padding=padding),
    ]
    
    if final_layer:
        layers.extend([nn.Sigmoid()])
    else:
        layers.extend([nn.BatchNorm2d(output_channels), nn.LeakyReLU()])

    return nn.Sequential(*layers)



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


def add_noise(x: torch.Tensor, noise_factor: float = 0.2) -> torch.Tensor:
    # the first step is to add Guassian noise
    x_noise = x + noise_factor * torch.randn(*x.shape).to(pu.get_module_device(x))
    # make sure to clip the noise to the range [0, 1] # since the original image has pixel values in that range
    # return x_noise
    return torch.clip(x_noise, min=0, max=1)
