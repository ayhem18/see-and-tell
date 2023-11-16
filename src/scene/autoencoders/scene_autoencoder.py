"""This script contains the Denoising autoencoder used to extract features from the Big Bang Theory dataset 
"""
import os
import sys
import lightning as L
import torch
import torch.nn.functional as F

from pathlib import Path
from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT

HOME = os.path.dirname(os.path.realpath(__file__))
DATA_FOLDER = os.path.join(HOME, 'data')
current = HOME

while 'src' not in os.listdir(current):
    current = Path(current).parent

sys.path.append(str(current))
sys.path.append(os.path.join(current, 'src'))

from torch import nn
from src.scene.classifier.pretrained_conv import load_model


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

    def __init__(self, convolutional_block, *args: Any, **kwargs: Any):
        # the first step is to load the resnet model pretrained on the place365 dataset
        super().__init__(*args, **kwargs)
        self.encoder = load_model(feature_extractor=False)
        self.conv_block = convolutional_block

        x_temp = torch.randn((3, 14, 14))
        y_temp = self.conv_block(x_temp)
        if y_temp.shape != (3, 14, 14):
            raise ValueError(f"The convolutional block is expected to keep the dimensions of the input."
                             f"Expected: {(3, 14, 14)}. Found: {y_temp.shape}")

        # at this point we are sure the model will not silently break.
        self.decoder = nn.Sequential(
            *[deconvolution_block(input_channels=512, output_channels=256, stride=2, kernel_size=9),

              deconvolution_block(input_channels=256, output_channels=128, stride=1, kernel_size=6),

              deconvolution_block(input_channels=128, output_channels=64, stride=1, kernel_size=7),

              deconvolution_block(input_channels=64, output_channels=32, stride=1, kernel_size=7),

              deconvolution_block(input_channels=32, output_channels=16, stride=2, kernel_size=9),

              deconvolution_block(input_channels=16, output_channels=3, stride=2, kernel_size=4),
              ])

    def training_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        # the batch is expected to include 2 vectors
        x, x_noise = batch
        # make sure both x, and x_noise are of the expected dimensions
        batch_size = x.size(dim=0)

        if tuple(x.shape) != ((batch_size, ) + self._input_shape):
            raise ValueError(f"The input is not of the expected shape: expected {self._input_shape}. "
                             f"Found: {x[0].shape}")

        x_r = self.decoder(self.conv_block(self.encoder(x_noise)))

        # the loss is the Mean Squared Error between the constructed image and the original image
        mse_loss = F.mse_loss(x_r, x)

        return mse_loss
    
    