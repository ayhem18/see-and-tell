# first let's build a simple linear block
import torch
import numpy as np
import random
import itertools

from torch import nn
from typing import Iterator, Tuple, Union, List

import os, sys
from pathlib import Path
from abc import ABC, abstractmethod

from torch.nn.modules.module import Module

HOME = os.getcwd()
DATA_FOLDER = os.path.join(HOME, 'data')
current = HOME

while 'pytorch_modular' not in os.listdir(current):
    current = Path(current).parent

sys.path.append(str(current))
sys.path.append(os.path.join(current, 'pytorch_modular'))

# let's first set the random seeds 
random.seed(69)
np.random.seed(69)
torch.manual_seed(69)


class LinearBlock(nn.Module):
    # let's define a class method that will map the activation name to the correct layer
    activation_functions = {"leaky_relu": nn.LeakyReLU,
                            "relu": nn.ReLU,
                            "tanh": nn.Tanh}

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation: str = 'leaky_relu',
                 is_final: bool = True,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # the idea here is quite simple. 
        components = [nn.Linear(in_features=in_features, out_features=out_features)]
        # depending on the value of 'is_final' 
        if not is_final:
            norm_layer = nn.BatchNorm1d(num_features=out_features)
            activation_layer = self.activation_functions[activation]()
            components.extend([norm_layer, activation_layer])

        self._block = nn.Sequential(*components)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._block.forward(x)

    def __str__(self) -> str:
        return self._block.__str__()

    @property
    def block(self):
        return self._block

    @block.setter
    def block(self, new_block: Union[nn.Sequential, nn.Module]):
        # make sure the new block is either 
        if isinstance(new_block, (nn.Sequential, nn.Module)):
            raise TypeError((f"The block is expected to be either of type {nn.Module} or {nn.Sequential}\n"
                             f"Found: {type(new_block)}"))

    def children(self) -> Iterator[Module]:
        return self.block.children()

    def named_children(self) -> Iterator[Module]:
        return self.block.children()


class LinearNet(nn.Module):
    def build(self,
              in_features: int,
              out_features: int,
              num_layers: int,
              activation: str,
              last_layer_is_final: bool = True):
        # the first and final layer are always present
        num_layers = num_layers + 2
        # compute the number of hidden units
        hidden_units = [int(u) for u in np.linspace(in_features, out_features, num=num_layers)]

        layers = [LinearBlock(in_features=hidden_units[i], out_features=hidden_units[i + 1], activation=activation,
                              is_final=False) for i in range(len(hidden_units) - 2)]

        # the last linear block should be set as 'final'
        layers.append(
            LinearBlock(in_features=hidden_units[-2], out_features=hidden_units[-1], is_final=last_layer_is_final,
                        activation=activation))

        return nn.Sequential(*layers)

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 num_layers: int = 1,
                 activation: str = 'leaky_relu',
                 last_layer_is_final: bool = True,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # let's make it a simple encoder
        self.net = self.build(in_features=in_features,
                              out_features=out_features,
                              num_layers=num_layers,
                              activation=activation,
                              last_layer_is_final=last_layer_is_final)

    def forward(self, x: torch.Tensor):
        return self.net.forward(x)

    def __str__(self) -> str:
        return self.net.__str__()

    def children(self) -> Iterator[Module]:
        return self.net.children()

    def named_children(self) -> Iterator[Module]:
        return self.net.children()


class AutoEncoder(nn.Module, ABC):
    def __init__(self,
                 in_features: int,
                 bottleneck: int,
                 num_layers: int,
                 activation: str,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.in_features = in_features
        self.bottleneck = bottleneck
        self.num_layers = num_layers
        self.activation = activation

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def encode(self, x: torch.Tensor):
        pass

    @abstractmethod
    def decode(self, x: torch.Tensor):
        pass


class BasicAutoEncoder(AutoEncoder):
    def __init__(self,
                 in_features: int,
                 bottleneck: int = None,
                 num_layers: int = 2,
                 activation: str = 'leaky_relu',
                 *args,
                 **kwargs):
        super().__init__(in_features=in_features,
                         bottleneck=bottleneck,
                         num_layers=num_layers,
                         activation=activation,
                         *args, **kwargs)

        if bottleneck is None:
            bottleneck = in_features // 2

        # The idea is simple, let's first build
        self._encoder = LinearNet(in_features=in_features,
                                  out_features=bottleneck,
                                  num_layers=num_layers,
                                  activation=activation,
                                  last_layer_is_final=False
                                  )

        self._decoder = LinearNet(in_features=bottleneck,
                                  out_features=in_features,
                                  num_layers=num_layers,
                                  activation=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._decoder(self._encoder(x))

    def encode(self, x: torch.Tensor):
        return self._encoder(x)

    def decode(self, x: torch.Tensor):
        return self._decoder(x)


class SparseAutoEncoder(BasicAutoEncoder):
    def __init__(self,
                 in_features: int,
                 bottleneck: int = None,
                 num_layers: int = 1,
                 encoder_sparse_layers: int = 1,
                 decoder_sparse_layers: int = 1,
                 activation: str = 'leaky_relu',
                 ):

        # call the constructor of the parent class
        super().__init__(in_features=in_features,
                         bottleneck=bottleneck,
                         num_layers=num_layers,
                         activation=activation)

        if encoder_sparse_layers > num_layers:
            raise ValueError((f"The number sparse encoder layers cannot be larger than the total number of layers.\n"
                              f"Found: {encoder_sparse_layers} spare layers and {num_layers} layers"))

        if decoder_sparse_layers > num_layers:
            raise ValueError((f"The number of sparse decoder layers cannot be larger than the total number of layers.\n"
                              f"Found: {decoder_sparse_layers} spare layers and {num_layers} layers"))

        self.encoder_sparse = encoder_sparse_layers
        self.decoder_sparse = decoder_sparse_layers

    def encode(self, x: torch.Tensor, with_activation: bool = False) -> Union[
        torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        if not with_activation:
            return super().encode(x)

        # we know that the total number of layers in the encoder: num_layers + 2
        # calculate the number of non-sparse layers
        non_sparse = self.num_layers + 2 - self.encoder_sparse

        sparse_outputs = []

        # iterate through each linear block in the 'encoder'
        for index, linear_block in enumerate(self._encoder):
            # first pass 'x' through the block
            x = linear_block(x)
            if index >= non_sparse:
                sparse_outputs.append(x)

        return x, sparse_outputs

    def encode(self, x: torch.Tensor, with_activation: bool = False) -> Union[
        torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        if not with_activation:
            return super().encode(x)

        # we know that the total number of linear layers: self.num_layers + 1
        non_sparse = self.num_layers + 1 - self.encoder_sparse

        sparse_outputs = []

        # iterate through each linear block in the 'encoder'
        for index, linear_block in enumerate(self._encoder.children()):
            # first pass 'x' through the block
            x = linear_block(x)
            if index >= non_sparse:
                sparse_outputs.append(x)

        return x, sparse_outputs

    def decode(self, x: torch.Tensor, with_activation: bool = False):
        if not with_activation:
            return super().decode(x)

        sparse_outputs = []

        # unlike the encoder, the sparse layers are at the beginning of the decoder block
        for index, linear_block in enumerate(self._decoder.children()):
            x = linear_block(x)
            if index < self.decoder_sparse:
                sparse_outputs.append(x)

        return x, sparse_outputs

    def forward(self, x: torch.Tensor, with_activation: bool = True):
        if not with_activation:
            return self.forward(x)

        x, encoder_activations = self.encode(x, with_activation=True)
        x, decoder_activations = self.decode(x, with_activation=True)

        activations = list(itertools.chain(encoder_activations, decoder_activations))

        return x, activations
