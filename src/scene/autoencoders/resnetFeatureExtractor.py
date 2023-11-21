"""
This script contains functionalities to build classifiers on top of the pretrained model
'RESNET 50' provided by pytorch module

This script is mainly inspired by this paper: 'https://arxiv.org/abs/1411.1792'
as they suggest an experimental framework to find the most transferable / general layers
in pretrained network. I am applying the same framework on the resnet architecture.
"""

import torch

from typing import Iterator, Union
from collections import OrderedDict

from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
# Bottleneck is the class that contains the Residual block
from torchvision.models.resnet import Bottleneck


LAYER_BLOCK = 'layer'
RESIDUAL_BLOCK = 'residual'


# let's create a utility function real quick
def contains_fc_layer(module: nn.Module) -> bool:
    """
    This function returns whether the module contains a Fully connected layer or not
    """
    m = isinstance(module, nn.Linear)
    sub_m = any([isinstance(m, nn.Linear) for m in module.modules()])
    return m and sub_m


# noinspection PyUnresolvedReferences,PyShadowingNames
class ResNetFeatureExtractor(nn.Module):
    default_transform = ResNet50_Weights.DEFAULT.transforms()
    
    # the model's architecture refers to blocks with the same number of channels as 'layers'
    # a function to build the transferred part from the original resnet model
    # in case of 'layer' blocks
    def __feature_extractor_layers(self, number_of_layers: int):
        # passing a negative value would mean retrieving the entire feature extractor
        number_of_layers = number_of_layers if number_of_layers > 0 else float('inf')

        modules_generator = self.__net.named_children()
        # the modules will be saved with their original names in an OrderedDict and later merged
        # into a single nn.Module using nn.Sequential
        modules_to_keep = []
        counter = 0

        for name, module in modules_generator:
            # we will only consider non fully connected components of the network
            if not contains_fc_layer(module) and (self.add_gb_avg or module != self.__net.avgpool):
                if 'layer' not in name or counter < number_of_layers:
                    modules_to_keep.append((name, module))
                    # only increment counter if the name contains 'layer': meaning it is a layer block
                    counter += int('layer' in name)

        self.modules = [m for _, m in modules_to_keep]
        fe = nn.Sequential(OrderedDict(modules_to_keep))
        return fe

    def __init__(self,
                 num_blocks: int,  # the number of blocks to keep
                 blocks_type: str = LAYER_BLOCK,  # the type of blocks to consider (layers or residual blocks)
                 freeze: Union[bool, int] = True,  # whether to freeze the chosen layers or not
                 add_global_average: bool = True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        # first make sure the blocks_type argument is correct
        assert blocks_type in [LAYER_BLOCK, RESIDUAL_BLOCK], "MAKE SURE TO PASS A SUPPORTED TYPE OF BLOCKS"

        self.num_blocks = num_blocks
        # make sure to explicitly
        self.__net = resnet50(ResNet50_Weights.DEFAULT)
        
        self.add_gb_avg = add_global_average
        self.feature_extractor = None
        self.modules = None
        self.feature_extractor = self.__feature_extractor_layers(self.num_blocks)

        # freeze the weights
        if isinstance(freeze, bool) and freeze:
            for p in self.feature_extractor.parameters():
                p.requires_grad = False    

        elif isinstance(freeze, int):
            counter = 0
            for name, module in self.feature_extractor.named_children():
                if 'layer' not in name or counter < freeze:
                    for p in module.parameters():
                        p.requires_grad = False
                    # only increment counter if the name contains 'layer': meaning it is a layer block
                    counter += int('layer' in name)



    def forward(self, x: torch.Tensor):
        # the forward function in the ResNet class simply calls the forward function
        # of each submodule consecutively: which is equivalent to saving all modules in a nn.Sequential module
        # and calling the forward method.
        return self.feature_extractor.forward(x)

    def __str__(self):
        # the default __str__ function will display the self.__net module as well
        # which might be confusing as .__net is definitely not part of the forward pass of the model
        return self.feature_extractor.__str__()
    
    def __repr__(self):
        return self.feature_extractor.__repr__() 
    
    def children(self) -> Iterator['Module']:
        # not overloading this method will return to an iterator with 2 elements: self.__net and self.feature_extractor
        return self.feature_extractor.children()

    def modules(self) -> Iterator[nn.Module]:
        return self.feature_extractor.modules()
