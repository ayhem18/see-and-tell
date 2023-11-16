"""This script contains the functionalities needed to load a resnet18 model pretrained on the places365 dataset.
"""

import os
import torch
from torch import nn
from collections import OrderedDict

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def contains_fc_layer(module: nn.Module) -> bool:
    """
    This function returns whether the module contains a Fully connected layer or not
    """
    m = isinstance(module, nn.Linear)
    sub_m = any([isinstance(m, nn.Linear) for m in module.modules()])
    return m and sub_m


def _recursion_change_bn(module):
    # this function sets the track_running_stats field to '1' for every Batchnorm Layer in a give module
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = 1
    else:
        for i, (name, module1) in enumerate(module._modules.items()):
            module1 = _recursion_change_bn(module1)
    return module


def load_model(feature_extractor: bool = False):
    # this model has a last conv feature map as 14x14

    model_file = 'wideresnet18_places365.pth.tar'
    if not os.access(model_file, os.W_OK):
        model_weights = 'http://places2.csail.mit.edu/models_places365/' + model_file
        model_py = 'https://raw.githubusercontent.com/csailvision/places365/master/wideresnet.py'

        # download the weights if needed
        if not os.path.exists(os.path.join(SCRIPT_DIR, model_weights)):
            os.system(f'wget {model_weights}')

        # download the a utility script
        if not os.path.exists(os.path.join(SCRIPT_DIR, model_py)):
            os.system(f'wget {model_py}')

    import wideresnet
    model = wideresnet.resnet18(num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)

    # hacky way to deal with the upgraded batchnorm2D and avgpool layers...
    for i, (name, module) in enumerate(model._modules.items()):
        module = _recursion_change_bn(model)
    model.avgpool = torch.nn.AvgPool2d(kernel_size=14, stride=1, padding=0)

    # at this point determine whether to retur t
    if feature_extractor:
        modules_generator = model.named_children()
        modules_to_keep = [(name, mod) for name, mod in modules_generator if
                           not (contains_fc_layer(mod) or mod == model.avgpool)]
        return nn.Sequential(OrderedDict(modules_to_keep))

    return model


from torchvision import transforms as trn


def returnTF():
    # load the image transformer
    tf = trn.Compose([
        trn.Resize((224, 224)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return tf
