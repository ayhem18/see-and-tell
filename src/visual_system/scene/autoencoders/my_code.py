import os, sys


from pathlib import Path


HOME = os.path.dirname(os.path.realpath(__file__))
DATA_FOLDER = os.path.join(HOME, 'data')

current = HOME



while 'pytorch_modular' not in os.listdir(current):
    current = Path(current).parent

sys.path.append(str(current))
sys.path.append(os.path.join(current, 'pytorch_modular'))

import torch
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as tr
from building_blocks import BasicAutoEncoder
from pytorch_modular.image_classification import engine_classification as engine
from torch.optim import Adam 
from torch.optim.lr_scheduler import LinearLR
from torch.nn import MSELoss
from typing import Tuple, List
from pytorch_modular.pytorch_utilities import get_module_device
from building_blocks import SparseAutoEncoder


# since the task is not classification, a new dataset should be created to fit our training objectives
class GenerativeWrapper(Dataset):
    def __init__(self, data: Dataset) -> None:
        super().__init__()
        self.original_data = data

    def __getitem__(self, index):
        data, _ = self.original_data.__getitem__(index)
        return (data, data)
    
    def __len__(self):
        return len(self.original_data)
    


def solution():
    mnist_train = MNIST(root=DATA_FOLDER, train=True, download=True, transform=tr.Compose([tr.ToTensor(), lambda x: x.reshape(-1)]))
    mnist_test = MNIST(root=DATA_FOLDER, train=False, download=True, transform=tr.Compose([tr.ToTensor(), lambda x: x.reshape(-1)]))

    from sklearn.model_selection import train_test_split
    # create the val split
    mnist_train, mnist_val = train_test_split(mnist_train, random_state=69, test_size=0.1)

    # wrap the different splits
    mnist_train = GenerativeWrapper(mnist_train)
    mnist_val = GenerativeWrapper(mnist_val)
    mnist_test = GenerativeWrapper(mnist_test)


    # create dataloader
    train_dl = DataLoader(dataset=mnist_train, 
                        batch_size=512, 
                        shuffle=True,
                        pin_memory=True,
                        drop_last=True,
                        num_workers=os.cpu_count() // 2, 
                        ) 

    val_dl = DataLoader(dataset=mnist_val, 
                        batch_size=512, 
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False,
                        num_workers=os.cpu_count() // 2) 

    # for im, label in train_dl:
    #     print(im.shape)
    #     break

    # model = UnderCompleteAE(in_features=784, bottleneck=256, num_layers=2)

    model = SparseAutoEncoder(in_features=784, 
                              bottleneck=1024, 
                              num_layers=3, 
                              encoder_sparse_layers=1, 
                              decoder_sparse_layers=1, 
                              activation='relu')
    
    # print(model)
    # exit()

    # the loss associated with the sparse AE is not straightforward and thus should be written and fed to teh training functionalities

    def compute_loss(model_output: Tuple[torch.Tensor, List[torch.Tensor]], 
                    y: torch.Tensor,
                    activation_threshold: float, 
                    alpha: float, 
                    epsilon: float = 10 ** -5 
                    ):
        
        device = get_module_device(model)

        # first decouple the data from 
        x, activations = model_output
        # activations are expected to a list of N torch.Tensors where each element is the activations of a layer across the entire  
        mse_loss = MSELoss()(x, y)
        sparse_loss = torch.zeros(1).to(device=device)
        
        kl_loss_function = nn.KLDivLoss(reduction='batchmean').to(device=device)

        # iterate through each of the activation layers
        # we just need to compute
        

        for a in activations:
            if a.dim() != 2:
                raise ValueError((f"The activations are expected to be 1-dimensional\n"
                                f"Found: {a.dim() - 1} dimensions"))
            if not torch.all(a >= 0):
                raise ValueError(f"The activations are expected to be positive. Please make sure to use the 'ReLU' activation")
            
            # adding a small epsilon to avoid issues with log of '0'
            a = (a + epsilon).log()
            # the target should be 
            temp_loss = kl_loss_function(a, torch.full(size=a.shape, fill_value=activation_threshold).to(device=device)).abs()
            sparse_loss += temp_loss

        # divide by the number of activations
        sparse_loss /= len(activations)

        # return a linear combination of the mse_loss and sparse_loss
        return mse_loss + alpha * sparse_loss


    # training     
    optimizer = Adam(model.parameters(), lr=0.01)
    scheduler = LinearLR(optimizer=optimizer, start_factor=1, end_factor=0.05, total_iters=100)

    output_layer = lambda x: x
    # loss_function = MSELoss()

    train_configuration = {"optimizer": optimizer, 
                        'scheduler': scheduler,
                        'min_val_loss': 10 ** -4,
                        'max_epochs': 25,
                        'report_epoch': 5,
                        'metrics': {},
                        'output_layer': output_layer,
                        'compute_loss': compute_loss, 
                        'compute_loss_kwargs': {"activation_threshold": 0.05, "alpha": 0.5}
                        }
    
    engine.train_model(model=model, 
                    train_dataloader=train_dl, 
                    test_dataloader=val_dl, 
                    train_configuration=train_configuration,
                    log_dir=os.path.join(HOME, 'runs'),
                    )


if __name__ == '__main__':
    solution()

