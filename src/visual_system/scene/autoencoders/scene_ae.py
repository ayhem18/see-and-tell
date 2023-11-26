"""
This script contains the definition, training as well as utility code for the AutoEncoder trained on the Big Bang Theory dataset.
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
import pandas as pd

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
from torchvision import transforms as trn



SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
current = SCRIPT_DIR

while 'src' not in os.listdir(current):
    current = Path(current).parent

PARENT_DIR = str(current)
sys.path.append(str(current))
sys.path.append(os.path.join(current, 'src'))

from torch import nn
# from src.scene.classifier.pretrained_conv import load_model, returnTF
from src.visual_system.scene.autoencoders.resnetFeatureExtractor import ResNetFeatureExtractor
import src.utilities.directories_and_files as dirf
import src.utilities.pytorch_utilities as pu

seed_everything(69, workers=True)


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








def returnTF(add_augment: bool = True):
    # load the image transformer
    tf = trn.Compose([
        trn.Resize((224, 224)),
        trn.TrivialAugmentWide(),
        trn.ToTensor(),        
    ]) if add_augment else trn.Compose([trn.Resize((224, 224)), trn.ToTensor()])
    
    return tf

# let's define 2 decoders

_decoder_14_14 = nn.Sequential(
            *[deconvolution_block(input_channels=1024, output_channels=256, stride=2, kernel_size=9),

              deconvolution_block(input_channels=256, output_channels=128, stride=1, kernel_size=6),

              deconvolution_block(input_channels=128, output_channels=64, stride=1, kernel_size=7),

              deconvolution_block(input_channels=64, output_channels=32, stride=1, kernel_size=7),

              deconvolution_block(input_channels=32, output_channels=16, stride=2, kernel_size=9),

              deconvolution_block(input_channels=16, output_channels=3, stride=2, kernel_size=4, final_layer=True),
              ])

_decoder_7_7 = nn.Sequential(
    *[
    deconvolution_block(input_channels=2048, output_channels=1024, stride=1, kernel_size=5, padding=1),

    deconvolution_block(input_channels=1024, output_channels=512, stride=1, kernel_size=5, padding=1), 

    deconvolution_block(input_channels=512, output_channels=256, stride=1, kernel_size=4), 

    deconvolution_block(input_channels=256, output_channels=128, stride=2, kernel_size=9),

    deconvolution_block(input_channels=128, output_channels=64, stride=1, kernel_size=6),

    deconvolution_block(input_channels=64, output_channels=32, stride=1, kernel_size=7),

    deconvolution_block(input_channels=32, output_channels=16, stride=1, kernel_size=7),

    deconvolution_block(input_channels=16, output_channels=8, stride=2, kernel_size=9),

    deconvolution_block(input_channels=8, output_channels=3, stride=2, kernel_size=4, final_layer=True)]
)


class SceneDenoiseAE(L.LightningModule):
    _input_shape = (3, 224, 224)

    def __init__(self, 
                 architecture: int = 50, 
                 num_blocks: int = 3,
                 freeze: int = 2,
                 num_vis_images: int = 3, 
                 *args: Any, **kwargs: Any):
        # the first step is to load the resnet model pretrained on the place365 dataset
        super().__init__(*args, **kwargs)
        self.encoder = ResNetFeatureExtractor(architecture=architecture, 
                                              num_blocks=num_blocks, 
                                              freeze=freeze, 
                                              add_global_average=False)
        
        o = self.encoder(torch.randn(1, 3, 224, 224))

        if o.shape not in [(1, 2048, 7, 7), (1, 1024, 14, 14)]:
            raise ValueError(f"Please make sure the encoder is chosen such that the output is in {[(1, 2048, 7, 7), (1, 1024, 14, 14)]}")
        
        # at this point we are sure the model will not silently break.
        self.decoder = _decoder_14_14 if o.shape == (1, 1024, 14, 14) else _decoder_7_7

        # the number of images to visualize in a validation step
        self.num_vis_images = num_vis_images

        # shouldn't call save_hyperparameters() since, the 'conv_block' object is a nn.Module and should be quite heavy in size
        self.save_hyperparameters()

        self.log_data = pd.DataFrame(data=[], columns=['image', 'noisy_image', 'reconstructed_image', 'val_loss', 'epoch'])

    def _forward_pass(self, batch, loss_reduced: bool = True):
        x = batch
        x_noise = add_noise(x, noise_factor=random.random() * 0.5)
        # make sure both x, and x_noise are of the expected dimensions
        batch_size = x.size(dim=0)

        if tuple(x.shape) != ((batch_size,) + self._input_shape):
            raise ValueError(f"The input is not of the expected shape: expected {self._input_shape}. "
                             f"Found: {x[0].shape}")
        x_r = self.decoder(self.encoder(x_noise))
        # the loss is the sum of the Squared Error between the constructed image and the original image
        mse_loss = F.mse_loss(x_r, x, reduction=('sum' if loss_reduced else 'none'))
        return mse_loss, x_noise, x_r

    def training_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        mse_loss, _, _ = self._forward_pass(batch)
        self.log(name='train_loss', value=mse_loss.cpu().item())
        return mse_loss

    def validation_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> STEP_OUTPUT:        
        # applying such process on every batch in the validation set will significantly slow the training process.
        mse_losses, x_noise, x_r = self._forward_pass(batch, loss_reduced=False)
        # calculate the mse loss value
        mse = torch.sum(mse_losses).cpu().item()
        # first log the validation loss
        self.log(name='val_loss', value=mse)  

        if batch_idx <= 2:
            # compute the loss for each image 
            image_losses = torch.sum(mse_losses, dim=(1, 2, 3))
            # extract the images with the largest loss
            top_losses, top_indices = torch.topk(input=image_losses, k=self.num_vis_images, dim=-1)

            # convert all the data to numpy arrays
            b, xn, xr = (batch[top_indices].detach().cpu().permute(0, 2, 3, 1).numpy(), 
                        x_noise[top_indices].detach().cpu().permute(0, 2, 3, 1).numpy(), 
                        x_r[top_indices].detach().cpu().permute(0, 2, 3, 1).numpy())
            
            top_losses = top_losses.detach().cpu().numpy()

            data = [[wandb.Image(img), wandb.Image(img_noise), wandb.Image(img_r), l, self.current_epoch] 
                    for img, img_noise, img_r, l in zip(b, xn, xr, top_losses)]

            batch_df = pd.DataFrame(data=data, columns=['image', 'noisy_image', 'reconstructed_image', 'val_loss', 'epoch'])

            self.log_data = pd.concat([self.log_data, batch_df], axis=0)

            self.logger.log_table(key='val_summary', dataframe=self.log_data)
            return image_losses
    

    def configure_optimizers(self):
        # since the encoder is pretrained, we would like to avoid significantly modifying its weights/
        # on the other hand, the rest of the AE should have higher learning rates.
        parameters = [{"params": self.encoder.parameters(), "lr": 10 ** -5},
                      {"params": self.decoder.parameters(), "lr": 10 ** -1}]
        # add a learning rate scheduler        
        optimizer = optim.Adam(parameters)
        # create lr scheduler
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.995)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def forward(self, x: torch.Tensor):
        _, _, x_r =  self._forward_pass(x)
        return x_r


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


# we will need better checkpointing
from lightning.pytorch.callbacks import ModelCheckpoint


def train_ae(model: SceneDenoiseAE,
             train_dir: Union[str, Path],
             val_dir: Union[str, Path] = None,
             log_dir: Union[str, Path] = None,
             image_extensions: Iterable[str] = None,
             run_name: str = None,
             batch_size: int = 32,
             num_epochs: int = 10, 
             add_augmentation: bool = True):
    
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
    model_transformation = returnTF(add_augment=add_augmentation)

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
                            pin_memory=True)
    else:
        val_dl = None
    
    wandb_logger = WandbLogger(project='cntell_auto_encoder',
                               log_model="all", 
                               save_dir=log_dir, 
                               name=run_name)

    checkpnt_callback = ModelCheckpoint(dirpath=log_dir, 
                                        save_top_k=5, 
                                        monitor="val_loss",
                                        mode='min', 
                                        # save the checkpoint with the epoch and validation loss
                                        filename='autoencoder-{epoch:02d}-{val_loss:06d}')

    # define the trainer
    trainer = L.Trainer(accelerator='gpu',
                        devices=1,
                        logger=wandb_logger,
                        default_root_dir=log_dir,
                        
                        max_epochs=num_epochs,
                        check_val_every_n_epoch=5,

                        deterministic=True,
                        callbacks=[checkpnt_callback])

    # the val_dataloaders have 'None' values as default
    trainer.fit(model=model,
                train_dataloaders=train_dl,
                val_dataloaders=val_dl
                )


def main(model, run_name: str):
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

    # model = SceneDenoiseAE()

    train_ae(
            model=model,
            train_dir=train_dir,
             val_dir=val_dir,            
             run_name=run_name,
             batch_size=32,
             log_dir=os.path.join(logs, f'exp_{len(os.listdir(logs)) + 1}'),     
             num_epochs=250,
             add_augmentation=True)    


def sanity_check(run_name):
    wandb.login(key='36259fe078be47d3ffd8f3b2628a4d773c6e1ce7')
    train_dir = os.path.join(PARENT_DIR, 'src', 'scene', 'sanity_train')
    val_dir = os.path.join(PARENT_DIR, 'src', 'scene', 'val_dir')

    logs = os.path.join(PARENT_DIR, 'src', 'scene', 'autoencoders', 'runs')
    os.makedirs(logs, exist_ok=True)

    model = SceneDenoiseAE(architecture=152, num_blocks=4, freeze=2)

    train_ae(
            model=model,
            train_dir=train_dir,
             val_dir=val_dir,            
             run_name=run_name,
             batch_size=32,
             log_dir=os.path.join(logs, f'exp_{len(os.listdir(logs)) + 1}'),     
             num_epochs=400,
             add_augmentation=False)

if __name__ == '__main__':
    # model = SceneDenoiseAE(architecture=152, num_blocks=4, freeze=2)
    model = SceneDenoiseAE()
    main(model, 'ae_sum_loss')
    # sanity_check('ae_sanity_check_4')
