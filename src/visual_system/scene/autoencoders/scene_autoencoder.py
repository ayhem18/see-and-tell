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


import src.visual_system.scene.autoencoders.auxiliary as aux

from torch import nn
# from src.scene.classifier.pretrained_conv import load_model, returnTF
from src.visual_system.scene.autoencoders.resnetFeatureExtractor import ResNetFeatureExtractor
import src.utilities.directories_and_files as dirf
import src.utilities.pytorch_utilities as pu


seed_everything(69, workers=True)

WANDB_PROJECT_NAME = 'cntell_ae_2'

# let's define 2 decoders
_decoder_14_14 = nn.Sequential(
            *[aux.deconvolution_block(input_channels=1024, output_channels=256, stride=2, kernel_size=9),

              aux.deconvolution_block(input_channels=256, output_channels=128, stride=1, kernel_size=6),

              aux.deconvolution_block(input_channels=128, output_channels=64, stride=1, kernel_size=7),

              aux.deconvolution_block(input_channels=64, output_channels=32, stride=1, kernel_size=7),

              aux.deconvolution_block(input_channels=32, output_channels=16, stride=2, kernel_size=9),

              aux.deconvolution_block(input_channels=16, output_channels=3, stride=2, kernel_size=4, final_layer=True),
              ])

_decoder_7_7 = nn.Sequential(
    *[
    aux.deconvolution_block(input_channels=2048, output_channels=1024, stride=1, kernel_size=5, padding=1),

    aux.deconvolution_block(input_channels=1024, output_channels=512, stride=1, kernel_size=5, padding=1), 

    aux.deconvolution_block(input_channels=512, output_channels=256, stride=1, kernel_size=4), 

    aux.deconvolution_block(input_channels=256, output_channels=128, stride=2, kernel_size=9),

    aux.deconvolution_block(input_channels=128, output_channels=64, stride=1, kernel_size=6),

    aux.deconvolution_block(input_channels=64, output_channels=32, stride=1, kernel_size=7),

    aux.deconvolution_block(input_channels=32, output_channels=16, stride=1, kernel_size=7),

    aux.deconvolution_block(input_channels=16, output_channels=8, stride=2, kernel_size=9),

    aux.deconvolution_block(input_channels=8, output_channels=3, stride=2, kernel_size=4, final_layer=True)]
)


class SceneDenoiseAE(L.LightningModule):
    _input_shape = (3, 224, 224)

    def __init__(self, 
                 frozen_lr: float = 10 ** -5,
                 trained_lr: float = 10 ** -1,
                 gamma: float = 0.995,
                 architecture: int = 50, 
                 num_blocks: int = 3,
                 freeze: int = 2,
                 num_vis_images: int = 5, 
                 *args: Any, **kwargs: Any):
        # the first step is to load the resnet model pretrained on the place365 dataset
        super().__init__(*args, **kwargs)

        # save the different learning rates and the coefficient of the exponential scheduler
        self.frozen_lr = frozen_lr
        self.trained_lr = trained_lr
        self.gamma = gamma

        self.encoder = ResNetFeatureExtractor(architecture=architecture, 
                                              num_blocks=num_blocks, 
                                              freeze=freeze, 
                                              add_global_average=False)
        
        o = self.encoder(torch.randn(1, 3, 224, 224).to(pu.get_module_device(self.encoder)))
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
        x_noise = aux.add_noise(x, noise_factor=random.random() * 0.4)
        # make sure both x, and x_noise are of the expected dimensions
        batch_size = x.size(dim=0)

        # if tuple(x_noise[0].shape) != self._input_shape:
        #     raise ValueError(f"The input is not of the expected shape: expected {self._input_shape}. "
        #                      f"Found: {tuple(x_noise[0].shape)}")

        x_r = self.decoder(self.encoder(x_noise))
        # the loss is the sum of the Squared Error between the constructed image and the original image
        mse_loss = F.mse_loss(x_r, x, reduction=('mean' if loss_reduced else 'none'))
        return mse_loss, x_noise, x_r

    def training_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        mse_loss, _, _ = self._forward_pass(batch)
        self.log(name='train_loss', value=mse_loss.cpu().item())
        return mse_loss

    def validation_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> STEP_OUTPUT:        
        # applying such process on every batch in the validation set will significantly slow the training process.
        mse_losses, x_noise, x_r = self._forward_pass(batch, loss_reduced=False)
        # calculate the mse loss value
        mse = torch.mean(mse_losses).cpu().item()
        # first log the validation loss
        self.log(name='val_loss', value=mse)  

        if batch_idx <= 2:
            # compute the loss for each image 
            image_losses = torch.mean(mse_losses, dim=(1, 2, 3))
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
        parameters = [{"params": self.encoder.parameters(), "lr": self.frozen_lr},
                      {"params": self.decoder.parameters(), "lr": self.trained_lr}]
        # add a learning rate scheduler        
        optimizer = optim.Adam(parameters)
        # create lr scheduler
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.995)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def forward(self, x: torch.Tensor):
        _, _, x_r =  self._forward_pass(x)
        return x_r


# we will need better checkpointing
from lightning.pytorch.callbacks import ModelCheckpoint

def train_ae(model: SceneDenoiseAE,
             train_dir: Union[str, Path],
             configuration = None,
             val_dir: Union[str, Path] = None,
             log_dir: Union[str, Path] = None,
             image_extensions: Iterable[str] = None,
             run_name: str = None,
             batch_size: int = 32,
             num_epochs: int = 10, 
             add_augmentation: bool = True):
    
    wandb.init(project=WANDB_PROJECT_NAME, 
               config=configuration, 
               name=run_name)
    
    wandb_logger = WandbLogger(project=WANDB_PROJECT_NAME,
                            log_model="all", 
                            save_dir=log_dir, 
                            name=run_name)

    if configuration is not None:
        model = SceneDenoiseAE(frozen_lr=wandb.config.frozen_lr,
                               trained_lr=wandb.config.trained_lr,
                               gamma=wandb.config.gamma,
                               architecture=wandb.config.architecture,
                               num_blocks=wandb.config.num_blocks, 
                               freeze=wandb.config.freeze)

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
    model_transformation = aux.returnTF(add_augment=add_augmentation)

    train_dataset = aux.GenerativeDS(data_path=train_dir,
                                 transformation=model_transformation)

    train_dl = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          pin_memory=True)
    
    if val_dir is not None:
        val_dl = DataLoader(dataset=aux.GenerativeDS(data_path=val_dir, 
                                                     transformation=model_transformation),
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True)
    else:
        val_dl = None
    

    checkpnt_callback = ModelCheckpoint(dirpath=log_dir, 
                                        save_top_k=5, 
                                        monitor="val_loss",
                                        mode='min', 
                                        # save the checkpoint with the epoch and validation loss
                                        filename='autoencoder-{epoch:02d}-{val_loss:06f}')

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


def sanity_check_model_selection(all_train_dir: Union[str, Path], 
                       all_val_dir: Union[str, Path], 
                       log_dir: Union[Path, str],
                       run_name,
                       portion: float = 0.2, 
                       ):
    
    sanity_train = os.path.join(Path(all_train_dir).parent, 'sanity_train')
    sanity_val = os.path.join(Path(all_val_dir).parent, 'sanity_val')

    # let's split the data into train and test splits
    _, train_data = train_test_split(os.listdir(all_train_dir), test_size=portion, random_state=69)
    _, val_data = train_test_split(os.listdir(all_val_dir), test_size=portion, random_state=69)

    if not os.path.exists(sanity_train):
        os.makedirs(sanity_train)
        for f in train_data:
            shutil.copyfile(os.path.join(all_train_dir, f), os.path.join(sanity_train, f))

    if not os.path.exists(sanity_val):
        os.makedirs(sanity_val)
        for f in val_data:
            shutil.copyfile(os.path.join(all_val_dir, f), os.path.join(sanity_val, f))

    initial_model_selection_sweep_configuration = {
        "name": "my-awesome-sweep",
        "metric": {"name": "train_loss", "goal": "minimize"},
        "method": 'bayes',
        "parameters": {'architecture': {'value': 50}, 
                       'num_blocks': {'values': [3, 4]},
                       'freeze': {'values': [2, 3]},
                       'frozen_lr': {'min': 10 ** -6, 'max': 10 ** -5}, 
                       'trained_lr': {'max': 10  ** -1, 'min': 10 ** -3},
                       'gamma': {'max': 0.999, 'min': 0.8}
                       }
    }
    
    sweep_id = wandb.sweep(sweep=initial_model_selection_sweep_configuration, 
                        project=WANDB_PROJECT_NAME)

    # create the function to be called by the wandb sweep agent
    wandb.agent(sweep_id, 
                function=lambda : train_ae(model=None,
                                           configuration=initial_model_selection_sweep_configuration,
                                           train_dir=sanity_train,
                                           val_dir=sanity_val, 
                                           log_dir=log_dir,
                                           add_augmentation=False,
                                           num_epochs=15,
                                           run_name=run_name 
                                           ), 
                count=15)
    
from configparser import ConfigParser

if __name__ == '__main__':
    wandb.login(key='36259fe078be47d3ffd8f3b2628a4d773c6e1ce7')
    train_dir = os.path.join(PARENT_DIR, 'src', 'visual_system', 'scene', 'unlabeled_data', 'train_extended')
    val_dir = os.path.join(PARENT_DIR, 'src', 'visual_system', 'scene', 'unlabeled_data', 'val_extended')
    log_dir = os.path.join(PARENT_DIR, 'src', 'scene', 'autoencoders', 'runs')

    # sanity_check_model_selection(all_train_dir=train_dir, 
    #                              all_val_dir=val_dir,
    #                              log_dir=log_dir, 
    #                              run_name='arch_selection')

    # load the architecture with the highest capacity so far
    config_path = os.path.join(SCRIPT_DIR, 'ae_architecture.ini')

    if not os.path.exists(config_path):
        raise ValueError(f"WHERE IS THE CONFIG FILE !!!!")

    # parse the config file
    cg = ConfigParser()
    cg.read(config_path)
    model_parameters = {k: float(v) for k, v in cg['architecture'].items()}

    model = SceneDenoiseAE(**model_parameters)
    


    # with open(os.path.join(SCRIPT_DIR, 'ae_architecture.json')) as f: 
    #     model_parameters = json.load(f)
    # model_parameters = {k: float(v) for k, v in model_parameters['architecture']}
    # # model_parameters = {k: float(v) for k, v in model_parameters.items()}

    train_ae(model=model, 
             train_dir=train_dir, 
             val_dir=val_dir,
             log_dir=log_dir, 
             num_epochs=300,
             run_name='auto_encoder_train',
             add_augmentation=True
             )

