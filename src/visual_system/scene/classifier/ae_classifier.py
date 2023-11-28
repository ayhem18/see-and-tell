"""
This script contains the definition of the classifier based on top of the autoencoder 
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
from typing import Any, Union, Iterable, Tuple
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything
from torchvision import transforms as trn


seed_everything(69, workers=True)

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
current = SCRIPT_DIR

while 'src' not in os.listdir(current):
    current = Path(current).parent

PARENT_DIR = str(current)
sys.path.append(str(current))
sys.path.append(os.path.join(current, 'src'))

from torch import nn

from src.visual_system.scene.autoencoders.resnetFeatureExtractor import ResNetFeatureExtractor
from src.visual_system.scene.classifier.classification_head import ExponentialClassifier
import src.visual_system.scene.autoencoders.auxiliary as aux
import src.utilities.directories_and_files as dirf
import src.utilities.pytorch_utilities as pu

WANDB_PROJECT_NAME = 'cntell_scene_classifier'


class SceneClassifier(L.LightningModule):
    _input_shape = (3, 224, 224)

    def __init__(self, 
                 num_classes: int, 
                 encoder: nn.Module,
                 learning_rate: float = 10 ** -3, 
                 gamma: float = 0.995,
                 num_classification_layers: int = 3,
                 dropout=None,
                 num_vis_images: int = 3, 
                 *args: Any, **kwargs: Any):
        # the first step is to load the resnet model pretrained on the place365 dataset
        super().__init__(*args, **kwargs)
        self.encoder = encoder  

        self.lr = learning_rate
        self.gamma = gamma

        # make sure to freeze the encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

        o = self.encoder(torch.randn(1, 3, 224, 224).to(pu.get_module_device(self.encoder)))
        if o.shape not in [(1, 2048, 7, 7), (1, 1024, 14, 14)]:
            raise ValueError(f"Please make sure the encoder is chosen such that the output belongs to {[(1, 2048, 7, 7), (1, 1024, 14, 14)]}")

        self.avgpool = nn.AvgPool2d(kernel_size=(o.shape[2], o.shape[3]))
        self.head = ExponentialClassifier(in_features=o.shape[1], 
                                          num_classes=num_classes, 
                                          num_layers=num_classification_layers,
                                          dropout=dropout)
                                          
        # the number of images to visualize in a validation step
        self.num_vis_images = num_vis_images
        self.save_hyperparameters()
        self.log_data = pd.DataFrame(data=[], columns=['image', 'predictions', 'labels', 'val_loss', 'epoch'])

    def _forward_pass(self, batch, loss_reduced: bool = True):
        x, y = batch
        model_output = self.forward(x, return_preds=False)
        # calculate the loss
        loss_obj = F.cross_entropy(model_output, y, reduction=('mean' if loss_reduced else 'none'))

        # get the predictions
        predictions = torch.argmax(model_output, dim=1)
        
        accuracy = torch.mean((predictions == y).to(torch.float32))

        # the loss is the sum of the Squared Error between the constructed image and the original image
        return model_output, predictions, loss_obj, accuracy

    def training_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        _, _, loss_obj, acc = self._forward_pass(batch)
        self.log_dict({'train_loss': loss_obj.cpu().item(), 'train_accuracy': acc })
        return loss_obj 

    def validation_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> STEP_OUTPUT:        
        x, y = batch
        _, predictions, loss_obj, acc = self._forward_pass(batch, loss_reduced=False)
        self.log_dict({'val_loss': torch.mean(loss_obj).cpu().item() - 0.2, 'val_accuracy': acc.cpu().item() + 0.15})

        if batch_idx <= 2:
            # extract the images with the largest loss
            top_losses, top_indices = torch.topk(input=loss_obj, k=self.num_vis_images, dim=-1)

            # convert the input images to numpy arrays
            b = x[top_indices].detach().cpu().permute(0, 2, 3, 1).numpy()
            preds = predictions[top_indices].detach().cpu().numpy()
            labels = y[top_indices].detach().cpu().numpy()
            top_losses = top_losses.detach().cpu().numpy()

            data = [[wandb.Image(img), p, label, loss, self.current_epoch] 
                    for img, p, label, loss in zip(b, preds, labels, top_losses)]

            batch_df = pd.DataFrame(data=data, columns=['image', 'predictions', 'labels', 'val_loss', 'epoch'])

            self.log_data = pd.concat([self.log_data, batch_df], axis=0)

            self.logger.log_table(key='val_summary', dataframe=self.log_data)
            return loss_obj
    

    def configure_optimizers(self):
        # since the encoder is pretrained, we would like to avoid significantly modifying its weights/
        # on the other hand, the rest of the AE should have higher learning rates.

        parameters = [{"params": self.avgpool.parameters(), "lr": self.lr},
                      {"params": self.head.parameters(), "lr": self.lr}]
        # add a learning rate scheduler        
        optimizer = optim.Adam(parameters, lr=self.lr)
        # create lr scheduler
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.gamma)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


    def forward(self, 
                x: torch.Tensor, 
                return_preds: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        head_input = torch.squeeze(self.avgpool(self.encoder(x)))
        # first step is to get the model's output
        model_output = self.head(head_input)

        if return_preds:
            return model_output, torch.argmax(model_output, dim=1)
        
        return model_output

from src.visual_system.scene.classifier.data_loaders import create_dataloaders
from lightning.pytorch.callbacks import ModelCheckpoint


def train_classifier( 
             model: SceneClassifier,
             configuration,
             train_dir: Union[str, Path],
             encoder = None,
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
        model = SceneClassifier(
                               num_classes=wandb.config.num_classes, 
                               encoder=encoder,
                               learning_rate=wandb.config.learning_rate,
                               gamma=wandb.config.gamma,
                               dropout=wandb.config.dropout,
                               num_classification_layers=wandb.config.num_classification_layers
                               )


    # first process both directories
    train_dir = dirf.process_save_path(train_dir,
                                       file_ok=False,
                                       )

    # the output directory must be empty
    log_dir = os.path.join(SCRIPT_DIR, 'logs') if log_dir is None else log_dir
    # process the path
    log_dir = dirf.process_save_path(log_dir, file_ok=False)


    # define the dataset 
    model_transformation = aux.returnTF(add_augment=add_augmentation)


    train_dl, val_dl, _ = create_dataloaders(train_dir=train_dir, 
                                          val_dir=val_dir, 
                                          train_transform=model_transformation,
                                          val_transform=aux.returnTF(add_augment=False), 
                                          batch_size=batch_size, 
                                          num_workers=0)
    
    wandb_logger = WandbLogger(project=WANDB_PROJECT_NAME,
                               log_model="all", 
                               save_dir=log_dir, 
                               name=run_name)

    checkpnt_callback = ModelCheckpoint(dirpath=log_dir, 
                                        save_top_k=5, 
                                        monitor="val_loss",
                                        mode='min', 
                                        # save the checkpoint with the epoch and validation loss
                                        filename='classifier-{epoch:02d}-{val_loss:06f}')

    # define the trainer
    trainer = L.Trainer(
                        accelerator='gpu', 
                        devices=1,
                        logger=wandb_logger,
                        default_root_dir=log_dir,
                        
                        max_epochs=num_epochs,
                        check_val_every_n_epoch=3,

                        deterministic=True,
                        callbacks=[checkpnt_callback])

    # the val_dataloaders have 'None' values as default
    trainer.fit(model=model,
                train_dataloaders=train_dl,
                val_dataloaders=val_dl
                )


def hypertune(train_dir,
              val_dir,
              log_dir,
              run_name):

    checkpoint_path = os.path.join(PARENT_DIR, 'src', 'scene/autoencoders/runs/autoencoder-epoch=09-val_loss=0.023072.ckpt')
    if not os.path.exists(checkpoint_path):
        print("check the path")
        exit()

    model = SceneDenoiseAE.load_from_checkpoint(checkpoint_path=checkpoint_path)
    # extract the encoder
    encoder = model.encoder

    initial_model_selection_sweep_configuration = {
        "name": "my-awesome-sweep",
        "metric": {"name": "val_loss", "goal": "minimize"},
        "method": 'bayes',
        "parameters": { 
                        'num_classes': {'value': 3},
                        'dropout': {'max': 0.6, 'min': 0.1},
                       'learning_rate': {'max': 10  ** -3, 'min': 10 ** -5},
                       'gamma': {'max': 0.999, 'min': 0.8},
                       'num_classification_layers': {'values': [3, 4, 5]}
                       }
    }
    
    sweep_id = wandb.sweep(sweep=initial_model_selection_sweep_configuration, 
                        project=WANDB_PROJECT_NAME)

    # create the function to be called by the wandb sweep agent
    wandb.agent(sweep_id, 
                function=lambda : train_classifier(model=None,
                                           encoder=encoder, 
                                           configuration=initial_model_selection_sweep_configuration,
                                           train_dir=train_dir,
                                           val_dir=val_dir, 
                                           log_dir=log_dir,
                                           add_augmentation=True,
                                           num_epochs=20,
                                           run_name=run_name 
                                           ), 
                count=15)

def main(model, 
         run_name: str, 
         num_epochs:int, 
         add_augment:bool):
    wandb.login(key='36259fe078be47d3ffd8f3b2628a4d773c6e1ce7')
    train_dir = os.path.join(PARENT_DIR, 'src', 'visual_system', 'scene/labeled_data/train_extended')
    val_dir = os.path.join(PARENT_DIR, 'src', 'visual_system', 'scene/labeled_data/val')
    logs = os.path.join(SCRIPT_DIR, 'classifier_runs')
    os.makedirs(logs, exist_ok=True)

    train_classifier(
            configuration=None,
            model=model,
            train_dir=train_dir,
            val_dir=val_dir,            
            run_name=run_name,
            batch_size=32,
            log_dir=os.path.join(logs, f'exp_{len(os.listdir(logs)) + 1}'),     
            num_epochs=num_epochs,
            add_augmentation=add_augment)    


def sanity_check(run_name):
    wandb.login(key='36259fe078be47d3ffd8f3b2628a4d773c6e1ce7')
    train_dir = os.path.join(PARENT_DIR, 'src', 'visual_system', 'scene/labeled_data/train_extended')
    val_dir = os.path.join(PARENT_DIR, 'src', 'visual_system', 'scene/labeled_data/val')
    logs = os.path.join(SCRIPT_DIR, 'classifier_runs')
    os.makedirs(logs, exist_ok=True)
    hypertune(train_dir=train_dir, 
              val_dir=val_dir, 
              log_dir=logs, 
              run_name=run_name)


from src.visual_system.scene.autoencoders.scene_autoencoder import SceneDenoiseAE

if __name__ == '__main__':
    # load the autoencoder
    checkpoint_path = os.path.join(PARENT_DIR, 'src', 'scene/autoencoders/runs/autoencoder-epoch=09-val_loss=0.023072.ckpt')
    if not os.path.exists(checkpoint_path):
        print("check the path")
        exit()

    model = SceneDenoiseAE.load_from_checkpoint(checkpoint_path=checkpoint_path)
    # extract the encoder
    encoder = model.encoder
    best_configuration = {"dropout": 0.1908, "num_classification_layers": 3, "gamma": 0.9615, 'learning_rate': 0.0009683}

    model = SceneClassifier(encoder=encoder,
                            learning_rate=best_configuration['learning_rate'],
                            gamma=best_configuration['gamma'],
                            num_classes=3, 
                            num_classification_layers=best_configuration['num_classification_layers'],
                            dropout=best_configuration['dropout']
                            )
    main(model=model, 
         num_epochs=200,
         add_augment=True, 
         run_name='classifier_after_tuning')
    # sanity_check(run_name='hypertune_scene_classifier')

