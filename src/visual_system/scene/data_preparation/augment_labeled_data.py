"""This script contains the data augmentation functionalities
"""
import os
import sys
from pathlib import Path

current = os.path.dirname(os.path.realpath(__file__))

while 'src' not in os.listdir(current):
    current = Path(current).parent

PARENT_DIR = current

sys.path.append(str(current))
sys.path.append(os.path.join(current, 'src'))

from src.visual_system.scene.data_preparation import augmentation as aug
import src.utilities.directories_and_files as dirf

if __name__ == '__main__':
    all_train_data = os.path.join(PARENT_DIR, 'src', 'visual_system', 'scene/labeled_data') 
    train_aug_dir = os.path.join(PARENT_DIR, 'src', 'visual_system', 'scene/labeled_data_extended')
    os.makedirs(train_aug_dir, exist_ok=True)


    for cls_name in os.listdir(all_train_data):
        # quantize the validation split
        aug.data_quantization(
                        dataset_directory=os.path.join(all_train_data, cls_name), 
                        output_directory=os.path.join(train_aug_dir, cls_name), 
                        batch_size=64, 
                        debug=False, 
                        num_clusters=32)
        
        aug.human_data_augmentation(
                                dataset_directory=os.path.join(all_train_data, cls_name), 
                                output_directory=os.path.join(train_aug_dir, cls_name), 
                                batch_size=32, 
                                debug=False)

        dirf.copy_directories(
                            src_dir=os.path.join(all_train_data, cls_name), 
                            des_dir=os.path.join(train_aug_dir, cls_name), 
                            copy=True)

