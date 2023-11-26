"""This script contains the code to download the initial unlabeled data: The train and val splits are built from 2 different Youtube Playlists. 
"""
import os
import sys
from pathlib import Path
from typing import Union

HOME = os.path.dirname(os.path.realpath(__file__))

current = HOME
# find the project directory starting from the file's directory.
while 'src' not in os.listdir(current):
    current = Path(current).parent

sys.path.append(str(current))
sys.path.append(os.path.join(str(current), 'src'))

from src.visual_system.scene.data_preparation import YoutubeData as yd

def initial_data(yt_train_playlist: str,
                 train_output_dir: Union[str, Path], 
                 yt_val_playlist: str, 
                 val_output_dir: Union[str, Path], 
                 num_train_videos: int = 25, 
                 num_val_videos: int = 10):
    # download the training split
    yd.youtube_playlist_dataset(playlist_url=yt_train_playlist, 
                                output_dir=train_output_dir,
                                num_videos=num_train_videos
                                )
    
    # download the validation split
    yd.youtube_playlist_dataset(playlist_url=yt_val_playlist, 
                                output_dir=val_output_dir,
                                num_videos=num_val_videos
                                )

TRAIN_URL = 'https://www.youtube.com/watch?v=Tb627xDlqBs&list=PLdFmbsrJOaOJfuA8fqtpMaBjkB2xKwgLZ'
VAL_URL = 'https://www.youtube.com/watch?v=a14v7JrWOHU&list=PLXWxAAcCuOUQwQPb5vOYVgSI936P5GhgT&index=3'


if __name__ == '__main__':
    train_split_dir = os.path.join(Path(HOME).parent, 'data', 'train')
    val_split_dir = os.path.join(Path(HOME).parent, 'data', 'val')

    initial_data(yt_train_playlist=TRAIN_URL,
                 train_output_dir=train_split_dir, 
                 yt_val_playlist=VAL_URL, 
                 val_output_dir=val_split_dir)
    
