"""
This script contains a number of functionalities used to process and work with video data.
"""

import os
import shutil
import cv2 as cv

from typing import Union, Dict, Optional, List
from pathlib import Path

from scenedetect import open_video, SceneManager, split_video_ffmpeg, detect
from scenedetect.detectors import ContentDetector, AdaptiveDetector
from scenedetect.video_splitter import split_video_ffmpeg

from src.utilities.directories_and_files import process_save_path


def video_to_frames(video_path: Union[str, Path], 
                    output_dir: Union[Path, str], 
                    frame_stride: int = 15, 
                    return_files: bool = False) -> Optional[List[str]]:
    """This function accepts the path to a video, iterates through the video and save 
    the frames in an output directory.

    Args:
        video_path (Union[str, Path]): The path to the video
        output_dir (Union[Path, str]): The folder where the frames will be saved
        frame_stride (int, optional): _description_. Defaults to 15.
    """

    output_dir = process_save_path(output_dir, file_ok=False, dir_ok=True)    
    count = 0
    # create the cv2 capture video object
    video_iterator = cv.VideoCapture(video_path)
    
    # count the total number of frames in the video
    s = True
    total_count = 0
    
    frames = []

    while s:
        s, _ = video_iterator.read()
        total_count += 1
    video_iterator = cv.VideoCapture(video_path)
    
    while True:
        frame_exists, image = video_iterator.read()

        if not frame_exists: 
            # the video is over
            break
        

        count += 1
        if count % frame_stride == frame_stride - 1:
            frame_name = os.path.join(output_dir, f'frame_{len(os.listdir(output_dir)) + 1}.jpg')
            cv.imwrite(frame_name, image)
            frames.append(frame_name)

    if return_files:
        return frames  

def split_video_into_scenes(video_path,
                            output_file_template: str=None,
                            video_name: str = None, 
                            adaptive_threshold=3) -> int:
    if (output_file_template is None) != (video_name is None):
        raise ValueError(f"either both 'video_name' and 'output_file_template are passed or none of them")
    
    scene_list = detect(video_path=video_path, detector=AdaptiveDetector(adaptive_threshold=adaptive_threshold))

    if output_file_template is not None:
        split_video_ffmpeg(video_path, 
                       scene_list,
                       output_file_template=output_file_template,
                       video_name=video_name,
                       show_progress=True)
    else:
        split_video_ffmpeg(video_path, 
                       scene_list,
                       show_progress=True)

    return len(scene_list)


def video_to_cuts(video_path: Union[str, Path], 
                  output_dir: Union[Path, str], 
                  frame_stride: int = 15, ) -> Dict:
    video_dir = Path(video_path).parent
    output_file_template = os.path.join(video_dir, '$VIDEO_NAME _ $SCENE_NUMBER .mp4')    
    # split the video 
    num_cuts = split_video_into_scenes(video_path=video_path, 
                            output_file_template=output_file_template, 
                            video_name='temp_cut')

    frame_cut_map = {}

    for i in range(1, num_cuts + 1):
        # convert the number to the format imposed by the detectscene library
        index_str = f"{'0' * (1 + len(str(num_cuts)) - len(str(i)))}{str(i)} .mp4"


        file_name = os.path.join(video_dir, f'temp_cut _ {index_str}')
        cut_frames = video_to_frames(video_path=file_name, 
                                     output_dir=output_dir, 
                                     frame_stride=frame_stride, 
                                     return_files=True)
        for f in cut_frames:
            frame_cut_map[f] = i
        # delete the video after splitting it into cuts
        os.remove(file_name)
    # {frame -> cut}
    return frame_cut_map