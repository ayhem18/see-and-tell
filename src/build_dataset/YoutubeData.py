"""
This script contains the functionalities needed to build a small dataset   
"""


import os
import cv2 as cv
import shutil
import re
import numpy as np

import src.utilities.directories_and_files as dirf

from typing import Union, Iterable
from pathlib import Path
from pytube import YouTube, Playlist
from pytube.exceptions import AgeRestrictedError

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def video_to_images(video_path: Union[str, Path], 
                    output_dir: Union[str, Path],
                    fps: int = None,
                    frame_stride: Union[int, Iterable[int]] = None,
                    new_output_dir: bool = False, 
                    ) -> Path:
    """This function, given a path to a video file, converts it into a number of frames.

    Args:
        video_path (Union[str, Path]): The path to a video
        output_dir (Union[str, Path]): The folderw where the resulting frames should be saved
        fps (int, optional): frames per second. Defaults to None.
        frame_stride (Union[int, Iterable[int]], optional): _description_. Defaults to None.
        new_output_dir (bool, optional): where to delete the output_dir if it already exists. Defaults to False.

    Returns:
        Path: The path to the output directory
    """
    if os.path.isdir(output_dir) and new_output_dir:
        # remove the directory if it already exists
        shutil.rmtree(output_dir)

    output_dir = dirf.process_save_path(output_dir, file_ok=False, dir_ok=True) 

    # calculate the number of frames already in the 'output_dir'
    num_frames = len(os.listdir(output_dir))

    # first make sure at least 'fps' or 'frame_stride' arguments are passed
    if fps is None and frame_stride is None:
        raise TypeError(f"The function expects at least 'fps' or 'frame_stride' arguments to be non-None") 
    
    # if fps is None, then we will count the total number of frames in the video, then use frame_stride to save the frames
    if fps is None:
        if not isinstance(frame_stride, int):
            raise TypeError(f"if 'fps' is None, frame_stride is expected to be  single value.")
    
        count = 0
        video_iterator = cv.VideoCapture(video_path)

        # count the total number of frames in the video
        s = True
        total_count = 0
        
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
                num_frames += 1
                frame_str = f"frame_{num_frames}.jpg"
                cv.imwrite(os.path.join(output_dir, frame_str, image))

        return output_dir

    # assuming 'fps' is given, then we have more flexibility choosing the frames to save
    frame_stride = [frame_stride] if isinstance(frame_stride, int) else frame_stride
    video_iterator = cv.VideoCapture(video_path)
    count = 0

    while True:
        frame_exists, image = video_iterator.read()

        if not frame_exists: 
            # the video is over
            break

        count += 1
        if count % fps in frame_stride:
            num_frames += 1
            frame_str = f"frame_{num_frames}.jpg"
            cv.imwrite(os.path.join(output_dir, frame_str), image)

    return output_dir


def _process_video_name(file_name: str) -> str:
    # first remove the file system delimiter from the name
    sep = os.path.sep
    file_name = re.sub(sep, '_', file_name)
    return re.sub(r'\s+', '_', file_name.strip().lower())


def download_yt_video(video: Union[str, YouTube], 
                    output_dir: Union[Path, str]=None, 
                    file_name: str = None,
                    min_resolution: int = 480, 
                    only_video: bool = True, 
                    return_fps: bool = True,
                    ) -> Union[str, None]:
    
    # allow authentication to bypass any age restricted videos
    try: 
        yt = video if isinstance(video, YouTube) else YouTube(video, use_oauth=True,allow_oauth_cache=True)
    except AgeRestrictedError as e:
        print("Age restrictions !! ")
        return None
    except Exception as e:
        print(e)
        return None

    streams = yt.streams.filter(only_video=only_video, 
                      custom_filter_functions=
                      [lambda video_obj: int(video_obj.resolution[:video_obj.resolution.index('p')]) >= min_resolution])
    
    # account for the fact that none of the videos are available for a minimum required resolution
    if len(streams) == 0:
        if return_fps:
            return None, None
        return None
    
    # sort by resolution
    streams = sorted(streams, key=lambda x: int(x.resolution[:x.resolution.index('p')]), reverse=True)
    
    s = streams[0]
    output_dir = SCRIPT_DIR if output_dir is None else output_dir
    file_name = s.title if file_name is None else file_name
    # add the extension 
    if not file_name.endswith('.mp4'):
        file_name += '.mp4'
    # make sure to process the file_name: strip, lower_case and replace spaces with '_'
    file_name = _process_video_name(file_name)

    s.download(output_path=output_dir, filename=file_name, max_retries=5)
    
    if return_fps:
        return os.path.join(output_dir, file_name), s.fps
    
    return os.path.join(output_dir, file_name)


def youtube_playlist_dataset(playlist_url: str, 
                             output_dir: Union[Path, str], 
                             save_in_one_folder: bool = True, 
                             num_videos: int = None):
    
    p = Playlist(playlist_url)
    videos = p.videos[:num_videos] if num_videos is not None else p.videos
    video_urls = p.video_urls[:num_videos] if num_videos is not None else p.video_urls

    output_dir = dirf.process_save_path(output_dir)


    if 'video_urls.txt' not in os.listdir(output_dir):
        with open(os.path.join(output_dir, 'video_urls.txt'), 'w') as file: 
            pass


    existing_urls = set()   

    with open(os.path.join(output_dir, 'video_urls.txt'), 'r') as file: 
        for line in file.readlines():
            existing_urls.add(line[:-1])


    for index, (v, v_url) in enumerate(zip(videos, video_urls), start=1):
        # first check if this url has already been downloaded before        
        if v_url not in existing_urls:
            with open(os.path.join(output_dir, 'video_urls.txt'), 'a') as file:
                file.write(v_url + "\n") 
        else: 
            continue

        output_dir = output_dir if save_in_one_folder else os.path.join(output_dir, f'video_{index}')

        video_path, fps = download_yt_video(v_url, 
                          output_dir=output_dir, 
                          return_fps=True
                          )
        
        if video_path is None:
            continue

        # it is better to cut the video into frames at this point as it might prevent overloading the Youtube API leading 
        # to better overall performance             

        frame_stride = [0, fps // 2]

        video_to_images(video_path=video_path, 
                        output_dir=output_dir, 
                        fps=fps, 
                        frame_stride=frame_stride)

    