import os
import torch
import yaml

from argparse import Namespace
from ultralytics.trackers import BYTETracker, BOTSORT
# import the type of the Yolo's outputs
from ultralytics.engine.results import Results

from typing import Union, List
from pathlib import Path


class CustomByteTracker():
    def __init__(self, configuration_path: Union[str, Path]) -> None:
        # make sure the configuration has the .yaml extension
        _, file_extension = os.path.splitext(configuration_path)
        if file_extension != '.yaml':
            raise ValueError("The class expects a 'yaml' file.\nFound an {file_extension} file")

        # cfg : dict= yaml.load(open(os.path.join(os.path.dirname(__file__), 'tracker.yaml'), 'r'), Loader=yaml.Loader)
        self.configuration_path = configuration_path
        # create the configuration object and load the tracker
        cfg: dict = yaml.load(open(self.configuration_path, 'r'), Loader=yaml.Loader)
        # remove the 'tracker_type' field to avoid compatibility issues
        cfg.pop('tracker_type')
        self.configuration = cfg
        self.tracker = BYTETracker(args=Namespace(**cfg))


    def track(self, yolo_results: List[Results]):
        """Embeds tracking information into the results of a YOLOv8 model.
        
        Instantiates a tracker and uses it to track the objects in the video. It
        strictly follows the code provided in `ultralytics` package.

        Ref: https://github.com/ultralytics/ultralytics/blob/3ae81ee9d11c432189e109d7a1724635a2e451ca/ultralytics/tracker/track.py#LL39C43-L39C43 
        
        The function is intended to be used in the case detection results
        comes not instantly but in batches. In this case, results of YOLO detection
        are collected, combined and only then passed to the tracker. 

        Args:
            yolo_results: 
                The product of `YOLO.predict` call or concantenation of such.

        Returns:
            The same list of results with tracking information embedded. 
        """

        # creating a stand-alone tracker for each run so that the results do not overlap
        tracker = BYTETracker(args=Namespace(**self.configuration))
        for index, res in enumerate(yolo_results):
            det = res.boxes.cpu().numpy()
            if len(det) == 0:
                continue    
            
            tracks = tracker.update(det, res.orig_img)
            if len(tracks) == 0:
                continue

            idx = tracks[:, -1].astype(int)
            yolo_results[index] = yolo_results[index][idx]
            yolo_results[index].update(boxes=torch.as_tensor(tracks[:, :-1]))
        
        return yolo_results



