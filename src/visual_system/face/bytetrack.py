import os
import torch
import yaml
from argparse import Namespace
from ultralytics.trackers import BYTETracker


def embed_tracking_into_results(yolo_results: list[list], fps=30) -> list[list]:
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
    # Loading configuration from the same directory
    # The configuration follows the one provided in the `ultralytics` package
    cfg : dict= yaml.load(open(os.path.join(os.path.dirname(__file__), 'tracker.yaml'), 'r'), Loader=yaml.Loader)
    cfg.pop('tracker_type') # Compatibility with the YOLO.track

    # BYTETracker accepts an object that
    # provides a way to access its keys as attributes

    # This is a hack to make it work with the Namespace object
    tracker = BYTETracker(args=Namespace(**cfg))

    for i in range(len(yolo_results)):
        # (det)ections
        det = yolo_results[i].boxes.cpu().numpy()
        if len(det) == 0:
            continue
        tracks = tracker.update(det, yolo_results[i].orig_img)
        if len(tracks) == 0:
            continue

        idx = tracks[:, -1].astype(int)
        yolo_results[i] = yolo_results[i][idx]
        yolo_results[i].update(boxes=torch.as_tensor(tracks[:, :-1]))

    return yolo_results



