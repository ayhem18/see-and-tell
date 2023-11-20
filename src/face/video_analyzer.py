import os
import requests
import torch
import time
import numpy as np
import cv2 as cv
import itertools

from pathlib import Path
from typing import Union, List, Dict, Tuple
from _collections_abc import Sequence
from collections import defaultdict
from ultralytics import YOLO    
from ultralytics.engine.results import Results
from PIL import Image

from src.face.custom_tracker import CustomByteTracker
from face.utilities import FR_SingletonInitializer


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def images_to_numpy(images: Sequence[Union[str, Path]]) -> List[np.ndarray]:
    if isinstance(images, (str, Path)):
        images = [images]
    return [np.asarray(cv.imread(img)) for img in images]


def xywh_converter(xywh: Sequence[int, int, int, int]) -> \
        Tuple[int, int, int, int]:
    # the idea is to accept a xywh bounding box and return 4 coordinates that can directly be
    # used for cropping
    x_center, y_center, w, h = xywh
    return y_center - h // 2, y_center + h // 2, x_center - w // 2, x_center + w // 2


def crop_image(image: np.ndarray,
               bb_coordinates: Tuple[int, int, int, int],
               debug: bool = False,
               title: str = 'cropped') -> np.ndarray:
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
    y0, y1, x0, x1 = bb_coordinates
    # make sure to convert the coordinates to 'int' if needed
    y0, y1, x0, x1 = int(y0), int(y1), int(x0), int(x1)
    cropped_image = image[y0: y1, x0: x1, :]
    if debug:
        cv.imshow('original', image)
        cv.imshow(title, cropped_image)
        cv.waitKey(0)
        cv.destroyAllWindows()
    return cropped_image


class YoloAnalyzer(object):
    _tracker_URL = 'https://raw.githubusercontent.com/voyager-108/ml/main/apps/nn/yolov8/trackers/tracker.yaml'

    @classmethod
    def _file_dir(cls):
        try:
            return os.path.realpath(os.path.dirname(__file__))
        except NameError:
            return os.getcwd()

    @classmethod
    def _tracker_file(cls):
        # download the tracker file as needed
        req = requests.get(cls._tracker_URL, allow_redirects=True)
        with open(os.path.join(cls._file_dir(), 'tracker.yaml'), 'wb') as f:
            f.write(req.content)

    @classmethod
    def frame_signature(cls,
                         tracking_results: List[Results],
                         xywh: bool = False):
        """
        This function returns a unique signature for each frame: ([list of bounding boxes], [list of ids], [the confidence of the detections])
        Args:
            tracking_results: This a list of Results objects
            xywh: whether to return the bounding box in the xywh format or not

        Returns:
        
        """

        def signature(tres_obj: Results) -> Tuple[List, List]:
            if tres_obj.boxes is None :
                return [], [], []
            try:
                boxes = [b.xywh.squeeze().cpu().tolist() if xywh else xywh_converter(b.xywh.squeeze().cpu().tolist()) 
                        for b in tres_obj.boxes.cpu()]
                
                # the ids for whatever reason are wrapped in tensors
                ids = [b.id.int().cpu().item() for b in tres_obj.boxes.cpu()]
                probs = [(p.cpu().item() if isinstance(p, torch.Tensor) else p) for p in tres_obj.boxes.conf.cpu()]

                return ids, boxes, probs
                
            except AttributeError:
                return [], [], []
            

        return [signature(tres_obj) for tres_obj in tracking_results]

    def __init__(self,
                 top_persons_detected: int = 5,
                 top_faces_detected: int = 2,
                 yolo_path: Union[str, Path] = 'yolov8s.pt',
                 ) -> None:
        self.top_persons_detected = top_persons_detected
        # the top_faces_detected is preferably odd
        self.top_faces_detected = top_faces_detected + int(top_faces_detected % 2 == 0)

        # download the 'tracker.yaml' file if needed
        if not os.path.isfile(os.path.join(self._file_dir(), 'tracker.yaml')):
            # download the file in this case
            self._tracker_file()

        # the yolo components
        self.yolo = YOLO(yolo_path)

        self.tracker = CustomByteTracker(os.path.join(self._file_dir(), 'tracker.yaml'))

        singleton = FR_SingletonInitializer()
        self.face_detector = singleton.get_face_detector()
        self.face_encoder = singleton.get_encoder()
        self.device = singleton.get_device()

    def _track_single_cut(self, 
                          frames: Sequence[Union[Path, str, np.ndarray, torch.Tensor]],
                          ):
        """This function tracks the different people detected across the given sequence of frames

        Args:
            frames (Sequence[Union[Path, str, np.ndarray, torch.Tensor]]): a sequence of frames. The assumption is
            that frames are consecutive in time.

        Returns:
            List[Results]: a list of Yolo Results objects
        """

        tracking_results = self.yolo.track(source=frames,
                            persist=True,
                            classes=[0],  # only detect people in the image
                            device=self.device,
                            show=False, 
                            tracker='bytetrack.yaml',
                            verbose=False,)

        self.tracker.track(tracking_results)
        
        return tracking_results


    def _track(self, 
               frames: Sequence[Union[Path, str, np.ndarray, torch.Tensor]],
               frame_cuts: Dict[Union[str, int], int]) -> List[Results]:

        """This function tracks the different people detected across the given sequence of frames

        Args:
            frames (Sequence[Union[Path, str, np.ndarray, torch.Tensor]]): a sequence of frames. The assumption is
            that frames are consecutive in time.

        Returns:
            List[Results]: a list of Yolo Results objects
        """

        cut_frames = defaultdict(lambda : [])
        # this is the first step in the face detection pipeline: Detecting people in the image + tracking them
        if isinstance(frames, (np.ndarray, torch.Tensor)):
            for index, f in enumerate(frames):
                cut_frames[frame_cuts[index]].append(f)
        else:
            for f in frames:
                cut_frames[frame_cuts[f]].append(f)

        # make sure the cuts are sorted
        if list(cut_frames.keys()) != sorted(cut_frames.keys()):
            raise ValueError(f"Make sure that the 'frames' are sorted chronologically") 

        final_result = []
        largest_id = 0

        for cut_number, cf in cut_frames.items():
            cut_track_results = self._track_single_cut(cf)
            cut_frame_signs = self.frame_signature(cut_track_results)
            # the final step is to add the value of the largest_id to each id in the frame signatures
            cut_frame_signs = [([i + largest_id for i in ids], b, p) for ids, b, p in cut_frame_signs if len(ids) > 0]
            largest_id = max([max(s[0]) for s in cut_frame_signs])
            final_result.extend(cut_frame_signs)

        return final_result

    def _identify(self, frame_signs: List[Tuple[List[float], List[int], List[float]]]):
        ids_dict = defaultdict(lambda: [])

        for frame_index, sign in enumerate(frame_signs):
            ids, boxes, probs = sign
            assert len(ids) == len(boxes) == len(probs), "Check the lengths of ids, probabilities and boxes"

            for i, bb, p in zip(ids, boxes, probs):
                # convert the xywh bounding box to coordinates that can be used directly for image cropping
                ids_dict[i].append((frame_index, bb, p))

        # the final step is to filter the results
        # keep only the top self.person_detected boxes for each id
        for person_id, info in ids_dict.items():
            ids_dict[person_id] = sorted(info, key=lambda x: x[-1], reverse=True)[:self.top_persons_detected]

        return ids_dict

    # def _identify(self, tracking_results: List[Results]):
    #     # create a dictionary to save the information about each id detected in the results
    #     # the dictionary will be of the form {id: [(frame_index, boxes, probs)]}
    #     ids_dict = defaultdict(lambda: [])

    #     # iterate through the results to extract the ids
    #     for frame_index, results in enumerate(tracking_results):

    #         boxes = results.boxes

    #         if boxes is None:
    #             continue
            
    #         ids = boxes.id.int().cpu().tolist()
    #         probs = boxes.conf


    #         assert len(ids) == len(boxes) == len(probs), "Check the lengths of ids, probabilities and boxes"

    #         for i, bb, p in zip(ids, boxes.xywh.cpu().tolist(), probs):
    #             # convert the xywh bounding box to coordinates that can be used directly for image cropping
    #             bounding_coordinates = xywh_converter(bb)
    #             ids_dict[i].append((frame_index, bounding_coordinates, (p.cpu().item() if isinstance(p, torch.Tensor) else p)))

    #     # the final step is to filter the results
    #     # keep only the top self.person_detected boxes for each id
    #     for person_id, info in ids_dict.items():
    #         ids_dict[person_id] = sorted(info, key=lambda x: x[-1], reverse=True)[:self.top_persons_detected]

    #     return ids_dict

    def _detect_encode(self, 
                       frames,
                       person_dict: Dict[int, List],
                       crop_person: bool = True,
                       debug: bool = False) -> Dict[int, List[np.ndarray]]:

        if crop_person:
            frames = images_to_numpy(frames)

        # detect only the most probable face, as the input will be cropped to contain only one image
        self.face_detector.keep_all = False
        # the function crop_image will return a numpy array. The images will be of different dimensions.
        # the facenet-pytorch implementation only accepts batched input as:
        # either a list of PIL.Image files or torch.tensors

        id_cropped_images = dict([(person_id, [crop_image(frames[frame_index], bb, debug=debug, title=f'id: {person_id}, frame: {frame_index}')
                                               for frame_index, bb, _ in person_info])
                                  for person_id, person_info in person_dict.items()])
        
        # for each id get the faces in each of the cropped images
        id_faces = dict([(person_id, [self.face_detector.forward(c_im, return_prob=True) for c_im in cropped_images])
                         for person_id, cropped_images in id_cropped_images.items()])

        result = {}

        # for each id keep only the best self.person_detect images
        for person_id, face_prob in id_faces.items():
            # even though the images as those of humans (as detected by Yolo), they may not necessarily
            # include faces. the detector will return None of such input and this should be filtered
            sorted_output = sorted([output for output in face_prob if output[0] is not None], 
                                   key=lambda x: x[-1], reverse=True)[:self.top_faces_detected]
            
            # account for the fact that certain persons in the given images do not have their face shown 
            if len(sorted_output) == 0:
                continue

            # make sure to extract only the faces, (leave out the probabilities), stack the face tensors
            # and move them to the device used across the entire codebase. 
            faces = torch.stack([f for f, prob in sorted_output]).to(self.device)

            result[person_id] = self.face_encoder(faces).detach().cpu().numpy() if len(faces) != 0 else []

        return result

    def analyze_frames(self,
                       frames: Sequence[Union[Path, str, np.ndarray, torch.Tensor]],
                       frame_cuts: Dict[Union[str, int], int], 
                       debug: bool = False):
        # first make sure the frame_cuts variable satisfies certain requirements
        if isinstance(frames, (np.ndarray, torch.Tensor)):
            # if the frames are already in tabular form
            if set(frame_cuts.keys()) != set(range(len(frames))):
                raise ValueError((f"if the frames are passed {torch.Tensor} or {np.ndarray}. Make sure the range of keys "
                                  f"matches the number of elements"))
        else:
            if not isinstance(frames[0], (str, Path)):
                raise TypeError(f"Please make sure the data is either tabular of paths to existing frames")

            # make sure each path is present in the frames_cuts
            for f in frames:
                if f not in frame_cuts:
                    raise ValueError(f"The frame {f} is not associated with a specific cut / shot. Please add it to the 'frame_cuts' argument")

        # now we can proceed
        # the first step is to track the frames
        frame_signs = self._track(frames, frame_cuts=frame_cuts)
        # # create a signature for each frame
        # frame_signatures = self.frames_signature(tracking_results=tracking_results)
        # group each 'person' identified in the picture
        person_ids = self._identify(frame_signs=frame_signs)
        # time to encode each person ('id') detected by YOLO
        face_ids = self._detect_encode(frames, person_ids, debug=debug)

        return frame_signs, face_ids
