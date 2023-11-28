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

# from src.face.custom_tracker import CustomByteTracker
from src.visual_system.tracking.sort import Sort, convert_x_to_bbox
from .utilities import FR_SingletonInitializer


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
               title: str = 'cropped', 
               bb_mode: str = 'xyxy') -> np.ndarray:
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)

    if bb_mode == 'crop':
        y0, y1, x0, x1 = bb_coordinates
    else: 
        x0, y0, x1, y1 = bb_coordinates

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
                         xywh: bool = False) -> List[Tuple[List, List, List]]:
        """Assuming the tracking results applied on a set of frames, this function will return a tuple: 
        ('ids', 'bounding boxes' and 'probabilities') of humans detected in each frame.

        Args:
            tracking_results (List[Results]): The results return by the track() method of the Yolo Model.
            xywh (bool, optional): whether to convert the bounding boxes into x0, x1, y0, y1 format. Defaults to False.

        Returns:
            List[Tuple[List, List, List]]: The signature for each frame saved in a list 
        """

        "This function returns a unique signature for each frame: ([list of bounding boxes], [list of ids], [the confidence of the detections])"

        def signature(tres_obj: Results) -> Tuple[List, List, List]:
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
                # this error mainly rises when the field 'id' is None while boxes is not
                return [], [], []
            
        return [signature(tres_obj) for tres_obj in tracking_results]

    @classmethod
    def _process_sort_output(cls, sort_output: np.ndarray) -> Tuple[List[int], List[List[Tuple[float]]], List[float]]:
        """This accepts the tracker output for a single frame and return the frame signature:
        3 lists: ([ids], [bounding boxes], [bounding box probability])

        Args:
            sort_output (np.ndarray): The output of the SORT tracking algorithm
        """
        def _process_single_output(sort_single: np.ndarray) -> Tuple[int, List[float], float]:
            # this function will take the bounding box of a single 
            # make sure the input is of the expected shape
            if sort_single.shape != (6, ):
                raise ValueError(f"This function expects a single bounding box. Found: {sort_single}")

            # start with 'id', bounding box
            return int(sort_single[-1]), sort_single[:-2].tolist(), sort_single[-2]

        if sort_output.size == 0:
            return [], [], []

        if sort_output.ndim == 1:
            sort_output = np.expand_dims(sort_output, axis=0)

        if sort_output.shape[1] != 6:
            # the output is expected to be numpy array representing the  of each object
            # tracked in a grame
            raise ValueError((f"The tracker's output is expected to be an array representing: bounding boxes, confidence, and id"
                             f" for each frame."))
    
        ids, boxes, probs =  list(map(list, zip(*[_process_single_output(s) for s in sort_output])))             

        return ids, boxes, probs

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
        # when passed in batch the Yolo.track function does is not persistent across the frames inside the batch
        # a custom tracker to solve this problem.
        # self.tracker = CustomByteTracker(os.path.join(self._file_dir(), 'tracker.yaml'))
        
        singleton = FR_SingletonInitializer()
        self.face_detector = singleton.get_face_detector()
        self.face_encoder = singleton.get_encoder()
        self.device = singleton.get_device()

    def _track_single_cut(self, 
                          frames: Sequence[Union[Path, str, np.ndarray, torch.Tensor]],
                          ) -> List[Results]:
        """Given a sequence of frames (assuming they belong to the same cut / camera shot), this function returns
        the tracking results applied on each frame. 

        Args:
            frames (Sequence[Union[Path, str, np.ndarray, torch.Tensor]]): 
            A sequence of frames, either as paths, numpy arrays or tensors

        Returns:
            List[Results]: Each frame is associated with a Results object summarizing the frame's main results.
        """

        tracker = Sort()
        
        # pass the frame to yolo
        detection_results = self.yolo.predict(source=frames,
                                              classes=[0],
                                              device=self.device, 
                                              show=False,
                                              verbose=False, 
                                              conf=0.4)
        tracking_results = []

        box_to_prob = {}

        for res in detection_results:
            # consider the case where there no boxes are detected
            if res.boxes is None:
                tracker_input =  np.empty((0, 5))
            else:
                boxes = res.boxes.xyxy.cpu().numpy()
                probs = res.boxes.conf.cpu().numpy()
                if probs.ndim == 1:
                    probs = np.expand_dims(probs, axis=1)
                # concatenate both of them
                tracker_input = np.concatenate([boxes, probs], axis=1)
            
            # update the tracker
            tracked_objs, org_boxes = tracker.update(tracker_input)

            if len(tracked_objs) != len(org_boxes):
                raise ValueError(f"Make sure to return the original boxes only for the tracked boxes")

            # # it is important to keep in mind that the tracker does not return the probabilities
            # box_to_prob.update({tuple(b.astype(np.int32).squeeze()): i for i, b in enumerate(boxes)})
            
            # # build the final tracking result object
            # probs_tracked = np.asarray(
            #                             [probs[box_to_prob[tuple(t_obj[:4].astype(np.int32))]] for t_obj in org_boxes]
            #                         )
            
            # # expand the probabilities (if needed) to concatenate them with the tracking results 
            # if probs_tracked.ndim == 1:
            #     probs_tracked = np.expand_dims(probs_tracked, axis=1)
            try:
                boxes = np.concatenate([ob.reshape(1, -1) if ob.ndim == 1 else ob for ob in org_boxes])            
                frame_signature = np.concatenate([boxes, tracked_objs[range(0, len(tracked_objs)), -1].reshape(-1, 1)], axis=1)
                # add the probs to the tracking result
                tracking_results.append(frame_signature)
            except ValueError as e:
                print(e)
                tracking_results.append(np.empty((0, 6)))

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
        # initialize a dictionary that map the cut number of the frames belong to that cut.
        cut_frames = defaultdict(lambda : [])
        
        if isinstance(frames, (np.ndarray, torch.Tensor)):
            for index, f in enumerate(frames):
                cut_frames[frame_cuts[index]].append(f)
        else:
            for f in frames:
                cut_frames[frame_cuts[f]].append(f)
        
        # make sure the frames are sorted 
        if list(cut_frames.keys()) != sorted(cut_frames.keys()):
            raise ValueError(f"Make sure that the 'frames' are sorted chronologically") 

        # NOTE: the self._track_single_cut method will return ids starting from '1'
        # since this function is called for each cut, the ids will be repeated. Adding the largest id (so far)
        # will ensure id uniqueness.
        final_result = []
        largest_id = 0

        for cut_number, cf in cut_frames.items():
            # get the tracking results for the frames of a given cut.
            cut_track_results = self._track_single_cut(cf)
            # convert the results into frame signatures
            # cut_fs = self.frame_signature(cut_track_results)
            cut_fs = [self._process_sort_output(sort_output=cut_frame_res) for cut_frame_res in cut_track_results]

            # the final step is to add the value of the largest_id to each id in the frame signatures
            cut_fs = [([i + largest_id for i in ids], b, p) for ids, b, p in cut_fs]
            # extract the currently large id
            seq = [max(s[0]) for s in cut_fs if len(s[0]) > 0]
            if len(seq) != 0:
                largest_id = max(seq) # the condition is added since 'max' raise errors with empty lists.
            # save the results
            final_result.extend(cut_fs)

        # before returning the results, a final safety check: The number of results matches that of the frames
        if len(final_result) != len(frames):
            raise ValueError((f"The number of final frame signatures do not match the number of passed frames.\n"
                             f"frames {len(frames)}, signatures: {len(final_result)}"))
        
        return final_result


    # def _track(self, 
    #            frames: Sequence[Union[Path, str, np.ndarray, torch.Tensor]],
    #            frame_cuts: Dict[Union[str, int], int]) -> List[Results]:
    #     """This function tracks the different people detected across the given sequence of frames

    #     Args:
    #         frames (Sequence[Union[Path, str, np.ndarray, torch.Tensor]]): a sequence of frames. The assumption is
    #         that frames are consecutive in time.
    #     Returns:
    #         List[Results]: a list of Yolo Results objects
    #     """
    #     # initialize a dictionary that map the cut number of the frames belong to that cut.
    #     cut_frames = defaultdict(lambda : [])
    #     if isinstance(frames, (np.ndarray, torch.Tensor)):
    #         for index, f in enumerate(frames):
    #             cut_frames[frame_cuts[index]].append(f)
    #     else:
    #         for f in frames:
    #             cut_frames[frame_cuts[f]].append(f)
        
    #     # make sure the frames are sorted 
    #     if list(cut_frames.keys()) != sorted(cut_frames.keys()):
    #         raise ValueError(f"Make sure that the 'frames' are sorted chronologically") 

    #     # NOTE: the self._track_single_cut method will return ids starting from '1'
    #     # since this function is called for each cut, the ids will be repeated. Adding the largest id (so far)
    #     # will ensure id uniqueness.
    #     final_result = []
    #     largest_id = 0

    #     for cut_number, cf in cut_frames.items():
    #         # get the tracking results for the frames of a given cut.
    #         cut_track_results = self._track_single_cut(cf)
    #         # convert the results into frame signatures
    #         cut_fs = self.frame_signature(cut_track_results)
    #         # the final step is to add the value of the largest_id to each id in the frame signatures
    #         cut_fs = [([i + largest_id for i in ids], b, p) for ids, b, p in cut_fs]
    #         # extract the currently large id
    #         largest_id = max([max(s[0]) for s in cut_fs if len(s[0]) > 0]) # the condition is added since 'max' raise errors with empty lists.
    #         # save the results
    #         final_result.extend(cut_fs)

    #     # before returning the results, a final safety check: The number of results matches that of the frames
    #     if len(final_result) != len(frames):
    #         raise ValueError((f"The number of final frame signatures do not match the number of passed frames.\n"
    #                          f"frames {len(frames)}, signatures: {len(final_result)}"))
        
    #     return final_result

    def _identify(self, frame_signs: List[Tuple[List[float], List[int], List[float]]]) -> defaultdict[int: list]:
        """Given a the frame signatures, this function simply groups the 'ids' and selects the 
        'self.top_persons_detected' with the highest confidence and return 
            1. list of frames where this the id appears
            2. list of bounding boxes associated with the id
            3. The confidence of the bounding box
 
        Args:
            frame_signs (List[Tuple[List[float], List[int], List[float]]]): The signature of each frame.

        Returns:
            defaultdict: a map between track ids and the fom
        """
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

    def _detect_encode(self, 
                       frames: Sequence[Union[Path, str, np.ndarray, torch.Tensor]],
                       person_dict: Dict[int, List],
                       crop_person: bool = True,
                       debug: bool = False) -> Dict[int, np.ndarray]:
        """This function will simply map each id to 'self.top_faces_detected' faces with the highest confidence 
        saved as numpy arrays.

        Args:
            frames: a sequence of frames
            person_dict (Dict[int, List]): a dictionary that map 'ids' to their corresponding information.
            crop_person (bool, optional): whether to pass crop the input to the face detector or not . Defaults to True.
            debug (bool, optional): whether to display the cropped images. Defaults to False.

        Returns:
            Dict[int, np.ndarray]: the map between the 'id' and the face representations. 
        """

        if crop_person:
            frames = images_to_numpy(frames)

        # detect only the most probable face, as the input will be cropped to contain only one image
        self.face_detector.keep_all = False

        # the function crop_image will return a numpy array. The images will be of different dimensions.
        id_cropped_images = dict([(person_id, [crop_image(frames[frame_index], bb, debug=debug, title=f'id: {person_id}, frame: {frame_index}', bb_mode='xyxy')
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
            
            # account for the fact that certain 'ids' are associated with people but with blurry / not visible faces. 
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
                       debug: bool = False) -> Tuple[List, Dict[int, str]]:
        """This function returns the signature of each frame: ['ids', 'boxes', 'probs'] , groups the ids
        and associate each ids with the most probable / likely face embeddings (to be later passed to the face matching mechanism.) 

        Args:
            frames (Sequence[Union[Path, str, np.ndarray, torch.Tensor]]): a sequence of frames
            frame_cuts (Dict[Union[str, int], int]): a map between the frame and the cut it belongs to
            debug (bool, optional): whether to display the cropped images or not. Defaults to False.

        Returns:
            Tuple[List, Dict[int, str]]: The signature of each frame, and the mapping between the ids and their face representations.
        """


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
        # group each 'person' identified in the picture
        person_ids = self._identify(frame_signs=frame_signs)
        # time to encode each person ('id') detected by YOLO
        face_ids = self._detect_encode(frames, person_ids, debug=debug)

        return frame_signs, face_ids
