"""_summary_
"""

import numpy as np
import torch
import cv2 as cv

from typing import Union, Dict, List, Tuple
from pathlib import Path

from src.visual_system.face.localFR import FaceRecognizer
from src.visual_system.face.face_extractor import FaceExtractor
from src.visual_system.face.utilities import CONFIDENCE_THRESHOLD

from src.visual_system.visual_frame import VisualFrame
from src.visual_system.emotions.EmotionClassifier import EmotionClassifier
from src.visual_system.scene.SceneClassifier import SceneClassifier
from _collections_abc import Sequence




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


class VisualSystem:
    def __init__(self,
                reference_embeddings: Union[str, Path, Dict[str, np.ndarray]],
                top_persons_detected: int = 5,
                top_faces_detected: int = 2,
                yolo_path: Union[str, Path] = 'yolov8s.pt') -> None:
        # first create the face recognizer
        self.face_recognizer = FaceRecognizer(reference_embeddings=reference_embeddings, 
                                    top_persons_detected=top_persons_detected, 
                                    top_faces_detected=top_faces_detected,
                                    yolo_path=yolo_path)
        
        self.face_extractor = FaceExtractor()

#        self.emotion_classifier = EmotionClassifier()
        self.scene_classifier = SceneClassifier()

    def see(self,
            frames: Sequence[Union[Path, str, np.ndarray, torch.Tensor]],
            frame_cuts: Dict[Union[str, int], int], 
            debug: bool = False) -> List[VisualFrame]:
        # the first step is to determine the characters in the frames   
        frames_signatures, person_ids = self.face_recognizer.recognize_faces(frames=frames, 
                                                                             frame_cuts=frame_cuts, 
                                                                             debug=debug, 
                                                                             display=debug)

        characters = [[person_ids[i] for i in ids] for ids, _, _ in frames_signatures]

        emotions = []

        # the next step is to crop the images using the FaceExtractor class
        # for frame_index, (ids, boxes, _) in enumerate(frames_signatures):
        #     if len(ids) == 0:
        #         emotions.append([])        
        #         continue

        #     frame_np = cv.imread(frames[frame_index])

        #     people = [crop_image(frame_np, bb_coordinates=box, debug=debug, title=f'frame: {frame_index}, id: {i}') for 
        #                 i, box in zip(ids, boxes)]
                                
        #     # keep_all=False since there should be only one face in a 'person bounding box' 
        #     face_boxes, face_probs = self.face_extractor.extract_bboxes(images=people, keep_all=False, return_probs=True)
        #     # certain person boxes will be assigned a None value (the model cannot detect face within the cropped bounding box)

        #     people_and_boxes =  [(im, [b[1], b[3], b[0], b[2]]) if (b is not None and p >= CONFIDENCE_THRESHOLD) else (None, None)
        #                          for b, p, im in zip(face_boxes, face_probs, people) ]

        #     face_indices = [i for i in range(len(people_and_boxes)) if people_and_boxes[i][0] is not None]

        #     faces = [crop_image(im, bb_coordinates=box, debug=debug, title=f'frame: {frame_index}') for im, box in people_and_boxes if im is not None]
            
        #     # now I need to append the None values 
        #     temp_emotions = [None] * len(faces) # self.emotion_classifier.classify(faces)
        #     frame_emotions = [None for _ in range(len(people))]
            
        #     for i, e in enumerate(temp_emotions):
        #         frame_emotions[face_indices[i]] = e 

        #     emotions.append(frame_emotions)

        scenes = self.scene_classifier.classify(frames)
        emotions = [None] * len(frames_signatures)
        assert len(scenes) == len(emotions) == len(characters) == len(frames_signatures), f"scenes: {len(scenes)}, emotions: {len(emotions)}, characters: {len(characters)}, signatures: {len(frames_signatures)}"

        # wrap up everything in a VisualFrame object. 
        return [VisualFrame(characters=c, bboxes=fs[1], probs=fs[2], emotions=[None] * len(c), scene=s) for c, fs, e, s in zip(characters, frames_signatures, emotions, scenes)]
    

TEMP2 = 1