import os.path
import torch

import numpy as np
import cv2 as cv

from typing import Union, Dict
from pathlib import Path
from _collections_abc import Sequence
from collections import Counter

from src.face.embeddings import build_classes_embeddings, read_embeddings
from src.face.utilities import REPORT_THRESHOLD
from src.face.helper_functions import cosine_similarity
from src.face.video_analyzer import YoloAnalyzer


class FaceMatcher:
    def __init__(self, reference_embeddings: Union[str, Path, Dict[str, np.ndarray]]):
        # the reference embeddings will be initialized depending on the input
        if not isinstance(reference_embeddings, (str, Path, Dict)):
            raise TypeError(f'the reference embeddings are expected either as\n'
                            f'1. a directory with classes\n'
                            f'2. a json file\n'
                            f'3. a ready to use dictionary\n'
                            f'Found: {type(reference_embeddings)}')

        if os.path.isfile(reference_embeddings):
            reference_embeddings = read_embeddings(reference_embeddings)

        elif os.path.isdir(reference_embeddings):
            reference_embeddings = build_classes_embeddings(directory=reference_embeddings, 
                                                            save_faces=os.path.join(reference_embeddings, 'detected_faces'))

        self.reference_embeddings = reference_embeddings

    def match(self,
              face_embeddings: np.ndarray,
              report_threshold: float,
              reference_embeddings: dict[str: np.ndarray] = None) -> Union[list[str], dict[int, str]]:
        """
        Given face embeddings and
        Args:
            face_embeddings: the embeddings of given faces
            reference_embeddings: The embeddings
            report_threshold: the minimum similarity between 2 embeddings for them to be considered legitimate

        Returns: Returns the best match out of the reference embeddings for each given embedding
        (as long as the similarity exceeds the given threshold)
        """
        if reference_embeddings is None:
            reference_embeddings = self.reference_embeddings

        reference_items = list(reference_embeddings.items())
        # the first step is to build a matrix of cos distances between each instance and all the embeddings
        sims = np.asarray(
            [[np.mean(cosine_similarity(fe, ref)) for cls, ref in reference_items]
             for fe in face_embeddings])

        # a couple of assert statements to make sure everything is working correctly under the hood
        assert sims.size == 0 or \
               (len(sims.shape) == 2 and all([isinstance(n, np.number) and np.abs(n) <= 1 for n in sims.flatten()]))
        # find the maximum element on each row: the most similar reference for each given embedding
        initial_matching = np.argmax(sims, axis=1)

        matches = [reference_items[ref_max_index][0] if sims[face_index, ref_max_index] >= report_threshold else None
                   for face_index, ref_max_index in enumerate(initial_matching)]
        return matches


class FaceRecognizer:
    def __init__(self,
                 reference_embeddings: Union[str, Path, Dict[str, np.ndarray]],
                 top_persons_detected: int = 5,
                 top_faces_detected: int = 2,
                 yolo_path: Union[str, Path] = 'yolov8n.pt',
                 ):
        # initialize the face matcher
        self.face_matcher = FaceMatcher(reference_embeddings)
        # initialize the video analyzer
        self.analyzer = YoloAnalyzer(top_persons_detected, top_faces_detected, yolo_path)

    def recognize_faces(self,
                        frames: Sequence[Union[Path, str, np.ndarray, torch.Tensor]],
                        display: bool = True, 
                        debug: bool = True):
        # first analyze the video
        frames_signatures, person_ids = self.analyzer.analyze_frames(frames, debug=debug)
        # for each person_id finds the most probable face
        ids_labels = {}
        for person_id, face_embeddings in person_ids.items():
            id_matches = self.face_matcher.match(face_embeddings, report_threshold=REPORT_THRESHOLD)
            # set the label of the 'id' to the one with the highest occurrence
            ids_labels[person_id] = Counter(id_matches).most_common()[0][0]

        if display:
            self._display(frames, frames_signatures, ids_labels)
        return frames_signatures, ids_labels

    def _display(self, frames, frames_signatures, ids_labels):
        for f, sign in zip(frames, frames_signatures):
            # make sure to convert the frames to a numpy array if needed
            f_np = cv.imread(f) if isinstance(f, (Path, str)) else f 

            boxes, ids = sign
            for b, i in zip(boxes, ids):
                # it might be possible that certain ids present the frames were not associated with a class, 
                # thus these ids won't be present in the ids_labels (it is an iff relation)                
                if i not in ids_labels:
                    continue

                y0, y1, x0, x1 = b
                cv.rectangle(f_np, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 2)
                # add the label
                
                cv.putText(f_np, ids_labels[i], (int(x0), int(y0)), cv.FONT_HERSHEY_SIMPLEX, 1 ,(0.255, 0), 2)
                cv.imshow('frame', f_np)
            cv.waitKey()
            cv.destroyAllWindows()
