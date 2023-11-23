"""
This script contains a wrapper around the face detection /  extraction functions offered by the facenet-pytorch library.
"""

import torch
import numpy as np 

from pathlib import Path
from typing import Union, Sequence, List, Tuple
from PIL import Image

from src.face.utilities import CONFIDENCE_THRESHOLD, REPORT_THRESHOLD, FR_SingletonInitializer



class FaceExtractor():  
    def __init__(self) -> None:
        singleton = FR_SingletonInitializer()
        # access the face detector and the device
        self.face_detector = singleton.get_face_detector()
        self.device = singleton.get_device()

    def _batch_preprocess(images: Sequence[Union[str, Path, np.ndarray, torch.Tensor]]) -> np.ndarray:
        if isinstance(images, (str, Path)):
            images = [images]

        elif isinstance(images, Sequence) and isinstance(images[0], (str, Path)):
            images = np.asarray([np.array(Image.open(f)) for f in images])

        elif isinstance(images, torch.Tensor):
            # the tensor should be permuted so that the number of channels is placed first
            images = torch.unsqueeze(images, dim=0) if images.ndim == 3 else images
            images = images.permute((0, 2, 3, 1)) if images.size(dim=1) in [1, 3] else images

        if isinstance(images, np.ndarray):
            images = images if images.ndim == 4 else np.expand_dims(images, 0)

        return images

    def extract_faces(self, 
                      images: Sequence[Union[str, Path, np.ndarray, torch.Tensor]], 
                      keep_all: bool = True,
                      return_probs: bool = True, 
                      return_tensor: str = 'numpy') -> Union[torch.Tensor, tuple(torch.tensor, float)]:
        self.face_detector.keep_all = keep_all
        # convert images to a numpy array so it can be passed as a single batch to the face detection model 
        images = self._batch_preprocess(images)
        # after this preprocessing, the data should be passed as a single batch to the model
        output, probs = self.face_detector.forward(images, return_prob=True)
        # convert the output to the correct type.
        output = output.detach().cpu().numpy() if return_tensor == 'numpy' else output.detach()
        
        if return_probs:
            return output, probs

        return output 
    

    def extract_bboxes(self, 
                      images: Sequence[Union[str, Path, np.ndarray, torch.Tensor]], 
                      keep_all: bool = True,
                      return_probs: bool = True, 
                      return_tensor: str = 'numpy') -> Tuple[torch.Tensor, List[List[float]]]:
        self.face_detector = keep_all        
        images = self._batch_preprocess()
        output, probs = self.face_detector.detect(images)
        output = output.detach().cpu().numpy() if return_tensor == 'numpy' else output.detach()
        if return_probs:
            return output, probs
        
        return output 

    def extract_all(self,
                    images: Sequence[Union[str, Path, np.ndarray, torch.Tensor]], 
                    keep_all: bool = True,
                    return_probs: bool = True, 
                    return_tensor: str = 'numpy'):
        raise NotImplementedError("This function might be needed in a future release !!")
