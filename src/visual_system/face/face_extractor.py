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

    def _batch_preprocess(self, 
                          images: Sequence[Union[str, Path, np.ndarray, torch.Tensor]]) -> Union[List, torch.Tensor]:
        
        if isinstance(images, (str, Path)):
            images = [Image.open(images)]

        elif isinstance(images, Sequence) and isinstance(images[0], (str, Path)):
            images = [Image.open(f) for f in images]

        elif isinstance(images, torch.Tensor):
            # the tensor should be permuted so that the number of channels is placed first
            images = torch.unsqueeze(images, dim=0) if images.ndim == 3 else images
            images = images.permute((0, 2, 3, 1)) if images.size(dim=1) in [1, 3] else images

        elif isinstance(images, np.ndarray):
            images = [Image.fromarray(images)] if images.ndim == 3 else [Image.fromarray(i) for i in images]

        return images

    def _process_output(self, detected_faces: torch.Tensor, return_tensor: str = 'numpy') -> Union[torch.Tensor, np.ndarray]:   
        if return_tensor == 'numpy':
            if return_tensor.ndim == 3:
                return detected_faces.detach().cpu().permute(1, 2, 0).numpy()
            return detected_faces.detach().cpu().permute(0, 2, 3, 1).numpy()

        return detected_faces.detach()

    def extract_faces(self, 
                      images: Sequence[Union[str, Path, np.ndarray, torch.Tensor]], 
                      keep_all: bool = True,
                      return_probs: bool = True, 
                      return_tensor: str = 'numpy') -> List[Union[np.ndarray, Tuple[np.ndarray, float]]]:
        """This function will return the faces either as tensors or numpy arrays.
        The extracted faces use a different channel ordering than RGB. So they are most useful 
        for creating the embeddings / encodings by passing them to the encoder.

        THE MODEL WILL RETURN NONE WHEN IT CANNOT DETECT FACES.

        Args:
            images (Sequence[Union[str, Path, np.ndarray, torch.Tensor]]): The images
            keep_all (bool, optional): whether to detect all faces in a single images. Defaults to True.
            return_probs (bool, optional): whether to return the probabilities. Defaults to True.
            return_tensor (str, optional): the return data type. Defaults to 'numpy'.

        Returns:
            List[Union[np.ndarray, Tuple[np.ndarray, float]]]: The faces (+ probabilities) for each image in the given list.
        """

        self.face_detector.keep_all = keep_all
        # convert images to a numpy array so it can be passed as a single batch to the face detection model 
        images = self._batch_preprocess(images)
        # after this preprocessing, the data should be passed as a single batch to the model
        output, probs = self.face_detector.forward(images, return_prob=True)

        # convert to the correct type
        output = [self._process_output(detected_faces=o, return_tensor=return_tensor) if o is not None else o for o in output ]

        if return_probs:
            return output, probs

        return output 
    

    def extract_bboxes(self, 
                      images: Sequence[Union[str, Path, np.ndarray, torch.Tensor]], 
                      keep_all: bool = True,
                      return_probs: bool = True) -> Tuple[np.ndarray, List[List[float]]]:
        
        # so 
        self.face_detector.select_largest = False        
        images = self._batch_preprocess(images)

        # since the 'detect' function does not support batch processing for images with different size
        if any(img.size != images[0].size for img in images) :        
            output, probs = list(map(list, zip(*[self.face_detector.detect(im) for im in images])))
            # convert the result to a list
        else:
            # in this case, all the images are of the same size and it is possible to pass all data as a single batch             
            output, probs = self.face_detector.detect(images)
            # convert to a list: better than an array of arrays

        if not keep_all:
            output = [o[0].tolist() if o is not None else o for o in output ]
            probs = [p[0] if p is not None else p for p in probs]        

            for p in probs:
                if not (isinstance(p, float) or p is None):
                    raise TypeError(f"Make sure to return probabilities directly without a wrapper. Found: {p}")

            for o in output:
                if not ((isinstance(o, list) and len(o) == 4) or o is None):
                    raise ValueError(f"Expected to return the bounding box as a list: Found: {o}")
                
        else:
            output = [(o.tolist() if o is not None else o) for o in output]
            probs = [(p.tolist() if p is not None else p) for p in probs]

        if return_probs:
            return output, probs
        
        return output 

    def extract_all(self,
                    images: Sequence[Union[str, Path, np.ndarray, torch.Tensor]], 
                    keep_all: bool = True,
                    return_probs: bool = True, 
                    return_tensor: str = 'numpy'):
        raise NotImplementedError("This function might be needed in a future release !!")
