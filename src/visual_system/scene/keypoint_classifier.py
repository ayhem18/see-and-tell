"""
This script contains the definition of the Keypoint Scene Classifier
"""

import os, sys

import torch
import math
import numpy as np 
import cv2 as cv
import shutil

from PIL import Image
from typing import Union, List, Tuple, Dict, Optional
from pathlib import Path
from collections import defaultdict
from shutil import copyfile
from _collections_abc import Sequence

current = os.path.dirname(os.path.realpath(__file__))

while 'src' not in os.listdir(current):
    current = Path(current).parent

PARENT_DIR = current

sys.path.append(str(current))
sys.path.append(os.path.join(current, 'src'))


import src.utilities.directories_and_files as dirf
import src.utilities.pytorch_utilities as pu

from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
from lightglue import viz2d



def select_reference_files(data_directory: Union[str, Path],
                           output_directory: Union[str, Path],
                           extractor: SuperPoint,
                           min_kpnt_confidence: float = 0.4, 
                           min_num_kpnts: int = 20, 
                           batch_size: int = 8, 
                           debug: bool = False
                           ) -> None:
    # process the directory path
    data_dir = dirf.process_save_path(save_path=data_directory, file_ok=True, condition=dirf.all_images)
    output_dir = dirf.process_save_path(save_path=output_directory, file_ok=False)

    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]

    for i in range(0, len(files), batch_size):
        
        try:
            # extract the data and save them as a tensor
            batch = torch.stack([load_image(f, resize=(480, 480)) for f in files[i: i + batch_size]]).to(pu.get_module_device(extractor))
            # pass through the extractor
            try:
                
                image_feats = extractor.forward(data={"image": batch})
                kpnts, scores = image_feats['keypoints'].detach().cpu().numpy(), image_feats['keypoint_scores'].detach().cpu().numpy()

            except ValueError:
                # reaching this point means that the model does not detect a minimal number of keypoints 
                # on certain images
                image_feats = [extractor.forward(data={"image": i}) for i in batch]
                kpnts = [torch.squeeze()(im_f['keypoints']).detach().cpu().numpy() for im_f in image_feats]
                scores = [torch.squeeze()(im_f['keypoint_scores']).detach().cpu().numpy() for im_f in image_feats]

            # extract the reference images: don't forget to add the batch size into the indexing process
            ref_images = [(file_index, files[i + file_index]) for file_index, s in enumerate(scores) if np.sum(s >= min_kpnt_confidence).item() >= min_num_kpnts]

            for file_index, rf_im in ref_images: 
                im = cv.imread(rf_im)
                # save the reference image
                cv.imwrite(os.path.join(output_dir, os.path.basename(rf_im)), im)
                
                if debug:                    
                    # make sure to display the reference image
                    for x, y in kpnts[file_index]:
                        # draw the points
                        cv.circle(im, center=(int(x), int(y)), radius=1, color=(0, 255, 0), thickness=2)
                    
                    # display the image
                    cv.imshow('reference image', im)
                    cv.waitKey()
                    cv.destroyAllWindows()    

                    good_input = False
                    
                    while not good_input:
                        user_input = input((f"Please choose the class for which this image can server as a reference\n"
                                            f"\n Please enter: penny, Leo, hall, or none\n"))

                        good_input = user_input.lower() in ['penny', 'leo', 'hall', 'none']

                        if good_input and user_input != 'none':
                            cls_folder = os.path.join(output_dir, user_input) 
                            os.makedirs(cls_folder, exist_ok=True)
                            shutil.copyfile(src=rf_im, dst=os.path.join(cls_folder, os.path.basename(rf_im)))


        except Exception as e:
            print(e)
            print("error raised !!")
            continue        


def _keypoints_comparison(
        extractor: SuperPoint,
        matcher: LightGlue,
        reference_feats: Dict[str, torch.Tensor],
        images: List[str],
        resize: Union[Tuple, int],
        display: bool = False,
        device: str = None, 
        references: List[str] = None
        ):  
    # set the device
    device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # make sure to call the length operator on one of the keys (since reference_feats is a dictionary)
    num_references = len(reference_feats['descriptors'])
    num_images = len(images)
    # the original approach was to simple create an 'images' and 'references; tensors of length: num_references * num_images
    # as this step might be delayed
    
    # first let's get the descriptors and keypoints for each 
    images = torch.stack([load_image(im, resize=resize) for im in images]).to(device)

    if references is not None:
        references = torch.stack([load_image(im, resize=resize) for im in images]).to(device)

    image_feats = extractor.forward(data={"image": images})
    
    # extract both the descriptors and the keypoints
    k_images, d_images = image_feats['keypoints'], image_feats['descriptors']
    k_refs, d_refs = reference_feats['keypoints'], reference_feats['descriptors']

    # the images should be repeated in blocks of 'num_references': so the final tensor will be of length num_references * num_images
    # with each block of 'num_references' elements represent the same image  
    
    expanded_k_images = torch.stack([k_images[i].cpu() for i in range(num_images) for _ in range(num_references)]).to(device)
    expanded_d_images = torch.stack([d_images[i].cpu() for i in range(num_images) for _ in range(num_references)]).to(device)
    
    assert expanded_k_images.ndim == 3 and (expanded_k_images.shape[0], expanded_k_images.shape[2]) == (num_images * num_references, 2), f"Found: references of shape {expanded_k_images.shape}"
    
    assert (expanded_d_images.ndim == 3 and expanded_d_images.shape[0] == num_images * num_references, f"Found: references of shape {expanded_d_images.shape}")

    expanded_k_refs = torch.concat([k_refs for _ in range(num_images)], dim=0).to(device)
    expanded_d_refs = torch.concat([d_refs for _ in range(num_images)], dim=0).to(device)
    
    assert expanded_k_refs.ndim == 3 and (expanded_k_refs.shape[0], expanded_k_refs.shape[2]) == (
        num_images * num_references, 2), f"Found: references of shape {expanded_k_refs.shape}"
    
    assert (expanded_d_refs.ndim == 3 and expanded_d_refs.shape[0] == num_images * num_references, 
                f"Found: references of shape {expanded_d_refs.shape}")
    
    # create a tensor for the sizes of the images
    size_tensor = torch.from_numpy(np.asarray([list(resize) for _ in range(num_references * num_images)])).to(device)

    predictions = matcher.forward({"image0": {"keypoints": expanded_k_images,
                                              "descriptors": expanded_d_images,
                                              "image_size": size_tensor},
                                   "image1": {"keypoints": expanded_k_refs,
                                              "descriptors": expanded_d_refs,
                                              "image_size": size_tensor},
                                   })
    # extract the matches and the matching scores.
    matches = predictions['matches']
    scores = predictions['scores']
    assert isinstance(matches, List) and len(matches) == num_images * num_references, "check the matches object"

    if display:
        if references is None:
            raise TypeError(f"the 'references' argument must be passed when the 'display' argument is set to True")
        for index, (m, s) in enumerate(zip(matches, scores)):
            num_kpnts = s.size(dim=0)
            s = s.unsqueeze(dim=1).expand(-1, 2)
            assert s.shape == (num_kpnts, 2)
            m = torch.masked_select(m, s >= 0.7).reshape(-1, 2)
            viz2d.plot_images([images[index // num_references].cpu(), references[index % num_references].cpu()])
            m_kpts0, m_kpts1 = expanded_k_images[index][m[..., 0]].cpu(), expanded_k_refs[index][m[..., 1]].cpu()
            viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)

    # the number of matches between the given images and the reference ones.
    return [torch.sum(s >= 0.75) for index, s in enumerate(scores)]


class KeyPointClassifier:
    @classmethod
    def _build_references(cls, 
                          references: Union[Dict, str, Path],
                          image_extensions: Sequence[str] = None) -> Dict:
        
        if isinstance(references, Dict):
            for k, v in references.items():
                if not os.path.isfile(v):
                    raise ValueError(f"The dictionary is expected to map the class to image paths")

            return references

        if isinstance(references, (str, Path)):
            references = dirf.process_save_path(references) 

            if image_extensions is None:
                image_extensions = ['.png', '.jpg', 'jpeg']

            # quick data check
            # make sure the path to the data_folder is absolute
            if not os.path.isabs(references):
                raise ValueError(f"Please make sure the path to the data folder is absolute")

            for folder in os.listdir(references):
                folder_path = dirf.process_save_path(os.path.join(references, folder), 
                                                     file_ok=False)
                
                # make sure all the references are images
                for im in os.listdir(folder_path):
                    if not os.path.isfile(os.path.join(folder_path, im)) or os.path.splitext(im)[1] not in image_extensions:
                        raise ValueError(f"The file {im} is either a directory or a file with an unexpected extension.")

            references_map = {}
            
            for cls_name in os.listdir(references):
                references_map[cls_name] = [os.path.join(references, cls_name, im) for im in os.listdir(os.path.join(references, cls_name))]
            
            return references_map

    def __init__(self, 
                 references: Union[Dict, str, Path],
                 resize: Union[Tuple[int, int], int]=None, 
                 device: str = None, 
                 batch_size: int = 4,
                 referenes_threshold: float = 0.2,  
                 classification_treshold: int = 20) -> None:
        
        device = ('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.classification_treshold = classification_treshold
        self.batch_size = batch_size
        
        if resize is None:
            resize = 480

        self.resize = (resize, resize) if isinstance(resize, int) else resize
        # build a map between the class name and paths to the corresponding references.
        self.references_location_map = self._build_references(references=references)
        
        # initialize the extractor and the matcher
        self.extractor = SuperPoint(max_num_keypoints=512).eval().cuda()  # load the extractor
        self.matcher = LightGlue(features='superpoint', width_confidence=-1, filter_treshold=0.7).eval().cuda()        
        self.references_threshold = referenes_threshold

        # let's save the feats of the reference images
        self.references_feats = {}
        for cls_name, imgs in self.references_location_map.items():
            imgs_tensors = torch.stack([load_image(r, resize=resize) for r in imgs]).to(device)
            imgs_feats = self.extractor.forward(data={"image": imgs_tensors})            
            self.references_feats[cls_name] = imgs_feats
        # now the classifier is ready to go

    def classify(self, frames: List[str]) -> List[Optional[str]]:
        predictions = []

        for i in range(0, len(frames), self.batch_size):
            # create the data batch
            batch = frames[i: i + self.batch_size]
            
            batch_classification_map = defaultdict(lambda : {})

            for cls_name, cls_ref_feats in self.references_feats.items():

                threshold = int(math.ceil(len(cls_ref_feats) * self.references_threshold))

                images_score = defaultdict(lambda: 0)

                for j in range(0, len(cls_ref_feats), self.batch_size):
                    ref_batch = {k: v[j: j + self.batch_size] for k, v in cls_ref_feats.items()}

                    results = _keypoints_comparison(extractor=self.extractor,
                                                    matcher=self.matcher, 
                                                    reference_feats=ref_batch,
                                                    images=batch,
                                                    resize=self.resize)
                    
                    for pair_index, matches in enumerate(results):
                        # diving pair_index by the number of references produces the
                        # index of the image in the 'batch' object
                        images_score[pair_index // len(ref_batch)] += int(matches >= self.classification_treshold)

                # at this point the images in the batch have been compared to all reference images
                # we will create a map between each image index to another map mapping the cls to the image's score on that cls
                for im_index, score in images_score.items():
                    # assign 0 to classes with a score less than the thresold
                    batch_classification_map[im_index][cls_name] = score * (score >= threshold)
            
            # at this point all the images have been associated with a score for every reference
            # time to classify

            batch_preds = []
            for frame_batch_index, frame_score_map in batch_classification_map.items():
                label, label_score = max(list([(k, v) for k, v in frame_score_map.items()]), key=lambda x: (x[1]))                   
                # make sure to set the predcitions to None if the maximum score is less thatn the threshold
                batch_preds.append((label if label_score >= threshold else None))
             
            predictions.extend(batch_preds)                

        return predictions


if __name__ == '__main__':
    # current = os.path.dirname(os.path.realpath(__file__))
    # classifier = KeyPointClassifier(references=os.path.join(PARENT_DIR, 'src', 'build_dataset/test/references'), 
    #                                 resize=(480, 480))

    # data_path = os.path.join(PARENT_DIR, 'src', 'build_dataset/test/data')
    # frames = [os.path.join(data_path, f) for f in os.listdir(data_path)]
    # preds = classifier.classify(frames=frames)

    current = os.path.dirname(os.path.realpath(__file__))
    labeled_data_dir = os.path.join(PARENT_DIR, 
                                    'src', 
                                    'visual_system', 
                                    'scene',
                                    'unlabeled_data', 
                                    'sanity_train')
    
    extractor = SuperPoint(max_num_keypoints=512).eval().cuda()  # load the extractor

    select_reference_files(data_directory=os.path.join(PARENT_DIR, 'src', 'visual_system', 'scene','cls_references'), 
                           output_directory=os.path.join(PARENT_DIR, 'src', 'visual_system', 'scene','cls_references'),
                           extractor=extractor, 
                           min_kpnt_confidence=0.40, 
                           min_num_kpnts=60,
                           batch_size=4,
                           debug=True)
     