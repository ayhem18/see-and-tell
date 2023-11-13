"""
This script contains the functionalities of a simple tool used to partially automate
the labeling for the scene classification task
"""

import os, sys

import torch
import itertools
import math

from typing import Union, List
from pathlib import Path
from collections import defaultdict
from shutil import copyfile


current = os.path.dirname(os.path.realpath(__file__))

while 'src' not in os.listdir(current):
    current = Path(current).parent

PARENT_DIR = current

sys.path.append(str(current))
sys.path.append(os.path.join(current, 'src'))


import src.utilities.directories_and_files as dirf

from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
from lightglue import viz2d


def _keypoints_descriptors_classifier(
        extractor,
        matcher,
        references: List[str],
        images: List[str],
        display: bool = False,
        device: str = None):
    device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')

    num_references = len(references)
    num_images = len(images)

    # first create a list where each image is repeated 'num_references' on a row
    images = list(itertools.chain(*[[im] * num_references for im in images]))
    images = torch.stack([load_image(im, resize=(480, 480)) for im in images]).to(device)

    # prepare the references
    references = list(itertools.chain(*[references for _ in range(num_images)]))
    references = torch.stack([load_image(r, resize=(480, 480)) for r in references]).to(device)

    assert references.ndim == 4 and (references.shape[0], references.shape[1]) == (
        num_images * num_references, 3), f"Found: references of shape {references.shape}"

    # the data is ready to be passed to the models
    image_feats = extractor.forward(data={"image": images.to(device)})
    reference_feats = extractor.forward(data={"image": references.to(device)})

    predictions = matcher.forward({"image0": {"keypoints": image_feats['keypoints'],
                                              "descriptors": image_feats['descriptors'],
                                              "image": images},
                                   "image1": {"keypoints": reference_feats['keypoints'],
                                              "descriptors": reference_feats['descriptors'],
                                              "image": references},
                                   })

    kpnts1, kpnts2 = image_feats['keypoints'], reference_feats['keypoints']

    # extract the matches and the matching scores.
    matches = predictions['matches']
    scores = predictions['scores']

    assert isinstance(matches, List) and len(matches) == num_images * num_references, "check the matches object"

    if display:
        for index, (m, s) in enumerate(zip(matches, scores)):
            num_kpnts = s.size(dim=0)
            s = s.unsqueeze(dim=1).expand(-1, 2)
            assert s.shape == (num_kpnts, 2)
            m = torch.masked_select(m, s >= 0.6).reshape(-1, 2)

            viz2d.plot_images([images[index].cpu(), references[index].cpu()])
            m_kpts0, m_kpts1 = kpnts1[index][m[..., 0]].cpu(), kpnts2[index][m[..., 1]].cpu()
            viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)

    # the number of matches between the given images and the reference ones.
    return [torch.sum(s >= 0.7) for index, s in enumerate(scores)]


class SceneClassificationDSBuilder:
    def __init__(self,
                 root: Union[Path, str],
                 references_folder_name: str = 'references',
                 data_folder_name: str = 'data',
                 classification_threshold: int = 20,
                 references_threshold: Union[float, int] = 0.2,
                 image_extensions=None,
                 batch_size: int = 4):

        if image_extensions is None:
            image_extensions = ['.png', '.jpg', 'jpeg']

        # quick data check
        # make sure the path to the data_folder is absolute
        if not os.path.isabs(root):
            raise ValueError(f"Please make sure the path to the data folder is absolute")

        # inside the data folder there should be 2 other folders: references, data
        if set(os.listdir(root)) != {references_folder_name, data_folder_name}:
            raise ValueError(f"The class expects 2 folders in the data folder: {references_folder_name} and "
                             f"{data_folder_name}. Found:{os.listdir(root)}")

        self.root = root
        self.references_path = os.path.join(root, references_folder_name)
        self.data_path = os.path.join(root, data_folder_name)
        self.labeled_data_path = os.path.join(root, 'labeled_data')

        self.classification_threshold = classification_threshold
        self.batch_size = batch_size
        self.references_threshold = references_threshold

        # the references folder should contain only directories
        if any([not os.path.isdir(os.path.join(self.references_path, im)) for im in os.listdir(self.references_path)]):
            raise ValueError(f"All files in the reference folder are expected to be directories.")

        for im in os.listdir(self.data_path):
            if not os.path.isfile(os.path.join(self.data_path, im)) or os.path.splitext(im)[1] not in image_extensions:
                raise ValueError(f"The file {im} is either a directory or a file with an unexpected extension.")

        # now we are ready to create a folder for labeled data
        os.makedirs(os.path.join(root, 'labeled_data'), exist_ok=True)
        dirf.copy_directories(src_dir=self.references_path,
                              des_dir=self.labeled_data_path
                              )

        # initialize the extractor and the matcher
        self.extractor = SuperPoint(max_num_keypoints=512).eval().cuda()  # load the extractor
        self.matcher = LightGlue(features='superpoint', width_confidence=-1, filter_treshold=0.6).eval().cuda()

    def auto_label(self):
        data = [os.path.join(self.data_path, im) for im in os.listdir(self.data_path)]

        for i in range(0, len(data), self.batch_size):
            # create the data batch
            batch = data[i: i + self.batch_size]

            ref_dirs = [os.path.join(self.references_path, ref) for ref in os.listdir(self.references_path)]

            # iterate through the references directory
            for ref in ref_dirs:
                # build the absolute path for each image in the references folder
                references = [os.path.join(ref, im) for im in os.listdir(ref)]
                # calculate the threshold for this reference type
                # the number of references where the number of matches exceeds the classification threshold
                threshold = int(math.ceil(len(references) * self.references_threshold))

                images_score = defaultdict(lambda: 0)

                for j in range(0, len(references), self.batch_size):
                    ref_batch = references[j: j + self.batch_size]

                    results = _keypoints_descriptors_classifier(extractor=self.extractor,
                                                                matcher=self.matcher,
                                                                references=ref_batch,
                                                                images=batch,
                                                                display=False
                                                                )
                    # the 'results' object represents the following:
                    # each consecutive 'ref_batch' elements represent the number of strong matches
                    # between an image from 'batch' and all the given references in 'ref_batch'

                    for pair_index, matches in enumerate(results):
                        # diving pair_index by the number of references produces the
                        # index of the image in the 'batch' object
                        images_score[pair_index // len(ref_batch)] += int(matches >= self.classification_threshold)

                assert len(images_score) == len(batch)

                # iterate through the images to classify them if needed
                for im_index, score in images_score.items():
                    if score >= threshold:
                        # set the 'source' location
                        src_file = batch[im_index]
                        # set the 'destination' location
                        des_file = os.path.join(self.labeled_data_path, os.path.basename(ref), os.path.basename(batch[im_index]))
                        # copy the file
                        copyfile(src_file, des_file)


if __name__ == '__main__':
    current = os.path.dirname(os.path.realpath(__file__))
    builder = SceneClassificationDSBuilder(root=os.path.join(current, 'all_data'), batch_size=2)
    builder.auto_label()