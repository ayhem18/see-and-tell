"""This script contains the data augmentation functionalities
"""
import math
import os
import shutil
import sys
from pathlib import Path
from random import random
from typing import Union, Iterable, Tuple, List

import cv2 as cv
import numpy as np
import torch
from ultralytics import YOLO

current = os.path.dirname(os.path.realpath(__file__))

while 'src' not in os.listdir(current):
    current = Path(current).parent

PARENT_DIR = current

sys.path.append(str(current))
sys.path.append(os.path.join(current, 'src'))

import src.utilities.directories_and_files as dirf


# the first idea for data augmentation is to cover the human bounding boxes in each image
def xywh_converter(xywh: Iterable[int]) -> \
        Tuple[int, int, int, int]:
    # the idea is to accept a xywh bounding box and return 4 coordinates that can directly be
    # used for cropping
    x_center, y_center, w, h = xywh
    return y_center - h // 2, y_center + h // 2, x_center - w // 2, x_center + w // 2


def show_bounding_box(image: np.ndarray, x0: int, y0: int, x1: int, y1: int):
    cv.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), 2)
    # add the label
    cv.imshow('frame', image)
    cv.waitKey()
    cv.destroyAllWindows()


def detect_humans(batch: List[Union[np.ndarray, str, Path]],
                  yolo_model: YOLO,
                  device: str = None,
                  debug: bool = False) -> list[list[tuple[int, int, int, int]]]:
    """

    Args:
        batch: A batch of images (of a type that can be passed directly to a YOLO model)
        yolo_model: an instance of YOLO
        device: the model's device
        debug: a boolean flag to display a given image with the extracted bounding boxes

    Returns: a list with the same length as the batch where each element is a list with processed coordinates
    of bounding boxes

    """
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

    # only predict humans
    yolo_results = yolo_model.predict(batch, device=device, conf=0.4, iou=0.6, classes=0)
    results = [[xywh_converter(b) for b in r.boxes.xywh.cpu().numpy()] for r in yolo_results]
    return results


def human_image_augmentation(image: Union[np.ndarray, str, Path],
                             x0: int = None,
                             x1: int = None,
                             y0: int = None,
                             y1: int = None,
                             boxes: List[Tuple[int, int, int, int]] = None,
                             debug: bool = False):
    # simple check before proceeding
    if (x0 is None or x1 is None or y0 is None or y1 is None) and boxes is None:
        return TypeError(f"Please make sure to pass either a list of bounding boxes, or 4 points")

    if boxes is None:
        boxes = [[y0, y1, x0, x1]]

    image = cv.imread(image) if isinstance(image, (str, Path)) else image
    img_c = image.copy()

    for b in boxes:
        y0, y1, x0, x1 = b
        # define the bounding box to black out
        bounding_box = np.asarray([[x0, y0], [x0, y1], [x1, y1], [x1, y0]], np.int32)
        cv.fillPoly(img_c, pts=[bounding_box], color=(0, 0, 0))

    if debug:
        cv.imshow("augmented", img_c)
        cv.waitKey(0)
        cv.destroyAllWindows()

    return img_c


def human_data_augmentation(dataset_directory: Union[Path, str],
                            output_directory: Union[Path, str],
                            batch_size: int = 32,
                            max_human_area: float = 0.5,
                            debug: bool = False,
                            image_extensions: List[str] = None
                            ):

    # first process both directories
    src = dirf.process_save_path(dataset_directory,
                                 file_ok=False,
                                 condition=lambda x: dirf.all_images(x, image_extensions=image_extensions)
                                 )
    # the output directory must be empty
    des = dirf.process_save_path(output_directory,
                                 file_ok=False,
                                 condition=lambda x: len(os.listdir(x)) == 0)
    # first read the files
    files = sorted([os.path.join(src, f) for f in os.listdir(src)], key=lambda x: os.path.basename(x))

    yolo = YOLO('yolov8n.pt')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    num_batches = int(math.ceil(len(files) // batch_size)) + int(len(files) < batch_size)

    for i in range(num_batches):
        # extract the batch
        batch = files[i * batch_size: (i + 1) * batch_size]

        results = detect_humans(batch=batch, yolo_model=yolo, device=device)

        for im_file_name, res in zip(batch, results):
            im = cv.imread(os.path.join(im_file_name))
            im_area = im.shape[0] * im.shape[1]
            total_human_area = 0
            for r in res:
                y0, y1, x0, x1 = r
                total_human_area += np.abs(y0 - y1) * np.abs(x0 - x1)

            if debug:
                p = random()
                if p <= 0.9:
                    for r in res:
                        y0, y1, x0, x1 = r
                        show_bounding_box(im, x0=int(x0), y0=int(y0), x1=int(x1), y1=int(y1))

            # at this point calculate the human area for each picture and decide whether to discard the image
            # or augment it
            if total_human_area / im_area <= max_human_area:
                aug_im = human_image_augmentation(im, boxes=res, debug=debug)
                # copy the original image and save the augmented one in the output directory
                shutil.copy(os.path.join(im_file_name), os.path.join(des, os.path.basename(im_file_name)))
                base_name, ext = os.path.splitext(os.path.basename(im_file_name))
                aug_file_name = base_name + "_aug_" + ext
                cv.imwrite(filename=os.path.join(des, aug_file_name), img=aug_im)


if __name__ == '__main__':
    src = os.path.join(PARENT_DIR, 'src', 'build_dataset', 'tbbt_image_dataset', 'frames')
    des = os.path.join(PARENT_DIR, 'src', 'scene', 'augmented_data')
    human_data_augmentation(dataset_directory=src,
                            output_directory=des,
                            debug=False)

