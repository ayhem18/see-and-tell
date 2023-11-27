import os
import sys
from pathlib import Path

from src.visual_system.scene import keypoint_classifier as kc

current = os.path.dirname(os.path.realpath(__file__))

while 'src' not in os.listdir(current):
    current = Path(current).parent

PARENT_DIR = current

sys.path.append(str(current))
sys.path.append(os.path.join(current, 'src'))

class SceneClassifier:
    _cls_references_path = os.path.join(current, 'src', 'visual_system', 'scene', 'cls_references')
    def __init__(self) -> None:
        # initialize the classifier
        # self.classifier = kc.KeyPointClassifier()
        pass

    def classify(self, images):
        return ['hall' for _ in images]