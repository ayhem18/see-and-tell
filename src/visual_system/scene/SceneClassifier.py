import os
import sys
from pathlib import Path
from typing import Any

current = os.path.dirname(os.path.realpath(__file__))

while 'src' not in os.listdir(current):
    current = Path(current).parent

PARENT_DIR = current

sys.path.append(str(current))
sys.path.append(os.path.join(current, 'src'))


from src.visual_system.scene import keypoint_classifier as kc
from src.visual_system.scene.autoencoders.scene_autoencoder import SceneDenoiseAE
import src.visual_system.scene.classifier.ae_classifier as aec


class SceneClassifier:
    def __init__(self) -> None:
        self.classifier = kc.KeyPointClassifier(references=os.path.join(PARENT_DIR, 'src', 'visual_system', 'scene', 'cls_references'), 
                                        resize=(480, 480), batch_size=2)

    def classify(self, images):
        return self.classifier.classify(images)
    
if __name__ == '__main__':
    s = SceneClassifier()
    data_path = os.path.join(PARENT_DIR, 'src', "visual_system/scene/labeled_data/train/Penny's apartment")
    frames = [os.path.join(data_path, f) for f in os.listdir(data_path)][:5]
    preds = s.classify(frames)
    print(preds)
