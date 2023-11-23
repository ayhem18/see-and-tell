""" 
This script contains the class definition of VisualFrame, a class that wraps all the visual information associated with a given frame
"""

from typing import List, Union
from _collections_abc import Sequence

class VisualFrame():
    def __init__(self, 
                 characters: List[int], 
                 bboxes: List[Sequence[Union[float, int]]],
                 probs: List[float], 
                 emotions: List[Union[str, int]], 
                 scene: str
                 ) -> None:
        """
        Args:
            characters (List[int]): The characters present in the frame
            bboxes (List[Sequence[Union[float, int]]]): The bounding boxes surrounding each chacters (not the face)
            probs (List[float]): the confidence associated with each bounding box
            emotions (List[Union[str, int]]): The emotion associated with each face
            scene (str): The frame's scene
        """
        self.characters = characters
        self.bboxes = characters
        self.probs = characters
        self.emotions = characters
        self.scene = characters

        