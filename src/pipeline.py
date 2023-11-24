"""The main pipeline for the See And Tell project.

Process an input video in the following way:

1) Splits the video into frames and audio
2) Detects segments with no speech
3) Extracts descriptions for each segment
4) Recognizes faces in each segment and modifies descriptions
5) Generates audio for each description
6) Combines audios and video into a single file
"""

import os
import time
import uuid
import torch
import hashlib

from . import utils
from .face.who import FaceRecognizer
from .say.caption import SpeechToText
from .listen.speech import SpeechDetector
from .describe.frame import FrameDescriptor, CaptionOutput
from .visual_system.face import video_utilities as vu
from .visual_system.visual_system import VisualSystem

class SeeAndTell:

    def __init__(self, temp_folder: str, embeddings_folder: str = None) -> None:
        """Initialize the SeeAndTell class."""

        # Initialize the SpeechDetector class
        self.speech_detector = SpeechDetector(
            onset=0.5,
            offset=0.5,
            min_duration_off=0,
            min_duration_on=0,
        )

        self.speech_to_text = SpeechToText()
        self.use_embeddings = embeddings_folder is not None
        self.embeddings_folder = embeddings_folder
        self.describer = FrameDescriptor(
            model_name="microsoft/git-base",
            use_gpu=torch.cuda.is_available(),
        ) 
        self.face_detector = FaceRecognizer()

        if embeddings_folder:
            self.face_detector.load_series_embeddings(embeddings_folder)

        self.temp_folder = temp_folder
        os.makedirs(self.temp_folder, exist_ok=True)
        
    def __get_dir(self, run_id: str, local_dir_name: str) -> str:
        """Get the path to a directory in the temp folder."""
        os.makedirs(
            os.path.join(self.temp_folder, run_id, local_dir_name), exist_ok=True
        )
        return os.path.join(self.temp_folder, run_id, local_dir_name)
    
    def __get_path(self, run_id: str, file_name: str) -> str: 
        """Get the path to a file in the temp folder."""
        os.makedirs(os.path.join(
            self.temp_folder, run_id), exist_ok=True)
        return os.path.join(self.temp_folder, run_id, file_name)
    
    def __get_frames(self, run_id: str) -> str:
        """Get the path to the frames folder in the temp folder."""
        return self.__get_dir(run_id, "frames")
    
    def __get_cuts(self, run_id: str) -> str:
        """Get the path to the cuts folder in the temp folder."""
        return self.__get_dir(run_id, "cuts")
    
    def __get_audio(self, run_id: str) -> str:
        """Get the path to the audio file in the temp folder."""
        return self.__get_path(run_id, "audio.mp3")
    
    def _split_on_frames(self, video: str, run_id: str) -> list[str]:
        """Split the video on frames."""
        frames = utils.split_on_frames(video, self.__get_frames(run_id))
        frames.sort()
        return frames
    
    def _split_on_cuts(self, video: str, run_id: str) -> dict:
        """Split the video on cuts."""
        frame_cuts = vu.video_to_cuts(
            video_path=video, 
            output_dir=self.__get_cuts(run_id),
        )
        return frame_cuts
    
    def _split_on_audio(self, video: str, run_id: str) -> None:
        """Split the video on audio."""
        utils.split_on_audio(video, self.__get_audio(run_id))
        
        return self.__get_audio(run_id)
    
    def _get_segments(self, run_id: str, video: str) -> list[tuple[int, int]]:
        """Get segments with no speech."""
        segments = self.speech_detector(self.__get_audio(run_id))
        segments = utils.get_frames_with_no_speech(
            segments, utils.get_length_of_video(video)
        )
        segments = utils.split_segments(segments, 10, 1)
        segments.sort(key=lambda x: x[0])
        return segments
    
    def _get_descriptions(self, run_id: str, frames: str) -> list[CaptionOutput]:
        """Get descriptions for frames."""
        return self.describer(frames)
        
    
    def describe_video(self, video: str, save_to: str, from_series: str = None) -> None:
        # Step 0: Generate a hash for video name
        # and current time to ensure unique folder name

        run_id = uuid.uuid4().hex
        
        frame_cuts = self._split_on_cuts(video, run_id)
        frames = [*frame_cuts.keys()]
        audio = self._split_on_audio(video, run_id)
        segments = self._get_segments(run_id, video)
        
        vs = VisualSystem(
            reference_embeddings=os.path.join(
                self.embeddings_folder, from_series + '.json'
            )
        )
        
        seen = vs.see(
            frames=frames,
            frame_cuts=frame_cuts,
        )
        
        descriptions = ...


        # Step 3: Get descriptions for each segment
        descriptions = {}
        for frame in frames:
            descriptions[frame] = (
                run_descriptor(frame)
            )
        descriptions = {i: d.lower() for i, d in descriptions.items()}

        if self.use_embeddings:
            # Step 4: Recognize faces in each segment
            desc_with_faces = []
            desc_with_faces, desc_indices, detections = self.face_detector(
                list(descriptions.keys()),
                list(descriptions.values()),
                from_series
            )

          # Step 4.1: Get the most described frame for each segment
            frames_to_proceed = []
            for start, end in segments:
                most_described_frame = max(
                    [(ind, detections[i]) for i, ind in enumerate(
                        desc_indices) if start <= ind <= end],
                    key=lambda x: len(x[1])
                )
                frames_to_proceed.append(most_described_frame[0])

            # Step 4.2: Enhance descriptions for each segment
            descriptions = [
                desc_with_faces[desc_indices.index(i)] for i in frames_to_proceed]

        else:
            frames_to_proceed = [int(s[0]) for s in segments]
            descriptions = list(descriptions.values())
            descriptions = [descriptions[i] for i in frames_to_proceed]

        print(frames_to_proceed, descriptions)
        print(desc_indices)

        # Step 5: Generate audio for each description
        audio_arrays = []
        for description in descriptions:
            audio_array = self.speech_to_text(description)
            audio_arrays.append(audio_array)

        # Step 6: Combine clips
        utils.mix_video_and_audio(
            video,
            audio_arrays,
            frames_to_proceed,
            save_to
        )


def run_pipeline(
        video: str,
        output: str,
        temporary_folder: str,
        embeddings_folder: str = './embeddings',
        serie: str = None
):
    """Run the pipeline on a video."""
    see_and_tell = SeeAndTell(temporary_folder, embeddings_folder)
    see_and_tell.describe_video(video, output, serie)
