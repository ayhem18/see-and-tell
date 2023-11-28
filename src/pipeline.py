"""The main pipeline for the See And Tell project.

Process an input video in the following way:

1) Splits the video into frames and audio
2) Detects segments with no speech
3) Extracts descriptions for each segment
4) Recognizes faces in each segment and modifies descriptions
5) Generates audio for each description
6) Combines audios and video into a single file
"""

import asyncio
import os
import time
import uuid
import cv2
import torch
import hashlib

from .gpt.words import get_substitutions
from .gpt.captions import rewrite_with_llm

from . import utils
from .face.who import FaceRecognizer
from .say.caption import SpeechToText
from .captions.attention import CaptionAttentionSystem
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
            model_name="microsoft/git-large-textcaps",
            use_gpu=True,
        ) 
        
        
        self.cas = CaptionAttentionSystem(
            self.describer.git_vision_config.patch_size,
            self.describer.git_config.num_attention_heads,
            self.describer.git_config.num_hidden_layers,
            self.describer.git_vision_config.image_size,
        )

            
        # self.face_detector = FaceRecognizer()

        # if embeddings_folder:
        #     self.face_detector.load_series_embeddings(embeddings_folder)

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
        if not os.path.exists(self.__get_audio(run_id)):
            self._split_on_audio(video, run_id)
        segments = self.speech_detector(self.__get_audio(run_id))
        segments = utils.get_frames_with_no_speech(
            segments, utils.get_length_of_video(video)
        )
        segments = utils.split_segments(segments, 10, 1)
        segments.sort(key=lambda x: x[0])
        return segments
    
    def _get_descriptions(self, run_id: str, frames: list[str]) -> list[CaptionOutput]:
        """Get descriptions for frames."""
        return self.describer.describe_batch(frames)
        
    
    def describe_video(self, video: str, save_to: str, from_series: str = None, segments=None) -> None:
        # Step 0: Generate a hash for video name
        # and current time to ensure unique folder name

        run_id = uuid.uuid4().hex
        
        frame_cuts = self._split_on_cuts(video, run_id)
        frames = [*frame_cuts.keys()]
        
        height, width = cv2.imread(frames[0]).shape[:2]
        # audio = self._split_on_audio(video, run_id)
        # segments = self._get_segments(run_id, video)
        
        vs = VisualSystem(
            reference_embeddings=os.path.join(
                self.embeddings_folder, from_series + '.json'
            )
        )
        
        seen = vs.see(
            frames=frames,
            frame_cuts=frame_cuts,
        )
        print([x.scene for x in seen])
        if segments is None:
            segments = self._get_segments(run_id, video)
        
        
        frames_to_proceed = []
        for start, end in segments:
            lst = [(ind, vf,) for ind, vf in enumerate(seen) if start <= ind <= end]
            if not lst:
                continue
            most_described_frame = max(
                lst,
                key=lambda x: len(x[1].characters)
            )
            # Forgive me lord, but I have to do this
            # as the first frame could not be described
            if most_described_frame[0] == 0:
                frames_to_proceed.append(most_described_frame[0] + 1)
            else:
                frames_to_proceed.append(most_described_frame[0])
        
        frames = [frames[i] for i in frames_to_proceed]
        
        outputs = self._get_descriptions(run_id, frames)
        descriptions = [output.caption_tokens for output in outputs]
        subs =  asyncio.run(
            get_substitutions([x.caption_tokens for x in outputs]),
        ) 
        
        subs = [[y.split() for y in x.words] for x in subs]
        
        for i, output in enumerate(outputs):
            matches = self.cas.match(
                output=output, 
                words=output.caption_tokens, 
                phrases=subs[i],
                boxes=seen[frames_to_proceed[i]].bboxes,
                height=height,
                width=width,
            )
            for (left, right), j, _ in matches:
                person = seen[frames_to_proceed[i]].characters[j]
                for k in range(left, right):
                    descriptions[i][k] = f'<{person.title()}>'
        
        text_descriptions: list[str] = []
        for i in range(len(descriptions)):
            # Remove repeating words in descriptions
            for j in range(len(descriptions[i]) - 1):
                if descriptions[i][j] == descriptions[i][j + 1]:
                    descriptions[i][j] = ''
            text_descriptions.append(' '.join(descriptions[i]))
            
            
        final_descriptions = asyncio.run(rewrite_with_llm(
            text_descriptions,
            [seen[i].scene for i in frames_to_proceed],
        ))
        
        print([x.caption for x in final_descriptions])
        # Step 5: Generate audio for each description
        audio_arrays = []
        for description in final_descriptions:
            audio_array = self.speech_to_text(description.caption)
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
