"""Pipeline component that detects speech in an audio file."""""

import argparse
import logging

from pyannote.audio.pipelines import VoiceActivityDetection
from ..log import get_pipeline_logger


class SpeechDetector:
    def __init__(
            self,
            model_name: str = "anilbs/segmentation",
            **kwargs
        ) -> None:
        """Initialize the SpeechDetector class."""
        self.logger = get_pipeline_logger("SpeechDetector", 'purple')
        self.logger.info("Initialized SpeechDetector")
        self.pipeline = VoiceActivityDetection(segmentation=model_name)
        self.pipeline.instantiate(kwargs)
        self.logger.info("Loaded pyannote-audio model")

    def __call__(self, audio: str) -> list[tuple[float, float]]:
        """Detect speech in an audio file.

        Args:
            audio (str): The path to the audio file.

        Returns:
            list[tuple[float, float]]: A list of tuples containing the start and end times of each speech segment.
        """
        self.logger.info(f"Loading audio from {audio}")
        self.logger.info("Detecting segment with no speech")
        speech_segments = list(self.pipeline(audio).get_timeline())
        self.logger.info(f"Detected speech in {len(speech_segments)} segments")
        return speech_segments
