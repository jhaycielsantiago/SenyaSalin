"""
Landmark extraction module.

Converts raw video/webcam frames into structured landmark JSON
using MediaPipe Hands. Output conforms to schemas/landmark_schema.json.
"""

from .landmark_extractor import LandmarkExtractor
from .video_processor import VideoProcessor

__all__ = ["LandmarkExtractor", "VideoProcessor"]
