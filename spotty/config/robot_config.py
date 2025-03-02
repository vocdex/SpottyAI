from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class AudioConfig:
    """Audio configuration"""

    format: int = 8  # pyaudio.paInt16
    channels: int = 1
    rate: int = 48000
    chunk: int = 2048
    temp_file: str = "output.wav"


@dataclass
class WakeWordConfig:
    """Wake word detection configuration"""

    access_key: str
    keyword_path: str
    sensitivity: float = 0.5
    device_index: int = -1


@dataclass
class VisionConfig:
    """Vision system configuration"""

    required_sources: List[str] = field(default_factory=lambda: ["frontright_fisheye_image", "frontleft_fisheye_image"])
    rotation_angles: Dict[str, int] = field(
        default_factory=lambda: {
            "back_fisheye_image": 0,
            "frontleft_fisheye_image": -90,
            "frontright_fisheye_image": -90,
            "left_fisheye_image": 0,
            "right_fisheye_image": 180,
        }
    )


@dataclass
class RobotConfig:
    """Master configuration for the robot"""

    audio_config: AudioConfig = field(default_factory=AudioConfig)
    wake_word_config: Optional[WakeWordConfig] = None
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    system_prompt: str = ""
    map_path: str = ""
    vector_db_path: str = ""
    debug: bool = False
