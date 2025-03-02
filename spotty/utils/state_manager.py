# spotty/utils/state_manager.py
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class SpotState:
    """Centralized state management for the robot"""

    # Navigation state
    waypoint_id: str = ""
    location: str = ""
    prev_location: str = ""
    prev_waypoint_id: str = ""

    # Vision state
    current_images: Dict[str, Any] = field(default_factory=dict)
    what_it_sees: Optional[Dict[str, Any]] = None

    # System state
    is_running: bool = False
    is_recording: bool = False

    # Power state
    is_powered_on: bool = False

    def update_location(self, waypoint_id: str, location: str):
        """Update location state"""
        self.prev_waypoint_id = self.waypoint_id
        self.prev_location = self.location
        self.waypoint_id = waypoint_id
        self.location = location

    def update_vision(self, images: Dict[str, Any], annotations: Optional[Dict[str, Any]] = None):
        """Update vision state"""
        self.current_images = images
        if annotations:
            self.what_it_sees = annotations
