from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional


class NavigationStrategy(ABC):
    """Abstract base class for different navigation strategies"""

    @abstractmethod
    def select_destination(self, current_waypoint: str, matching_waypoints: List[str]) -> Optional[str]:
        """Select destination waypoint based on strategy"""
        pass


class ClosestWaypointStrategy(NavigationStrategy):
    """Strategy for selecting the closest waypoint"""

    def select_destination(self, current_waypoint: str, matching_waypoints: List[str]) -> Optional[str]:
        closest, _ = self._graph_nav.find_nearest_farthest_waypoints(current_waypoint, matching_waypoints)
        return closest


class FarthestWaypointStrategy(NavigationStrategy):
    """Strategy for selecting the farthest waypoint"""

    def select_destination(self, current_waypoint: str, matching_waypoints: List[str]) -> Optional[str]:
        _, farthest = self._graph_nav.find_nearest_farthest_waypoints(current_waypoint, matching_waypoints)
        return farthest


class NavigationStatus(Enum):
    """Enum representing possible navigation outcomes"""

    SUCCESS = "success"
    FAILED = "failed"
    NO_PATH = "no_path"
    INVALID_DESTINATION = "invalid_destination"
    ROBOT_NOT_READY = "robot_not_ready"


class NavigationResult:
    """Value object representing the result of a navigation attempt"""

    def __init__(
        self, status: NavigationStatus, destination: Optional[str] = None, error_message: Optional[str] = None
    ):
        self.status = status
        self.destination = destination
        self.error_message = error_message

    @property
    def is_successful(self) -> bool:
        return self.status == NavigationStatus.SUCCESS


class NavigationCommand:
    """Command object encapsulating navigation parameters"""

    def __init__(self, strategy: NavigationStrategy, matching_waypoints: List[str]):
        self.strategy = strategy
        self.matching_waypoints = matching_waypoints


class WaypointNavigator:
    """Class responsible for executing waypoint navigation"""

    def __init__(self, graph_nav_client):
        self._graph_nav = graph_nav_client

    def execute_navigation(self, command: NavigationCommand) -> NavigationResult:
        """Execute navigation using the provided command"""
        try:
            # Get current localization
            localization_state = self._graph_nav.get_localization_state()
            current_waypoint = localization_state.localization.waypoint_id

            if not current_waypoint:
                return NavigationResult(NavigationStatus.ROBOT_NOT_READY, error_message="Robot is not localized")

            # Select destination using strategy
            destination = command.strategy.select_destination(current_waypoint, command.matching_waypoints)

            if not destination:
                return NavigationResult(
                    NavigationStatus.NO_PATH, error_message="No valid path to any matching waypoint"
                )

            # Attempt navigation
            success = self._graph_nav._navigate_to([destination])

            if success:
                return NavigationResult(NavigationStatus.SUCCESS, destination=destination)
            else:
                return NavigationResult(
                    NavigationStatus.FAILED, destination=destination, error_message="Navigation command failed"
                )

        except Exception as e:
            return NavigationResult(NavigationStatus.FAILED, error_message=f"Navigation error: {str(e)}")
