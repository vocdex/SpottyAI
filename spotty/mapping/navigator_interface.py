# spotty/mapping/navigator_interface.py
import logging
from typing import Dict, Optional, Tuple

from spotty.mapping import GraphNavInterface
from spotty.utils.state_manager import SpotState


class NavigatorInterface:
    """Interface for handling robot navigation operations"""

    def __init__(self, graph_nav: GraphNavInterface, state: SpotState, logger=None):
        self.graph_nav = graph_nav
        self.state = state
        self.logger = logger or logging.getLogger(__name__)
        self.rag_system = None  # Set this after initialization

    def set_rag_system(self, rag_system):
        """Set the RAG system reference"""
        self.rag_system = rag_system

    def navigate_to_waypoint(self, waypoint_id: str, phrase: Optional[str] = None) -> bool:
        """Navigate to a specific waypoint"""
        if phrase:
            # Speak while moving
            # (Implementation to handle speech during navigation)
            pass

        is_successful = self.graph_nav._navigate_to([waypoint_id])

        if is_successful:
            self.state.update_location(waypoint_id, "")

            # Update location name if available
            if self.rag_system:
                annotations = self.rag_system.get_waypoint_annotations(waypoint_id)
                if annotations:
                    self.state.location = annotations.get("location", "")
                    self.state.what_it_sees = annotations

        return is_successful

    def navigate_to_location(self, location_name: str, phrase: Optional[str] = None) -> Tuple[bool, str]:
        """Navigate to a named location"""
        is_successful, waypoint_id = self.graph_nav._navigate_to_by_location([location_name])

        if is_successful:
            self.state.update_location(waypoint_id, location_name)

            # Get annotations if available
            if self.rag_system:
                annotations = self.rag_system.get_waypoint_annotations(waypoint_id)
                if annotations:
                    self.state.what_it_sees = annotations

        return is_successful, waypoint_id

    def search_for_object(self, query: str) -> Dict:
        """Search for an object using the RAG system"""
        if not self.rag_system:
            return {"success": False, "message": "RAG system not available"}

        enhanced_query = "Where do you see a " + query + "?"
        results = self.rag_system.query_location(enhanced_query, k=5)

        if not results:
            return {"success": False, "message": "No results found"}

        # Group results by location
        locations = {}
        for result in results:
            location = result["location"]
            if location not in locations:
                locations[location] = result

        return {"success": True, "results": results, "locations": locations}
