import os
import time
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from threading import Thread

# Mock Spot SDK classes
@dataclass
class MockLease:
    lease_proto: str = "mock_lease"

class MockClient:
    """Base mock client class"""
    def __init__(self, name: str):
        self.name = name
        
class MockLeaseClient(MockClient):
    def __init__(self):
        super().__init__("lease")
        self.lease_wallet = self
        
    def acquire(self):
        return MockLease()
        
    def return_lease(self, lease):
        pass
        
    def get_lease(self):
        return MockLease()

class MockRobotCommandClient(MockClient):
    def __init__(self):
        super().__init__("command")
    
    def robot_command(self, *args, **kwargs):
        logging.info("Mock robot command executed")

class MockGraphNavClient(MockClient):
    def __init__(self):
        super().__init__("graph_nav")
        self.current_waypoint = None
        
    def navigate_to(self, waypoint_id: str, *args, **kwargs):
        logging.info(f"Mock navigation to waypoint: {waypoint_id}")
        self.current_waypoint = waypoint_id
        return "mock_cmd_id"
        
    def navigation_feedback(self, cmd_id: str):
        class MockStatus:
            STATUS_REACHED_GOAL = "reached_goal"
            status = STATUS_REACHED_GOAL
        return MockStatus()

class MockPowerClient(MockClient):
    def __init__(self):
        super().__init__("power")

class MockRobotStateClient(MockClient):
    def __init__(self):
        super().__init__("state")
        
    def get_robot_state(self):
        class MockPowerState:
            motor_power_state = "STATE_ON"
            STATE_ON = "STATE_ON"
        class MockState:
            power_state = MockPowerState()
        return MockState()

class MockRobot:
    """Mock Spot robot for testing"""
    def __init__(self):
        self._clients = {
            "lease": MockLeaseClient(),
            "command": MockRobotCommandClient(),
            "graph_nav": MockGraphNavClient(),
            "power": MockPowerClient(),
            "state": MockRobotStateClient()
        }
        
    def ensure_client(self, name: str) -> MockClient:
        return self._clients.get(name)
        
    def time_sync(self):
        class MockTimeSync:
            def wait_for_sync(self):
                pass
        return MockTimeSync()

class MockLeaseKeepAlive:
    """Mock lease keep-alive"""
    def __init__(self, lease_client):
        self.lease_client = lease_client
        
    def shutdown(self):
        pass

class MockConversationAgent:
    """Mock wake word and conversation system"""
    def __init__(self, *args, **kwargs):
        self.running = False
        self.callback = None
        
    def start(self):
        """Start mock conversation system"""
        self.running = True
        print("\nMock conversation system started!")
        print("Type 'hey spot' to activate, or 'quit' to exit")
        
        # Start input thread
        self.input_thread = Thread(target=self._input_loop)
        self.input_thread.daemon = True
        self.input_thread.start()
        
    def _input_loop(self):
        """Mock input loop"""
        while self.running:
            try:
                user_input = input("> ").lower().strip()
                if user_input == "quit":
                    self.running = False
                    break
                    
                if user_input == "hey spot":
                    print("🔵 Wake word detected! What can I help you with?")
                    command = input("> ")
                    print(f"Processing command: {command}")
                    # Process command through main system
                    
            except Exception as e:
                print(f"Error in mock input: {e}")
                
    def stop(self):
        """Stop mock conversation system"""
        self.running = False
        print("\nMock conversation system stopped!")
        
    def text_to_speech(self, text: str, tts: str = 'openai'):
        """Mock TTS output"""
        print(f"🤖 Spot says: {text}")

class MockMultimodalRAG:
    """Mock RAG system for testing"""
    def __init__(self, *args, **kwargs):
        # Mock waypoint data
        self.mock_locations = {
            "kitchen": {
                "waypoint_id": "wp1",
                "location": "kitchen area",
                "description": "A kitchen setup with appliances and countertops"
            },
            "office": {
                "waypoint_id": "wp2", 
                "location": "office space",
                "description": "Office area with desks and computers"
            },
            "lab": {
                "waypoint_id": "wp3",
                "location": "robotics lab",
                "description": "Laboratory with robotic equipment"
            }
        }
        
    def query_location(self, query: str, k: int = 3) -> list:
        """Mock location query"""
        # Simple keyword matching for testing
        results = []
        query = query.lower()
        
        for key, data in self.mock_locations.items():
            if key in query or query in data['description'].lower():
                results.append({
                    'waypoint_id': data['waypoint_id'],
                    'location': data['location'],
                    'description': data['description'],
                    'distance': 1.0
                })
                
        return results[:k]

def test_integrated_system():
    """Test the integrated system with mocks"""
    from main_interface import IntegratedSpotSystem
    
    # Monkey patch the real classes with mocks
    import main_interface as integrated_spot
    integrated_spot.WakeWordConversationAgent = MockConversationAgent
    integrated_spot.MultimodalRAGAnnotator = MockMultimodalRAG
    integrated_spot.LeaseKeepAlive = MockLeaseKeepAlive
    
    # Create mock robot
    mock_robot = MockRobot()
    
    # Initialize system with mock components
    system = IntegratedSpotSystem(
        robot=mock_robot,
        map_path="/Users/shuk/Desktop/spot/practical-seminar-mobile-robotics/spotty/assets/maps/chair_graph_images",
        vector_db_path="/Users/shuk/Desktop/spot/practical-seminar-mobile-robotics/spotty/assets/rag_db/vector_db_chair"
    )
    
    print("\nStarting mock Spot system for testing...")
    print("Available test locations: kitchen, office, lab")
    print("Try commands like:")
    print("- 'hey spot' followed by 'where is the kitchen?'")
    print("- 'hey spot' followed by 'take me to the lab'")
    print("Type 'quit' to exit\n")
    
    try:
        system.start()
    except KeyboardInterrupt:
        system.stop()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_integrated_system()