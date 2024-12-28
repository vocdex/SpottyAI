"""This script contains the main interface for loading/uploading maps, and navigating to waypoints via GraphNav and Whisper/GPT4o-mini audio commands
The main parts are combined from graph_nav_command_line.py and spot_assistant.py scripts
"""
import os
import json
import time
from typing import Optional, Dict, List
import logging
from dataclasses import dataclass
from openai import OpenAI

from bosdyn.client.robot_command import RobotCommandClient
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.graph_nav import GraphNavClient
from bosdyn.client.power import PowerClient
from bosdyn.client.robot_state import RobotStateClient

@dataclass
class SpotState:
    curr_location_id: str
    location_description: str
    nearby_locations: List[str]
    spot_sees: str

class IntegratedSpotSystem:
    """
    Integrated system combining wake word detection, RAG, and navigation.
    """
    def __init__(
        self,
        robot,
        map_path: str,
        vector_db_path: str = "vector_db",
        system_prompt: Optional[str] = None,
        openai_api_key: Optional[str] = None,
    ):
        self.robot = robot
        
        # Initialize clients
        self._lease_client = self.robot.ensure_client(LeaseClient.default_service_name)
        self._robot_command_client = self.robot.ensure_client(RobotCommandClient.default_service_name)
        self._graph_nav_client = self.robot.ensure_client(GraphNavClient.default_service_name)
        self._power_client = self.robot.ensure_client(PowerClient.default_service_name)
        self._robot_state_client = self.robot.ensure_client(RobotStateClient.default_service_name)
        
        # Initialize lease management
        self._lease_wallet = self._lease_client.lease_wallet
        self._lease = self._lease_client.acquire()
        self._lease_keepalive = LeaseKeepAlive(self._lease_client)
        
        # Initialize RAG system
        from rag_label import MultimodalRAGAnnotator
        graph_file_path, snapshot_dir, _ = self._get_map_paths(map_path)
        self.rag_system = MultimodalRAGAnnotator(
            graph_file_path=graph_file_path,
            snapshot_dir=snapshot_dir,
            vector_db_path=vector_db_path,
            load_clip=False
        )
        
        # Initialize conversation system
        from .audio_control.spot_assistant import WakeWordConversationAgent
        self.conversation_agent = WakeWordConversationAgent(
            access_key=os.getenv("PICOVOICE_ACCESS_KEY"),
            keyword_path="./hey_spot_version_02/Hey-Spot_en_mac_v3_0_0.ppn",
            transcription_method='openai',
            inference_method='openai',
            tts='openai'
        )
        
        # Initialize OpenAI client for action planning
        self.openai_client = OpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))
        self.system_prompt = system_prompt or """
        # Spot Robot API
        You are controlling a Spot robot that can navigate autonomously and interact through speech.
        
        Available actions:
        1. navigate_to(waypoint_id, phrase): Move to a specific waypoint while speaking
        2. say(phrase): Say something using text-to-speech
        3. ask(question): Ask a question and wait for response
        4. search(query): Search the environment using RAG system
        
        Be concise and use exactly one action per response.
        """
        
        # Store current state
        self._current_graph = None
        self._current_edges = {}
        self._current_waypoint_snapshots = {}
        self._current_annotation_name_to_wp_id = {}
        
        # Power state
        power_state = self._robot_state_client.get_robot_state().power_state
        self._started_powered_on = (power_state.motor_power_state == power_state.STATE_ON)
        self._powered_on = self._started_powered_on

    async def process_voice_command(self, voice_input: str) -> str:
        """Process voice command through LLM to get next action"""
        # First search environment if needed
        if "where" in voice_input.lower() or "find" in voice_input.lower():
            search_results = self.rag_system.query_location(voice_input)
            if search_results:
                destination = search_results[0]['waypoint_id']
                location_desc = search_results[0]['location']
                return f'navigate_to("{destination}", "I found {location_desc}. Follow me!")'
        
        # Otherwise get next action from LLM
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"User said: {voice_input}\nEnter exactly one action:"}
        ]
        
        response = await self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=100
        )
        
        return response.choices[0].message.content

    async def execute_action(self, action: str):
        """Execute the LLM-generated action"""
        try:
            # Safely evaluate the action
            action = action.strip()
            if action.startswith("navigate_to"):
                waypoint_id = eval(action.split("(")[1].split(",")[0])
                phrase = eval(",".join(action.split(",")[1:])[:-1])
                await self._navigate_to(waypoint_id, phrase)
            elif action.startswith("say"):
                phrase = eval(action.split("(")[1][:-1])
                self.conversation_agent.text_to_speech(phrase, tts='openai')
            elif action.startswith("ask"):
                question = eval(action.split("(")[1][:-1])
                self.conversation_agent.text_to_speech(question, tts='openai')
            elif action.startswith("search"):
                query = eval(action.split("(")[1][:-1])
                return self.rag_system.query_location(query)
        except Exception as e:
            logging.error(f"Error executing action: {e}")

    async def _navigate_to(self, waypoint_id: str, phrase: Optional[str] = None):
        """Navigate to a specific waypoint"""
        if phrase:
            self.conversation_agent.text_to_speech(phrase, tts='openai')
            
        nav_cmd_id = None
        while True:
            try:
                nav_cmd_id = self._graph_nav_client.navigate_to(
                    waypoint_id,
                    1.0,
                    leases=[self._lease.lease_proto],
                    command_id=nav_cmd_id
                )
            except Exception as e:
                logging.error(f"Navigation error: {e}")
                break
                
            time.sleep(0.5)
            status = self._graph_nav_client.navigation_feedback(nav_cmd_id)
            if status.status == status.STATUS_REACHED_GOAL:
                break
            elif status.status in [status.STATUS_LOST, status.STATUS_STUCK]:
                logging.error(f"Navigation failed: {status.status}")
                break

    def start(self):
        """Start the integrated system"""
        try:
            # Start wake word detection and conversation
            self.conversation_agent.start()
            
            # Keep the main thread running
            while True:
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            self.stop()
            
    def stop(self):
        """Stop the integrated system"""
        self.conversation_agent.stop()
        self._lease_keepalive.shutdown()
        self._lease_client.return_lease(self._lease)

    @staticmethod
    def _get_map_paths(map_path: str):
        """Get paths for graph and snapshots"""
        from utils.common_utils import get_map_paths
        return get_map_paths(map_path)

def main():
    """Main entry point"""
    import argparse
    from bosdyn.client import create_standard_sdk
    from utils.robot_utils import auto_authenticate
    
    parser = argparse.ArgumentParser(description="Integrated Spot System")
    parser.add_argument("--hostname", required=True, help="Robot hostname")
    parser.add_argument("--map-path", required=True, help="Path to map directory")
    args = parser.parse_args()
    
    # Initialize SDK and robot
    sdk = create_standard_sdk('IntegratedSpotClient')
    robot = sdk.create_robot(args.hostname)
    auto_authenticate(robot)
    
    # Create and start integrated system
    system = IntegratedSpotSystem(
        robot=robot,
        map_path=args.map_path
    )
    
    try:
        system.start()
    except Exception as e:
        logging.error(f"Error running integrated system: {e}")
    finally:
        system.stop()

if __name__ == "__main__":
    main()