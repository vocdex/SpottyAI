import os
import time

from typing import Optional, List, Dict, Any
import threading
import bosdyn.client
import logging
from queue import Queue, Empty
from dataclasses import dataclass

from spotty.audio import WakeWordConversationAgent
from spotty.mapping import GraphNavInterface
from spotty.annotation import MultimodalRAGAnnotator
from dotenv import load_dotenv
from spotty import ASSETS_PATH
from spotty.audio import system_prompt_assistant
from spotty.utils.common_utils import get_map_paths
from spotty.utils.robot_utils import auto_authenticate, HOSTNAME

os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()
KEYWORD_PATH = os.path.join(ASSETS_PATH, "/hey_spot_version_02/Hey-Spot_en_mac_v3_0_0.ppn")




class UnifiedSpotInterface:
    def __init__(
        self,
        robot,
        map_path: str,
        vector_db_path: str,
        system_prompt: str,
        keyword_path: str,
        transcription_method: str = 'openai',
        inference_method: str = 'openai',
        tts: str = 'openai',
        audio_device_index: int = -1,
        debug: bool = False
    ):
        # Set up logging
        logging.basicConfig(
            level=logging.DEBUG if debug else logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        """Initialize the unified interface for Spot robot control."""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Unified Spot Interface...")
        self.command_queue = Queue()

        
        try:
            self.logger.info("Initializing GraphNav...")
            # Initialize GraphNav
            self.graph_nav = GraphNavInterface(robot, map_path)
            # Initialize map
            self.graph_nav._initialize_map(maybe_clear=False)
            self.logger.info("GraphNav initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize GraphNav: {str(e)}")
            raise
        
        try:
            self.logger.info("Initializing RAG system...")
            # Initialize RAG system
            graph_file_path, snapshot_dir, _ = get_map_paths(map_path)
            self.rag_system = MultimodalRAGAnnotator(
                graph_file_path=graph_file_path,
                logger=self.logger,
                snapshot_dir=snapshot_dir,
                vector_db_path=vector_db_path,
                load_clip=False
            )
            self.logger.info("RAG system initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG system: {str(e)}")
            raise
        
        try:
            self.logger.info("Initializing Voice Interface...")
            # Check for required environment variables
            picovoice_key = os.getenv("PICOVOICE_ACCESS_KEY")
            if not picovoice_key:
                raise ValueError("PICOVOICE_ACCESS_KEY environment variable not set")
            
            # Initialize Voice Interface
            self.voice_interface = WakeWordConversationAgent(
                access_key=picovoice_key,
                system_prompt=system_prompt,
                keyword_path=keyword_path,
                transcription_method=transcription_method,
                inference_method=inference_method,
                tts=tts,
                audio_device_index=audio_device_index
            )
            self.voice_interface.command_queue = self.command_queue

            self.logger.info("Voice Interface initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Voice Interface: {str(e)}")
            raise
        
        # Command queue for handling navigation and queries
        self.command_thread = None
        self.is_running = False

    def _extract_command(self, response: str) -> Dict[str, Any]:
        """Extract command and parameters from LLM response."""
        try:
            # Parse the response to identify the command and parameters
            if "navigate_to(" in response:
                cmd = "navigate_to"
                # Extract waypoint_id and phrase
                parts = response.split("navigate_to(")[1].split(")")[0].split(",")
                params = {
                    "waypoint_id": parts[0].strip(),
                    "phrase": parts[1].strip() if len(parts) > 1 else ""
                }
            elif "say(" in response:
                cmd = "say"
                phrase = response.split("say(")[1].split(")")[0]
                params = {"phrase": phrase}
            elif "ask(" in response:
                cmd = "ask"
                question = response.split("ask(")[1].split(")")[0]
                params = {"question": question}
            elif "search(" in response:
                cmd = "search"
                query = response.split("search(")[1].split(")")[0]
                params = {"query": query}
            else:
                raise ValueError(f"Unknown command in response: {response}")
            
            return {"command": cmd, "parameters": params}
        except Exception as e:
            self.logger.error(f"Error extracting command: {e}")
            return {"command": "say", "parameters": {"phrase": "I'm sorry, I couldn't process that command."}}

    def _execute_command(self, cmd_dict: Dict[str, Any]):
        """Execute the parsed command."""
        command = cmd_dict["command"]
        params = cmd_dict["parameters"]
        from spotty.audio.spot_assistant import text_to_speech

        try:
            if command == "navigate_to":
                assert "waypoint_id" in params, "Waypoint ID not provided"
                assert type(params["waypoint_id"]) == str, "Waypoint ID must be a string"
                text_to_speech(params["phrase"])
                destination_waypoint_tuple = (params["waypoint_id"], None)
                is_finished = self.graph_nav._navigate_to(destination_waypoint_tuple)
                if is_finished:
                    text_to_speech(f"You have arrived at your destination")
                    
            elif command == "say":
                text_to_speech(params["phrase"])
                
            elif command == "ask":
                from spotty.audio.spot_assistant import play_beep,record_audio
                # Play a sound to indicate listening
                self.command_queue.put({"command": "say", "parameters": {"phrase": params["question"]}})
                # Also record the response and add it to the chat history
                    
            elif command == "search":
                results = self.rag_system.query_location(params["query"], k=3, distance_threshold=2.0)

                if results:
                    # Get the closest matching waypoint
                    closest_waypoint = results[0]["waypoint_id"]
                    description = results[0]["description"]
                    print("Closest waypoint:", closest_waypoint) # Working until here
                    response = f"I found what you're looking for. Let me take you there"
                    self.command_queue.put({
                        "command": "navigate_to",
                        "parameters": {
                            "waypoint_id": closest_waypoint,
                            "phrase": response
                        }

                    })
                else:
                    self.command_queue.put({
                        "command": "say",
                        "parameters": {"phrase": "I couldn't find anything matching your search."}
                    })
                    
        except Exception as e:
            self.logger.error(f"Error executing command: {e}")
            self.command_queue.put({
                "command": "say",
                "parameters": {"phrase": f"I encountered an error: {str(e)}"}
            })

    def _command_processing_loop(self):
        """Main loop for processing commands."""
        self.logger.info("Starting command processing loop")
        while self.is_running:
            try:
                self.logger.debug("Waiting for next command...")
                cmd_dict = self.command_queue.get(timeout=2.0)
                self.logger.info(f"Received command: {cmd_dict}")
                self._execute_command(cmd_dict)
            except Empty:
                continue  # No commands in queue, continue listening
            except Exception as e:
                self.logger.error(f"Error in command processing loop: {e}")

    def start(self):
        """Start the unified interface."""
        self.is_running = True
        
        # Start command processing thread
        self.command_thread = threading.Thread(target=self._command_processing_loop)
        self.command_thread.daemon = True
        self.command_thread.start()
        
        # Start voice interface
        self.voice_interface.start()
        
        self.logger.info("Unified Spot Interface started")

    def stop(self):
        """Stop the unified interface."""
        self.is_running = False
        self.voice_interface.stop()
        
        if self.command_thread:
            self.command_thread.join()
        
        self.logger.info("Unified Spot Interface stopped")

        
def main():

    sdk = bosdyn.client.create_standard_sdk('UnifiedSpotInterface')
    robot = sdk.create_robot(HOSTNAME)
    auto_authenticate(robot)

    # Create the unified interface
    interface = UnifiedSpotInterface(
        robot=robot,
        map_path="/Users/shuk/Desktop/spot/practical-seminar-mobile-robotics/spotty/assets/maps/chair_graph_images",
        vector_db_path="/Users/shuk/Desktop/spot/practical-seminar-mobile-robotics/spotty/assets/rag_db/vector_db_chair",
        system_prompt=system_prompt_assistant,
        keyword_path="/Users/shuk/Desktop/spot/practical-seminar-mobile-robotics/spotty/assets/hey_spot_version_02/Hey-Spot_en_mac_v3_0_0.ppn"
    )

    # Start the interface
    interface.start()

    # Keep main thread running
    while True:
        pass

if __name__ == "__main__":
    main()