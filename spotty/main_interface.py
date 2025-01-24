import os
import logging
import threading
from queue import Queue, Empty
from typing import Dict, Any
from dataclasses import dataclass
import bosdyn.client
from dotenv import load_dotenv

from spotty.mapping import GraphNavInterface
from spotty.annotation import MultimodalRAGAnnotator
from spotty.utils.common_utils import get_map_paths
from spotty.utils.robot_utils import auto_authenticate, HOSTNAME
from spotty import MAP_PATH, RAG_DB_PATH, KEYWORD_PATH

from spotty.audio import VoiceAssistant

os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

@dataclass
class SpotState:
    waypoint_id: str
    location: str
    prev_location: str
    prev_waypoint_id: str
    what_it_sees: str


class UnifiedSpotInterface:
    """Unified interface for controlling Spot robot with voice commands"""
    
    def __init__(
        self,
        robot,
        map_path: str,
        vector_db_path: str,
        system_prompt: str,
        keyword_path: str,
        audio_device_index: int = -1,
        debug: bool = False
    ):
        """Initialize the unified interface"""
        # Set up logging
        logging.basicConfig(
            level=logging.DEBUG if debug else logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Unified Spot Interface...")
        
        # Initialize command queue
        self.command_queue = Queue()
        self.is_running = False
        
        # Initialize components
        self._init_graph_nav(robot, map_path)
        self._init_rag_system(map_path, vector_db_path)
        self._init_voice_interface(system_prompt, keyword_path, audio_device_index)
        
    def _init_graph_nav(self, robot, map_path: str):
        """Initialize the GraphNav component"""
        try:
            self.logger.info("Initializing GraphNav...")
            self.graph_nav = GraphNavInterface(robot, map_path)
            self.graph_nav._initialize_map(maybe_clear=False)
            self.logger.info("GraphNav initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize GraphNav: {str(e)}")
            raise

    def _init_rag_system(self, map_path: str, vector_db_path: str):
        """Initialize the RAG system"""
        try:
            self.logger.info("Initializing RAG system...")
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

    def _init_voice_interface(self, system_prompt: str, keyword_path: str, audio_device_index: int):
        """Initialize the voice interface"""
        try:
            self.logger.info("Initializing Voice Interface...")
            picovoice_key = os.getenv("PICOVOICE_ACCESS_KEY")
            openai_key = os.getenv("OPENAI_API_KEY")
            
            if not all([picovoice_key, openai_key]):
                raise ValueError("Required API keys not set in environment variables")
            
            self.voice_interface = VoiceAssistant(
                system_prompt=system_prompt,
                picovoice_access_key=picovoice_key,
                openai_api_key=openai_key,
                keyword_path=keyword_path,
                audio_device_index=audio_device_index
            )
            self.voice_interface.command_queue = self.command_queue
            self.logger.info("Voice Interface initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Voice Interface: {str(e)}")
            raise

    def _execute_command(self, cmd_dict: Dict[str, Any]):
        """Execute the parsed command"""
        command = cmd_dict["command"]
        params = cmd_dict["parameters"]
        
        try:
            if command == "navigate_to":
                self._handle_navigation(params)
            elif command == "say":
                self._handle_speech(params["phrase"])
            elif command == "search":
                self._handle_search(params["query"])
            elif command == "ask":
                self._handle_question(params["question"])
            else:
                self.logger.warning(f"Unknown command: {command}")
                self._handle_speech("I don't understand that command.")
        except Exception as e:
            self.logger.error(f"Error executing command: {e}")
            self._handle_speech(f"I encountered an error: {str(e)}")

    def _handle_navigation(self, params: Dict[str, Any]):
        """Handle navigation commands"""
        if "waypoint_id" not in params:
            raise ValueError("Waypoint ID not provided")
        
        if "phrase" in params and params["phrase"]:
            self._handle_speech(params["phrase"])
            
        destination = (params["waypoint_id"], None)
        if self.graph_nav._navigate_to(destination):
            self._handle_speech("You have arrived at your destination")
        else:
            self._handle_speech("Failed to reach the destination")

    def _handle_speech(self, phrase: str):
        """Handle speech output"""
        self.voice_interface.openai_client.text_to_speech(phrase)

    def _handle_search(self, query: str):
        """Handle search commands"""
        results = self.rag_system.query_location(query, k=3, distance_threshold=2.0)
        
        if results:
            closest_waypoint = results[0]["waypoint_id"]
            description = results[0]["description"]
            location = results[0]["location"]
            self.command_queue.put({
                "command": "navigate_to",
                "parameters": {
                    "waypoint_id": closest_waypoint,
                    "phrase": "I found what you're looking for. Let me take you there, we're going to " + location
                }
            })
        else:
            self._handle_speech("I couldn't find anything matching your search.")

    def _handle_question(self, question: str):
        """Handle questions by recording both the question and response in chat history"""
        self._handle_speech(question)
        import time
        time.sleep(1)

        audio_file = self.voice_interface.audio_manager.record_audio(max_recording_time=6)
        if audio_file:
            response_text = self.voice_interface.openai_client.transcribe_audio(audio_file)

            # Add both question and response to chat history
            self.voice_interface.openai_client.chat_history.extend([
                {"role": "assistant", "content": question},
                {"role": "user", "content": response_text}
            ])
            # Get AI's follow-up response using the updated chat history
            follow_up = self.voice_interface.openai_client.chat_completion(
                response_text,
                self.voice_interface.system_prompt
            )
            
            # Speak the follow-up response
            self._handle_speech(follow_up)
        else:
            self._handle_speech("I couldn't understand your response. Please try again.")
            


    def _command_processing_loop(self):
        """Main loop for processing commands"""
        self.logger.info("Starting command processing loop")
        while self.is_running:
            try:
                cmd_dict = self.command_queue.get(timeout=2.0)
                self.logger.info(f"Received command: {cmd_dict}")
                self._execute_command(cmd_dict)
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in command processing loop: {e}")

    def start(self):
        """Start the unified interface"""
        self.is_running = True
        
        # Start command processing thread
        self.command_thread = threading.Thread(target=self._command_processing_loop)
        self.command_thread.daemon = True
        self.command_thread.start()
        
        # Start voice interface
        self.voice_interface.start()
        self.logger.info("Unified Spot Interface started")

    def stop(self):
        """Stop the unified interface"""
        self.is_running = False
        self.voice_interface.stop()
        
        if self.command_thread:
            self.command_thread.join()
            
        self.logger.info("Unified Spot Interface stopped")


def main():
    """Main entry point"""
    sdk = bosdyn.client.create_standard_sdk('UnifiedSpotInterface')
    robot = sdk.create_robot(HOSTNAME)
    auto_authenticate(robot)
    from spotty.audio import system_prompt_assistant
    interface = UnifiedSpotInterface(
        robot=robot,
        map_path=os.path.join(MAP_PATH, "chair_graph_images"),
        vector_db_path=os.path.join(RAG_DB_PATH, "vector_db_chair_v1"),
        system_prompt=system_prompt_assistant,
        keyword_path=os.path.join(KEYWORD_PATH),
    )
    try:
        interface.start()
        while True:
            pass
    except KeyboardInterrupt:
        interface.stop()


if __name__ == "__main__":
    main()