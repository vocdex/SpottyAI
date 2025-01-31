import os
import logging
import threading
from queue import Queue
from typing import Dict, Any, Optional
from dataclasses import dataclass
from spotty.audio.robot_interface import WakeWordConfig, WakeWordDetector, AudioConfig, AudioManager, ChatClient
from spotty.mapping import GraphNavInterface
from spotty.annotation import MultimodalRAGAnnotator
from spotty.utils.common_utils import get_map_paths
from spotty import MAP_PATH, RAG_DB_PATH, KEYWORD_PATH


@dataclass
class SpotState:
    """Tracks the robot's current state"""
    waypoint_id: str
    location: str
    prev_location: str
    prev_waypoint_id: str
    what_it_sees: Optional[Dict[str, Any]] = None  # Store RAG annotations


class UnifiedSpotInterface:
    """Unified interface combining voice, RAG, and robot control"""
    
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
        # Set up logging
        logging.basicConfig(
            level=logging.DEBUG if debug else logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize state and queues
        self.state = SpotState(
            waypoint_id="",
            location="",
            prev_location="",
            prev_waypoint_id="",
        )
        self.command_queue = Queue()
        self.is_running = False
        
        # Initialize components
        self._init_graph_nav(robot, map_path)
        self._init_rag_system(map_path, vector_db_path)
        self._init_audio_components(system_prompt, keyword_path, audio_device_index)

    def _init_rag_system(self, map_path: str, vector_db_path: str):
        """Initialize RAG system"""
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
            self.logger.info("RAG system initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG system: {str(e)}")
            raise

    def _init_audio_components(self, system_prompt: str, keyword_path: str, audio_device_index: int):
        """Initialize audio and chat components"""
        try:
            self.logger.info("Initializing audio components...")
            
            # Initialize audio manager
            self.audio_manager = AudioManager(AudioConfig())
            
            # Initialize chat client
            self.chat_client = ChatClient(system_prompt=system_prompt)
            
            # Initialize wake word detector
            wake_config = WakeWordConfig(
                access_key=os.getenv("PICOVOICE_ACCESS_KEY"),
                keyword_path=keyword_path,
                device_index=audio_device_index
            )
            self.wake_detector = WakeWordDetector(wake_config)
            
            self.logger.info("Audio components initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize audio components: {str(e)}")
            raise

    def _handle_interaction(self):
        """Handle a single interaction turn"""
        try:
            # Record and transcribe audio
            audio_file = self.audio_manager.record_audio(max_recording_time=6)
            self.audio_manager.play_feedback_sound("stop")
            
            if not audio_file:
                return
            
            # Convert speech to text
            user_input = self.chat_client.speech_to_text(audio_file)
            print(f"\nUser: {user_input}")
            
            # Get LLM response with function calling
            response = self.chat_client.chat_completion(user_input)
            print(f"\nSpot's decision: {response}")
            
            # Parse and execute command
            self._parse_and_execute_command(response)
            
        except Exception as e:
            self.logger.error(f"Error in interaction: {e}")
            self._handle_speech("I encountered an error processing your request.")

    def _parse_and_execute_command(self, response: str):
        """Parse LLM response and execute corresponding command"""
        try:
            # Extract command and parameters from response
            if "navigate_to(" in response:
                parts = response.split("navigate_to(")[1].split(")")[0].split(",")
                self._handle_navigation(parts[0].strip(), parts[1].strip() if len(parts) > 1 else None)
            
            elif "say(" in response:
                phrase = response.split("say(")[1].split(")")[0].strip('"')
                self._handle_speech(phrase)
            
            elif "ask(" in response:
                question = response.split("ask(")[1].split(")")[0].strip('"')
                self._handle_question(question)
            
            elif "search(" in response:
                query = response.split("search(")[1].split(")")[0].strip('"')
                self._handle_search(query)
            
            else:
                self.logger.warning(f"Unknown command in response: {response}")
                self._handle_speech("I'm not sure how to handle that request.")
                
        except Exception as e:
            self.logger.error(f"Error parsing command: {e}")
            self._handle_speech("I had trouble understanding how to handle that request.")

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

    def _handle_navigation(self, waypoint_id: str, phrase: Optional[str] = None):
        """Handle navigation to waypoint"""
        if phrase:
            self._handle_speech(phrase)
            
        # Execute navigation
        destination = (waypoint_id, None)
        if self.graph_nav._navigate_to(destination):
            # Update state after successful navigation
            self.state.prev_waypoint_id = self.state.waypoint_id
            self.state.waypoint_id = waypoint_id
            
            # Get waypoint annotations from RAG
            annotations = self.rag_system.get_waypoint_annotations(waypoint_id)
            if annotations:
                self.state.location = annotations.get('location', '')
                self.state.what_it_sees = annotations
                self._handle_speech(f"Arrived at {self.state.location}")
            else:
                self._handle_speech("Arrived at destination")
        else:
            self._handle_speech("I was unable to reach the destination")

    def _handle_speech(self, text: str):
        """Handle text-to-speech output"""
        audio_file = self.chat_client.text_to_speech(text)
        if audio_file:
            self.audio_manager.play_audio(audio_file)

    def _handle_search(self, query: str):
        """Handle environment search using RAG"""
        results = self.rag_system.query_location(query, k=3) # Get top 3 results
        # Get the first result and navigate to it
        if results:
            result = results[0]
            self._handle_navigation(
                result["waypoint_id"],
                f"I found what you're looking for at {result['location']}. Let me take you there."
            )
        else:
            self._handle_speech("I couldn't find anything matching your search.")

    def _handle_question(self, question: str):
        """Handle interactive questions"""
        self._handle_speech(question)
        
        # Record and process response
        audio_file = self.audio_manager.record_audio(max_recording_time=6)
        if audio_file:
            response = self.chat_client.speech_to_text(audio_file)
            
            self.chat_client.add_to_history({
                "role": "assistant",
                "content": question
            })
            self.chat_client.add_to_history({
                "role": "user",
                "content": response
            })
            
            # Get follow-up response
            follow_up = self.chat_client.chat_completion(response)
            self._handle_speech(follow_up)

    def _command_processing_loop(self):
        """Main loop for processing wake word detection"""
        self.logger.info("Starting command processing loop")
        
        def wake_word_callback():
            self.audio_manager.play_feedback_sound("start")
            self._handle_interaction()
        
        self.wake_detector.start(callback=wake_word_callback)
        
        while self.is_running:
            try:
                self.wake_detector.wake_word_queue.get(timeout=1.0)
            except:
                continue

    def start(self):
        """Start the unified interface"""
        self.is_running = True
        self.command_thread = threading.Thread(target=self._command_processing_loop)
        self.command_thread.daemon = True
        self.command_thread.start()
        self.logger.info("Unified interface started")

    def stop(self):
        """Stop the unified interface"""
        self.is_running = False
        self.wake_detector.stop()
        self.audio_manager.cleanup()
        if self.command_thread:
            self.command_thread.join()
        self.logger.info("Unified interface stopped")

def main():
    """Main entry point"""
    import bosdyn.client
    from spotty.utils.robot_utils import auto_authenticate, HOSTNAME
    from spotty.audio import system_prompt_assistant
    
    # Initialize robot
    sdk = bosdyn.client.create_standard_sdk('UnifiedSpotInterface')
    robot = sdk.create_robot(HOSTNAME)
    auto_authenticate(robot)
    
    interface = UnifiedSpotInterface(
        robot=robot,
        map_path=os.path.join(MAP_PATH, "chair_v2"),
        vector_db_path=os.path.join(RAG_DB_PATH, "chair_v2"),
        system_prompt=system_prompt_assistant,
        keyword_path=KEYWORD_PATH,
    )
    try:
        interface.start()
        while True:
            pass
    except KeyboardInterrupt:
        interface.stop()

if __name__ == "__main__":
    main()