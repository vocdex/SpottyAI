import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

import cv2
from openai import OpenAI

from spotty import KEYWORD_PATH, MAP_PATH, RAG_DB_PATH
from spotty.annotation import MultimodalRAGAnnotator
from spotty.audio.command_parser import CommandParser
from spotty.audio.robot_interface import (
    AudioConfig,
    AudioManager,
    ChatClient,
    WakeWordConfig,
    WakeWordDetector,
)
from spotty.config.robot_config import RobotConfig
from spotty.mapping import GraphNavInterface
from spotty.mapping.navigator_interface import NavigatorInterface
from spotty.utils.common_utils import get_map_paths
from spotty.utils.state_manager import SpotState
from spotty.vision.image_processor import ImageProcessor
from spotty.vision.vqa_handler import VQAHandler


class UnifiedSpotInterface:
    """Unified interface combining voice, RAG, and robot control"""

    def __init__(
        self,
        # robot,
        config: RobotConfig,
    ):
        logging.basicConfig(
            level=logging.DEBUG if config.debug else logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)
        self.config = config

        self.state = SpotState()

        # self.robot = robot
        # self.image_client = robot.ensure_client("image")
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # self._init_graph_nav(robot, config.map_path)
        self._init_rag_system(config.map_path, config.vector_db_path)
        self._init_audio_components(config.system_prompt, config.wake_word_config)

        # # self.image_processor = ImageProcessor(
        #     self.image_client,
        #     self.logger,
        #     config.vision_config.rotation_angles
        # )
        # self.vqa_handler = VQAHandler(self.openai_client, self.logger)
        self.command_parser = CommandParser()

        self.navigator = NavigatorInterface(self.graph_nav, self.state, self.logger)
        self.navigator.set_rag_system(self.rag_system)

        self._init_image_fetching(config.vision_config.required_sources)

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
        """Initialize RAG system"""
        try:
            self.logger.info("Initializing RAG system...")
            graph_file_path, snapshot_dir, _ = get_map_paths(map_path)
            self.rag_system = MultimodalRAGAnnotator(
                graph_file_path=graph_file_path,
                logger=self.logger,
                snapshot_dir=snapshot_dir,
                vector_db_path=vector_db_path,
                load_clip=False,
            )
            self.logger.info("RAG system initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG system: {str(e)}")
            raise

    def _init_audio_components(self, system_prompt: str, wake_word_config: WakeWordConfig):
        """Initialize audio and chat components"""
        try:
            self.logger.info("Initializing audio components...")

            self.audio_manager = AudioManager(AudioConfig())
            self.chat_client = ChatClient(system_prompt=system_prompt)
            self.wake_detector = WakeWordDetector(wake_word_config)

            self.logger.info("Audio components initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize audio components: {str(e)}")
            raise

    def _init_image_fetching(self, required_sources: List[str]):
        """Initialize image fetching thread"""
        self.logger.info("Starting image fetching thread...")

        image_sources = self.image_client.list_image_sources()
        image_sources_name = [source.name for source in image_sources]

        for source in required_sources:
            if source not in image_sources_name:
                self.logger.error(f"Required image source {source} not available")
                raise Exception(f"Required image source {source} not available")

        self.image_thread = threading.Thread(target=self._fetch_images_loop, daemon=True)
        self.image_thread.start()
        self.logger.info("Image fetching thread started")

    def _fetch_images_loop(self):
        """Continuously fetch images from the robot's cameras"""
        image_sources = ["frontright_fisheye_image", "frontleft_fisheye_image"]
        self.state.is_running = True

        while self.state.is_running:
            try:
                images = self.image_processor.fetch_images(image_sources)
                self.state.current_images = images
                time.sleep(0.5)
            except Exception as e:
                self.logger.error(f"Error in image fetching loop: {str(e)}")
                time.sleep(1.0)  # Sleep longer on error

    def _sit_robot(self):
        """Command the robot to sit using GraphNav interface."""
        return self.graph_nav.sit()

    def _stand_robot(self):
        """Command the robot to stand using GraphNav interface."""
        return self.graph_nav.stand()

    def _handle_interaction(self):
        """Handle a single interaction turn"""
        try:
            audio_file = self.audio_manager.record_audio(max_recording_time=6)
            self.audio_manager.play_feedback_sound("stop")

            if not audio_file:
                return

            user_input = self.chat_client.speech_to_text(audio_file)
            print(f"\nUser: {user_input}")

            response = self.chat_client.chat_completion(user_input)
            print(f"\nSpot's decision: {response}")

            command_data = self.command_parser.extract_command(response)
            self._execute_command(command_data)

        except Exception as e:
            self.logger.error(f"Error in interaction: {e}")
            self._handle_speech("I encountered an error processing your request.")

    def _execute_command(self, command_data: Dict[str, Any]):
        """Execute a command based on parsed command data"""
        command = command_data.get("command")
        params = command_data.get("parameters", {})

        try:
            if command == "navigate_to":
                self._handle_navigation(params.get("waypoint_id"), params.get("phrase"))
            elif command == "describe_scene":
                self._handle_vqa(params.get("query"))
            elif command == "say":
                self._handle_speech(params.get("phrase"))
            elif command == "ask":
                self._handle_question(params.get("question"))
            elif command == "search":
                self._handle_search(params.get("query"))
            elif command == "sit":
                if self._sit_robot():
                    self._handle_speech("I have sat down and turned off my motors.")
                else:
                    self._handle_speech("I had trouble sitting down.")
            elif command == "stand":
                if self._stand_robot():
                    self._handle_speech("I am now standing and ready to assist.")
                else:
                    self._handle_speech("I had trouble standing up.")
            else:
                self.logger.warning(f"Unknown command: {command}")
                self._handle_speech("I'm not sure how to handle that request.")
        except Exception as e:
            self.logger.error(f"Error executing command {command}: {e}")
            self._handle_speech("I had trouble executing that command.")

    def _handle_navigation(self, waypoint_id: str, phrase: Optional[str] = None):
        """Handle navigation to waypoint or location"""
        if phrase:
            self._handle_speech(phrase)
            time.sleep(4)  # Give time for speech to complete

        destination = (waypoint_id, None)
        is_successful = False
        print(f"Destination: {destination}")

        if waypoint_id in self.rag_system.get_vector_store_info().get("total_documents", []):
            # This is likely a waypoint ID
            # is_successful = self.graph_nav._navigate_to(destination)
            self._handle_speech("I am navigating to the specified location.")
        else:
            # This is likely a location name, try to find matching waypoint
            # is_successful, matching_waypoint_id = self.graph_nav._navigate_to_by_location(destination)
            # waypoint_id = matching_waypoint_id
            self._handle_speech("I am navigating to the specified location.")

        if is_successful:
            self.state.prev_waypoint_id = self.state.waypoint_id
            self.state.waypoint_id = waypoint_id
            print(f"Current location: {self.state.location}")
            print(f"Previous location: {self.state.prev_location}")

            annotations = self.rag_system.get_waypoint_annotations(waypoint_id)
            if annotations:
                self.state.location = annotations.get("location", "")
                self.state.what_it_sees = annotations
                self._handle_speech(f"Arrived at {self.state.location}")
            else:
                self._handle_speech("Arrived at destination")
        else:
            self._handle_speech("I was unable to reach the destination")

    def _handle_vqa(self, query: str):
        """Handle visual question answering using VQA handler"""
        try:
            if not self.state.current_images:
                self._handle_speech("I don't have access to camera images")
                return

            # Prepare images for VQA processing
            image_dict = {}
            for source, img in self.state.current_images.items():
                try:
                    success, buffer = cv2.imencode(".jpg", img)
                    if success:
                        image_dict[source] = buffer.tobytes()
                except Exception as e:
                    self.logger.error(f"Error processing image from {source}: {str(e)}")

            vqa_response = self.vqa_handler.process_query(query, image_dict)
            self._handle_speech(vqa_response)

            self.chat_client.add_to_history({"role": "user", "content": f"[Visual Query] {query}"})
            self.chat_client.add_to_history({"role": "assistant", "content": vqa_response})

        except Exception as e:
            self.logger.error(f"Error in VQA: {str(e)}")
            self._handle_speech("I encountered an error processing your visual query.")

    def _handle_speech(self, text: str):
        """Handle text-to-speech output"""
        audio_file = self.chat_client.text_to_speech(text)
        if audio_file:
            self.audio_manager.play_audio(audio_file)
        else:
            self.logger.error("Failed to generate audio file")

    def _handle_search(self, query: str):
        """Handle environment search using RAG with location-based disambiguation"""
        enhanced_query = "Where do you see a " + query + "?"
        results = self.rag_system.query_location(enhanced_query, k=5)  # Increased k to get more potential locations

        if not results:
            self._handle_speech("I couldn't find anything matching your search.")
            return

        # Group results by location
        locations = {}
        for result in results:
            location = result["location"]
            if location not in locations:
                locations[location] = result

        if len(locations) == 1:
            result = next(iter(locations.values()))
            self._handle_navigation(
                result["waypoint_id"],
                f"I found {query} in {result['location']}. Let me take you there.",
            )
        else:
            # Multiple locations found, ask user for preference
            location_list = ", ".join([f"{i+1}: {loc}" for i, loc in enumerate(locations.keys())])
            question = f"I found {query} in multiple locations: {location_list}. Which location would you prefer?"
            self._handle_speech(question)

            # Record and process user's response
            audio_file = self.audio_manager.record_audio(max_recording_time=6)
            if not audio_file:
                return

            user_response = self.chat_client.speech_to_text(audio_file)

            try:
                # Use the chat completion API to interpret the user's choice
                system_prompt = f"""You are helping identify which location a user has chosen from a list.
                                Available locations: {', '.join(locations.keys())}
                                The user's response is: "{user_response}"
                                Respond ONLY with the exact name of the chosen location from the available list, or respond with "unknown" if the choice is unclear."""

                location_choice = self.chat_client.chat_completion(user_response, messages=[{"role": "system", "content": system_prompt}])

                location_choice = location_choice.strip().lower()
                for loc in locations.keys():
                    if loc.lower() in location_choice:
                        location_choice = loc
                        break

                if location_choice in locations:
                    result = locations[location_choice]
                    self._handle_navigation(result["waypoint_id"], f"Taking you to {query} in {location_choice}.")
                else:
                    self._handle_speech("I'm sorry, I couldn't understand which location you prefer. Please try your search again.")

            except Exception as e:
                self.logger.error(f"Error processing location choice: {e}")
                self._handle_speech("I had trouble understanding your choice. Please try your search again.")

    def _handle_question(self, question: str):
        """Handle interactive questions with improved context awareness"""
        try:
            self._handle_speech(question)

            audio_file = self.audio_manager.record_audio(max_recording_time=6)
            if not audio_file:
                return

            user_response = self.chat_client.speech_to_text(audio_file)

            self.chat_client.add_to_history({"role": "assistant", "content": question})
            self.chat_client.add_to_history({"role": "user", "content": user_response})

            follow_up = self.chat_client.chat_completion(user_response)

            self.chat_client.add_to_history({"role": "assistant", "content": follow_up})

            command_data = self.command_parser.extract_command(follow_up)

            if command_data["command"] == "say":
                self._handle_speech(command_data["parameters"]["phrase"])
            else:
                # Execute any other command
                self._execute_command(command_data)

        except Exception as e:
            self.logger.error(f"Error in _handle_question: {str(e)}")
            self._handle_speech("I encountered an error processing your response.")

    def _command_processing_loop(self):
        """Main loop for processing wake word detection"""
        self.logger.info("Starting command processing loop")

        def wake_word_callback():
            self.audio_manager.play_feedback_sound("start")
            self._handle_interaction()

        self.wake_detector.start(callback=wake_word_callback)

        while self.state.is_running:
            try:
                self.wake_detector.wake_word_queue.get(timeout=1.0)
            except Exception as e:
                self.logger.debug(f"Queue get timeout: {e}")
                continue

    def start(self):
        """Start the unified interface"""
        self.state.is_running = True
        self.command_thread = threading.Thread(target=self._command_processing_loop)
        self.command_thread.daemon = True
        self.command_thread.start()
        self.logger.info("Unified interface started")

    def stop(self):
        """Stop the unified interface"""
        self.state.is_running = False
        self.wake_detector.stop()
        self.audio_manager.cleanup()

        if hasattr(self, "image_thread") and self.image_thread:
            self.image_thread.join(timeout=2.0)

        if hasattr(self, "command_thread") and self.command_thread:
            self.command_thread.join(timeout=2.0)

        self.graph_nav._on_quit()
        self.logger.info("Unified interface stopped")


def main():
    """Main entry point"""
    # import bosdyn.client

    from spotty.audio import system_prompt_assistant
    from spotty.config.robot_config import RobotConfig, VisionConfig, WakeWordConfig
    from spotty.utils.robot_utils import HOSTNAME, auto_authenticate

    # Initialize robot
    # sdk = bosdyn.client.create_standard_sdk("UnifiedSpotInterface")
    # robot = sdk.create_robot(HOSTNAME)
    # auto_authenticate(robot)

    config = RobotConfig(
        wake_word_config=WakeWordConfig(access_key=os.getenv("PICOVOICE_ACCESS_KEY"), keyword_path=KEYWORD_PATH),
        vision_config=VisionConfig(),
        system_prompt=system_prompt_assistant,
        map_path=os.path.join(MAP_PATH, "chair_v3"),
        vector_db_path=os.path.join(RAG_DB_PATH, "chair_v3"),
    )

    interface = UnifiedSpotInterface(
        # robot=robot,
        config=config
    )

    try:
        interface.start()
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        interface.stop()
    except Exception as e:
        logging.error(f"Error in main thread: {e}")
        interface.stop()


if __name__ == "__main__":
    main()
