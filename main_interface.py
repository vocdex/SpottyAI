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

from bosdyn.api import image_pb2
from bosdyn.client.image import build_image_request
import numpy as np
import cv2
from scipy import ndimage
import base64
import time
from openai import OpenAI


ROTATION_ANGLE = {
    'back_fisheye_image': 0,
    'frontleft_fisheye_image': -90,
    'frontright_fisheye_image': -90,
    'left_fisheye_image': 0,
    'right_fisheye_image': 180
}

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
        self.image_client = robot.ensure_client('image')
        self.current_images = {}
        self.image_thread = None
        self._init_image_fetching()
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _init_image_fetching(self):
        """Initialize image fetching thread"""
        self.logger.info("Starting image fetching thread...")
        
        image_sources = self.image_client.list_image_sources()
        image_sources_name = [source.name for source in image_sources]
        
        # Check if the required sources are available
        required_sources = ['frontright_fisheye_image', 'frontleft_fisheye_image']
        for source in required_sources:
            if source not in image_sources_name:
                self.logger.error(f"Required image source {source} not available")
                raise Exception(f"Required image source {source} not available")
        
        self.image_thread = threading.Thread(target=self._fetch_images_loop, daemon=True)
        self.image_thread.start()
        self.logger.info("Image fetching thread started")
    
    def _fetch_images_loop(self):
        """Continuously fetch images from the robot's cameras"""
        image_sources = ['frontright_fisheye_image', 'frontleft_fisheye_image']
        image_requests = [build_image_request(source, quality_percent=75) for source in image_sources]
        self.is_running = True
        while self.is_running:
            try:
                self.logger.debug("Fetching images...")
                image_responses = self.image_client.get_image(image_requests)
                self.logger.debug(f"Received {len(image_responses)} images")
                
                for image in image_responses:
                    dtype = np.uint8
                    img = np.frombuffer(image.shot.image.data, dtype=dtype)
                    
                    if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
                        img = img.reshape(
                            (image.shot.image.rows, image.shot.image.cols, 3)
                        )
                    else:
                        img = cv2.imdecode(img, -1)
                    
                    if image.source.name in ROTATION_ANGLE:
                        img = ndimage.rotate(img, ROTATION_ANGLE[image.source.name])
                    
                    self.current_images[image.source.name] = img
                    self.logger.debug(f"Stored image from {image.source.name}")

                time.sleep(0.5)

            except Exception as e:
                self.logger.error(f"Error fetching images: {str(e)}")
                continue


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
            
            self.audio_manager = AudioManager(AudioConfig())
            
            self.chat_client = ChatClient(system_prompt=system_prompt)
            
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
    
    def log_current_images(self):
        """Log the current images for debugging"""
        if not self.current_images:
            self.logger.warning("No images in self.current_images")
        else:
            for source, img in self.current_images.items():
                self.logger.info(f"Image from {source} has shape {img.shape}")

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
            
            self._parse_and_execute_command(response)
            
        except Exception as e:
            self.logger.error(f"Error in interaction: {e}")
            self._handle_speech("I encountered an error processing your request.")

    def _parse_and_execute_command(self, response: str):
        """Parse LLM response and execute corresponding command"""
        try:
            if "navigate_to(" in response:
                parts = response.split("navigate_to(")[1].split(")")[0].split(",")
                parts[0] = parts[0].strip('"')
                self._handle_navigation(parts[0].strip(), parts[1].strip() if len(parts) > 1 else None)
            
            elif "describe_scene(" in response:
                query = response.split("describe_scene(")[1].split(")")[0].strip('"')
                self._handle_vqa(query)
            elif "say(" in response:
                phrase = response.split("say(")[1].split(")")[0].strip('"')
                self._handle_speech(phrase)
            
            elif "ask(" in response:
                question = response.split("ask(")[1].split(")")[0].strip('"')
                self._handle_question(question)
            
            elif "search(" in response:
                query = response.split("search(")[1].split(")")[0].strip('"')
                self._handle_search(query)
            elif "sit(" in response:
                if self._sit_robot():
                    self._handle_speech("I have sat down and turned off my motors.")
                else:
                    self._handle_speech("I had trouble sitting down.")
                    
            elif "stand(" in response:
                if self._stand_robot():
                    self._handle_speech("I am now standing and ready to assist.")
                else:
                    self._handle_speech("I had trouble standing up.")
            else:
                self.logger.warning(f"Unknown command in response: {response}")
                self._handle_speech("I'm not sure how to handle that request.")
                
        except Exception as e:
            self.logger.error(f"Error parsing command: {e}")
            self._handle_speech("I had trouble understanding how to handle that request.")

    
    
    def _handle_vqa(self, query: str):
        """Handle visual question answering using GPT-4V-mini with unified scene understanding"""
        try:
            self.log_current_images()
            if not self.current_images:
                self._handle_speech("I don't have access to camera images")
                return
            elif not self.openai_client:
                self._handle_speech("I don't have access to the OpenAI vision model")

            messages = [
                {
                    "role": "system",
                    "content": """
                    You are assisting a robot in analyzing its environment through two camera feeds 
                    that show the left and right front views. These images together form a wider 
                    view of the scene. When describing what you see:
                    
                    1. Provide a single, coherent description that combines information from both views
                    2. Focus on spatial relationships between objects across both images
                    3. Avoid mentioning "left camera" or "right camera" unless specifically asked
                    4. Describe the scene as if you're looking at it from the robot's perspective
                    5. If you see the same object in both images, mention it only once
                    Be creative and have fun with your responses!
                    Be concise(2-3 sentences)
                    Remember to address the user's specific query in your response.
                    """
                }
            ]

            user_content = [{"type": "text", "text": query}]

            for source, img in self.current_images.items():
                try:
                    success, buffer = cv2.imencode('.jpg', img)
                    if success:
                        base64_image = base64.b64encode(buffer).decode('utf-8')
                        user_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        })
                except Exception as e:
                    self.logger.error(f"Error processing image from {source}: {str(e)}")
                    continue

            messages.append({
                "role": "user",
                "content": user_content
            })

            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages
                )
                
                vqa_response = response.choices[0].message.content
                self._handle_speech(vqa_response)
                
                self.chat_client.add_to_history({
                    "role": "user",
                    "content": f"[Visual Query] {query}"
                })
                self.chat_client.add_to_history({
                    "role": "assistant",
                    "content": vqa_response
                })
            except Exception as e:
                self.logger.error(f"Error calling vision model: {str(e)}")
                self._handle_speech("I had trouble analyzing the images. Please try again.")

        except Exception as e:
            self.logger.error(f"Error in VQA: {str(e)}")
            self._handle_speech("I encountered an error processing your visual query.")

    def _handle_navigation(self, waypoint_id: str, phrase: Optional[str] = None, search_query: Optional[str] = False):
        """Handle navigation to waypoint"""
        if phrase:
            self._handle_speech(phrase)
            time.sleep(5)
        
        # Execute navigation
        destination = (waypoint_id, None)
        is_successful = False
        print(f"Destination: {destination}")
        if search_query:
            is_successful=self.graph_nav._navigate_to(destination)
        else:
            is_successful, matching_waypoint_id= self.graph_nav._navigate_to_by_location(destination)
            waypoint_id = matching_waypoint_id
        
        if is_successful:
            self.state.prev_waypoint_id = self.state.waypoint_id
            self.state.waypoint_id = waypoint_id
            print(f"Current location: {self.state.location}")
            print(f"Previous location: {self.state.prev_location}")
            
            
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
            location = result['location']
            if location not in locations:
                locations[location] = result
        
        if len(locations) == 1:
            result = next(iter(locations.values()))
            self._handle_navigation(
                result["waypoint_id"],
                f"I found {query} in {result['location']}. Let me take you there.",
                search_query=True
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
                
            response = self.chat_client.speech_to_text(audio_file)
            
            try:
                # Try to match either number or location name
                chosen_location = None
                response = response.lower()
                
                for i, loc in enumerate(locations.keys()):
                    if str(i+1) in response:
                        chosen_location = loc
                        break
                
                if not chosen_location:
                    for loc in locations.keys():
                        if loc.lower() in response:
                            chosen_location = loc
                            break
                
                if chosen_location and chosen_location in locations:
                    result = locations[chosen_location]
                    self._handle_navigation(
                        result["waypoint_id"],
                        f"Taking you to {query} in {chosen_location}.",
                        search_query=True
                    )
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
            
            history_context = []
            for msg in self.chat_client.history:
                history_context.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            context_prompt = {
                "role": "system",
                "content": """Consider the conversation history and current context when responding.
                            Provide a natural follow-up that builds on previous interactions.
                            Remember to use exactly one function call in your response."""
            }
            
            self.chat_client.add_to_history({
                "role": "assistant",
                "content": question
            })
            self.chat_client.add_to_history({
                "role": "user",
                "content": user_response
            })
            
            messages = [
                context_prompt,
                *history_context,
                {"role": "assistant", "content": question},
                {"role": "user", "content": user_response}
            ]
            
            follow_up = self.chat_client.chat_completion(
                user_response,
                messages=messages  # Pass full message history for context
            )
            
            self.chat_client.add_to_history({
                "role": "assistant",
                "content": follow_up
            })
            
            if "say(" in follow_up:
                follow_up = follow_up.split("say(")[1].split(")")[0].strip('"')
            elif "ask(" in follow_up:
                follow_up = follow_up.split("ask(")[1].split(")")[0].strip('"')
                
            self._handle_speech(follow_up)
            
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
        if self.image_thread:
            self.image_thread.join()
        if self.command_thread:
            self.command_thread.join()
        self.graph_nav._on_quit()

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
        map_path=os.path.join(MAP_PATH, "chair_v3"),
        vector_db_path=os.path.join(RAG_DB_PATH, "chair_v3"),
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