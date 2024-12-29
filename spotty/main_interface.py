import os
from typing import Optional, List, Dict
import threading
import queue
from dataclasses import dataclass

from spotty.audio import WakeWordConversationAgent
from spotty.mapping import GraphNavInterface
from spotty.annotation import MultimodalRAGAnnotator
from dotenv import load_dotenv
from spotty import ASSETS_PATH
from spotty.audio import system_prompt_assistant
load_dotenv()
KEYWORD_PATH = os.path.join(ASSETS_PATH, "/hey_spot_version_02/Hey-Spot_en_mac_v3_0_0.ppn")

@dataclass
class SpotAction:
    """Represents an action for Spot to take"""
    action_type: str  # 'navigate', 'say', 'ask', 'search'
    params: Dict

class IntegratedSpotSystem:
    def __init__(
        self,
        robot,
        upload_path: str,
        vector_db_path: str,
        logger,
        snapshot_dir: str,
        graph_file: str
    ):
        # Initialize components
        self.robot = robot
        self.nav_interface = GraphNavInterface(robot, upload_path)
        self.voice_agent = WakeWordConversationAgent(
            access_key=os.getenv("PICOVOICE_ACCESS_KEY"),
            keyword_path=KEYWORD_PATH,
            transcription_method='openai',
            inference_method='openai',
            tts='openai'
        )
        self.rag = MultimodalRAGAnnotator(
            graph_file_path=graph_file,
            logger=logger,
            snapshot_dir=snapshot_dir,
            vector_db_path=vector_db_path
        )
        
        # Action queue for coordinating between voice commands and actions
        self.action_queue = queue.Queue()
        
        # Threading control
        self.running = False
        self.threads = []

    def start(self):
        """Start all system components"""
        self.running = True
        
        # Start voice agent
        self.voice_agent.start()
        
        # Start navigation interface
        self.nav_interface._list_graph_waypoint_and_edge_ids()
        
        # Start action processing thread
        action_thread = threading.Thread(target=self._process_actions)
        action_thread.daemon = True
        action_thread.start()
        self.threads.append(action_thread)

    def stop(self):
        """Stop all system components"""
        self.running = False
        self.voice_agent.stop()
        self.nav_interface.return_lease()
        
        for thread in self.threads:
            thread.join()

    def _process_actions(self):
        """Process actions from the queue"""
        while self.running:
            try:
                action = self.action_queue.get(timeout=1.0)
                self._execute_action(action)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing action: {e}")

    def _execute_action(self, action: SpotAction):
        """Execute a single action"""
        if action.action_type == "navigate":
            # Get waypoint ID and optional speech
            waypoint_id = action.params["waypoint_id"]
            speech = action.params.get("speech")
            
            if speech:
                self.voice_agent.text_to_speech(speech, tts="openai")
            
            # Navigate to waypoint
            self.nav_interface._navigate_to([waypoint_id])

        elif action.action_type == "say":
            # Text-to-speech
            self.voice_agent.text_to_speech(action.params["text"], tts="openai")

        elif action.action_type == "ask":
            # Ask question and get response
            self.voice_agent.text_to_speech(action.params["question"], tts="openai")
            # Record and process response
            audio_file = self.voice_agent.record_audio(max_recording_time=5)
            if audio_file:
                response = self.voice_agent.transcribe_audio_openai(audio_file)
                return response

        elif action.action_type == "search":
            # Search using RAG
            results = self.rag.query_location(
                action.params["query"],
                k=action.params.get("k", 3)
            )
            return results

    def _get_tools(self):
        """Get the list of available tools/functions for the model"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "navigate_to",
                    "description": "Navigate the robot to a specific location while speaking",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "waypoint_id": {
                                "type": "string",
                                "description": "The waypoint ID to navigate to"
                            },
                            "speech": {
                                "type": "string",
                                "description": "What to say while navigating"
                            }
                        },
                        "required": ["waypoint_id", "speech"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "say",
                    "description": "Make the robot say something using text-to-speech",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "The text to speak"
                            }
                        },
                        "required": ["text"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function", 
                "function": {
                    "name": "search",
                    "description": "Search the environment for locations or objects",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query"
                            },
                            "k": {
                                "type": "integer",
                                "description": "Number of results to return",
                                "default": 3
                            }
                        },
                        "required": ["query"],
                        "additionalProperties": False
                    }
                }
            }
        ]

    def process_voice_command(self, command: str) -> Optional[str]:
        """Process voice command using OpenAI function calling"""
        
        # Call GPT with the tools
        response = self.voice_agent.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt_assistant},
                {"role": "user", "content": command}
            ],
            tools=self._get_tools(),
            tool_choice="auto"
        )

        # Get assistant's response
        assistant_message = response.choices[0].message
        
        # Check if the model wants to call a function
        if assistant_message.tool_calls:
            tool_call = assistant_message.tool_calls[0]
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            # Create corresponding action
            action = SpotAction(
                action_type=function_name,
                params=function_args
            )
            
            # Add action to queue
            self.action_queue.put(action)
            
            # Return appropriate response
            if function_name == "search":
                results = self._execute_action(action)
                if results:
                    best_result = results[0]
                    nav_action = SpotAction(
                        action_type="navigate",
                        params={
                            "waypoint_id": best_result["waypoint_id"],
                            "speech": f"Taking you to {best_result['location']}."
                        }
                    )
                    self.action_queue.put(nav_action)
                    return f"Found {best_result['location']}. Navigating there now."
                return "Location not found"
            
            return f"Executing {function_name} command"
        
        # If no function call, just return the assistant's response
        return assistant_message.content

def main():
    # Initialize robot SDK connection
    from bosdyn.client import create_standard_sdk
    from bosdyn.client.util import authenticate
    
    sdk = create_standard_sdk('IntegratedSpotSystem')
    robot = sdk.create_robot('ROBOT_IP')
    authenticate(robot)
    
    # Initialize system
    system = IntegratedSpotSystem(
        robot=robot,
        upload_path="./maps",
        vector_db_path="./vector_db",
        logger=None,  # Add your logger here
        snapshot_dir="./snapshots",
        graph_file="./maps/graph"
    )
    
    try:
        system.start()
        print("System started. Press Ctrl+C to exit.")
        
        # Keep main thread running
        while True:
            pass
            
    except KeyboardInterrupt:
        print("\nStopping system...")
        system.stop()

if __name__ == "__main__":
    main()