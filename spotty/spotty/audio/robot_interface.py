import os
import time
import queue
import threading
import wave
import pyaudio
import numpy as np
from pvrecorder import PvRecorder
import pvporcupine
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from dotenv import load_dotenv
from spotty.audio import system_prompt_robin, system_prompt_assistant
from spotty import KEYWORD_PATH
from openai import OpenAI
from datetime import datetime
from collections import deque
import json
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

load_dotenv()


@dataclass
class AudioConfig:
    """Audio configuration settings"""
    format: int = pyaudio.paInt16
    channels: int = 1
    rate: int = 48000   # Sample rate
    chunk: int = 2048
    temp_file: str = "output.wav"


class AudioManager:
    """Handles all audio-related operations"""
    def __init__(self, config: AudioConfig):
        self.config = config
        self.audio = pyaudio.PyAudio()
        self._init_pygame_mixer()

    def _init_pygame_mixer(self):
        """Initialize pygame mixer only if not already initialized"""
        if not pygame.mixer.get_init():
            try:
                pygame.mixer.init(44100, -16, 1, 1024)
            except Exception as e:
                print(f"Could not initialize pygame mixer: {e}")

    def record_audio(self, max_recording_time: int = 5, stop_queue: Optional[queue.Queue] = None) -> Optional[str]:
        """Record audio with progress indication"""
        print("\nRecording...")
        stream = self.audio.open(
            format=self.config.format,
            channels=self.config.channels,
            rate=self.config.rate,
            input=True,
            frames_per_buffer=self.config.chunk
        )
        
        frames = []
        start_time = time.time()
        
        try:
            while True:
                if time.time() - start_time > max_recording_time:
                    print("\nMax recording time reached.")
                    break
                
                if stop_queue and not stop_queue.empty():
                    stop_queue.get()
                    print("\nRecording stopped by signal.")
                    break
                
                data = stream.read(self.config.chunk, exception_on_overflow=False)
                frames.append(data)
                
        finally:
            stream.stop_stream()
            stream.close()
            
            if frames:
                with wave.open(self.config.temp_file, 'wb') as wf:
                    wf.setnchannels(self.config.channels)
                    wf.setsampwidth(self.audio.get_sample_size(self.config.format))
                    wf.setframerate(self.config.rate)
                    wf.writeframes(b''.join(frames))
                
                print(f"\nRecording saved. Duration: {time.time() - start_time:.2f} seconds")
                return self.config.temp_file
            
            return None
        
    def play_audio(self, audio_file: str) -> None:
        """Play audio file
        Args:
            audio_file (str): Path to audio file (.mp3, .wav, etc.)
        """
        self._init_pygame_mixer()
        try:
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        except Exception as e:
            print(f"Error playing audio: {e}") 
        finally:
            pygame.mixer.quit() # Clean up mixer resources

    def play_feedback_sound(self, sound_type: str = "start"):
        """Play audio feedback sounds"""
        self._init_pygame_mixer()
        try:
            frequency = 1500 if sound_type == "start" else 800
            samples = np.zeros((int(44100 * 0.1),))
            for i in range(len(samples)):
                samples[i] = np.sin(2.0 * np.pi * frequency * i / 44100)
            
            samples = (samples * 32767).astype(np.int16)
            sound = pygame.sndarray.make_sound(samples)
            sound.play()
            pygame.time.wait(100)
        except Exception as e:
            print(f"Error playing feedback sound: {e}")
    
    def cleanup(self):
        """Clean up audio resources"""
        # Close any open audio streams first TODO
        if self.audio:
            self.audio.terminate()
        if pygame.mixer.get_init():
            pygame.mixer.quit()
        


class ChatClient:
    """Client for interacting with OpenAI chat API"""
    def __init__(self, 
                 system_prompt: str = system_prompt_robin,
                 max_context_length: int = 10,  # Number of messages to keep
                 max_tokens: int = 4000):  # Max tokens for context window
        self.system_prompt = system_prompt
        self.max_context_length = max_context_length
        self.max_tokens = max_tokens
        self.client = OpenAI()
        self.chat_history = deque(maxlen=max_context_length)  # Use deque for automatic truncation

    def add_to_history(self, message: Dict[str, Any]) -> None:
        """Add a message to chat history with timestamp"""
        message["timestamp"] = datetime.now().isoformat()
        self.chat_history.append(message)

    def get_recent_context(self) -> List[Dict[str, Any]]:
        """Get recent chat history formatted for API context"""
        return list(self.chat_history)

    def save_history(self, filename: str) -> None:
        """Save chat history to file"""
        with open(filename, 'w') as f:
            json.dump(list(self.chat_history), f)

    def load_history(self, filename: str) -> None:
        """Load chat history from file"""
        try:
            with open(filename, 'r') as f:
                history = json.load(f)
                self.chat_history = deque(history, maxlen=self.max_context_length)
        except FileNotFoundError:
            print(f"No history file found at {filename}")

    def chat_completion(self, user_input: str) -> str:
        """Get chat completion from OpenAI API. Responses are already added to chat history
        No need to add them again outside this function.
        """
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.get_recent_context())
        messages.append({"role": "user", "content": user_input})
        
        # Add user message to history
        self.add_to_history({"role": "user", "content": user_input})
        
        try:
            response = self.client.chat.completions.create(
                model = "gpt-4o-mini",
                messages = messages,
                max_tokens = self.max_tokens
            )
            response = response.choices[0].message.content
            self.add_to_history({"role": "assistant", "content": response})
            
            return response
        except Exception as e:
            print(f"Error in chat completion: {e}")
            return "I'm sorry, I couldn't process that request."
    
    def speech_to_text(self, audio_file: str) -> str:
        """Transcribe audio using OpenAI Whisper API"""
        audio_file = open(audio_file, 'rb')
        transcription = self.client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
        return transcription.strip()
    
    def text_to_speech(self, text: str) -> None:
        """Convert text to speech using OpenAI API"""
        try:
            output_file = "tts_output.mp3"
            response = self.client.audio.speech.create(
                model="tts-1",
                voice = "shimmer",
                input = text
            )
            response.stream_to_file(output_file)
            return output_file
        except Exception as e:
            print(f"Error in text-to-speech: {e}")
            return None
    
    def clear_history(self):
        """Clear chat history"""
        self.chat_history.clear()
        print("Chat history cleared.")


@dataclass
class WakeWordConfig:
    """Configuration for wake word detection"""
    access_key: str
    keyword_path: str
    sensitivity: float = 0.5
    device_index: int = -1

class WakeWordDetector:
    """Handles wake word detection functionality"""
    def __init__(self, config: WakeWordConfig):
        self.config = config
        self.porcupine = None
        self.recorder = None
        self.detection_thread = None
        self.stop_event = threading.Event()
        self.wake_word_queue = queue.Queue()
        
        self._initialize_porcupine()
    
    def _initialize_porcupine(self):
        """Initialize Porcupine wake word engine"""
        try:
            self.porcupine = pvporcupine.create(
                access_key=self.config.access_key,
                keyword_paths=[self.config.keyword_path],
                sensitivities=[self.config.sensitivity]
            )
            self.recorder = PvRecorder(
                device_index=self.config.device_index,
                frame_length=self.porcupine.frame_length
            )
        except Exception as e:
            print(f"Error initializing Porcupine: {e}")
            raise

    def _detection_loop(self, callback: Optional[Callable] = None):
        """Main detection loop"""
        try:
            self.recorder.start()
            while not self.stop_event.is_set():
                pcm = self.recorder.read()
                result = self.porcupine.process(pcm)
                
                if result >= 0:
                    print("Wake word detected! Preparing for interaction...")
                    self.wake_word_queue.put(True)
                    if callback:
                        callback()
        
        except Exception as e:
            print(f"Wake word detection error: {e}")
        
        finally:
            self._cleanup_recorder()

    def _cleanup_recorder(self):
        """Clean up recorder resources"""
        try:
            if self.recorder:
                self.recorder.stop()
        except Exception as e:
            print(f"Error stopping recorder: {e}")

    def start(self, callback: Optional[Callable] = None):
        """Start wake word detection"""
        self.stop_event.clear()
        self.detection_thread = threading.Thread(
            target=self._detection_loop,
            args=(callback,)
        )
        self.detection_thread.daemon = True
        self.detection_thread.start()

    def stop(self):
        """Stop wake word detection and clean up"""
        self.stop_event.set()
        if self.detection_thread:
            self.detection_thread.join()
        self._cleanup_resources()

    def _cleanup_resources(self):
        """Clean up all resources"""
        try:
            if self.recorder:
                self.recorder.delete()
            if self.porcupine:
                self.porcupine.delete()
        except Exception as e:
            print(f"Cleanup error: {e}")

    def reset(self):
        """Reset detector for next interaction"""
        self._cleanup_resources()
        self._initialize_porcupine()

@dataclass
class VoiceInterfaceConfig:
    """Configuration for voice interface"""
    system_prompt: str = system_prompt_assistant
    max_context_length: int = 10
    max_recording_time: int = 6
    keyword_path: str = KEYWORD_PATH


class VoiceInterface:
    """Manages voice-based interaction with the robot"""
    def __init__(self, config: VoiceInterfaceConfig):
        self.config = config
        self._initialize_components()
        
    def _initialize_components(self) -> None:
        """Initialize all required components"""
        self.chat_client = ChatClient(
            system_prompt=self.config.system_prompt,
            max_context_length=self.config.max_context_length
        )
        self.audio_manager = AudioManager(AudioConfig())
        
        wake_word_config = WakeWordConfig(
            access_key=os.getenv("PICOVOICE_ACCESS_KEY"),
            keyword_path=self.config.keyword_path
        )
        self._init_wake_word_detector(wake_word_config)

    def _init_wake_word_detector(self, config: WakeWordConfig) -> None:
        """Initialize or reinitialize wake word detector"""
        self.wake_detector = WakeWordDetector(config)
        self.wake_detector.start(callback=self._handle_wake_word)

    def _handle_wake_word(self) -> None:
        """Handle wake word detection"""
        self.audio_manager.play_feedback_sound("start")

    def _process_voice_input(self) -> Optional[str]:
        """Record and transcribe voice input
        args: None
        returns: str or None. Transcribed text or None if no audio file is recorded. 
        """
        audio_file = self.audio_manager.record_audio(
            max_recording_time=self.config.max_recording_time
        )
        self.audio_manager.play_feedback_sound("stop")
        time.sleep(0.3)
        
        if audio_file:
            return self.chat_client.speech_to_text(audio_file)
        return None

    def _handle_conversation(self) -> None:
        """Handle a single conversation turn"""
        try:
            # Stop wake word detection during recording
            self.wake_detector.stop()
            time.sleep(0.5)  # Allow time for detection to stop
            
            user_input = self._process_voice_input()
            if not user_input:
                return
                
            print(f"\nYou said: {user_input}")
            
            response = self.chat_client.chat_completion(user_input)
            print(f"\nSpot: {response}")
            
            audio_file = self.chat_client.text_to_speech(response)
            if audio_file:
                self.audio_manager.play_audio(audio_file)
                time.sleep(0.3)
                
            # Allow time for audio resources to be released
            time.sleep(0.5)
            
        except Exception as e:
            print(f"\nError during conversation: {e}")
        finally:
            # Reinitialize wake word detection
            wake_word_config = WakeWordConfig(
                access_key=os.getenv("PICOVOICE_ACCESS_KEY"),
                keyword_path=self.config.keyword_path
            )
            self._init_wake_word_detector(wake_word_config)
            print("\nListening for wake word...")

    def run(self) -> None:
        """Run the voice interface"""
        print("\nVoice Interface Active")
        print("Say 'Hey Spot' to start a conversation")
        print("-" * 50)
        
        try:
            while True:
                self.wake_detector.wake_word_queue.get()
                self._handle_conversation()
                
        except KeyboardInterrupt:
            print("\nStopping voice interface...")
        finally:
            self.wake_detector.stop()
            time.sleep(0.5)
            print("Voice interface stopped.")

    def cleanup(self) -> None:
        """Clean up resources"""
        if hasattr(self, 'wake_detector'):
            self.wake_detector.stop()
            
        if hasattr(self, 'audio_manager'):
            self.audio_manager.cleanup()


# Command Stuff:
            
class CommandParser:
    """Parses assistant responses into structured commands"""
    @staticmethod
    def extract_command(response: str) -> Dict[str, Any]:
        """Extract command and parameters from response"""
        try:
            if "navigate_to(" in response:
                parts = response.split("navigate_to(")[1].split(")")[0].split(",")
                return {
                    "command": "navigate_to",
                    "parameters": {
                        "waypoint_id": parts[0].strip(),
                        "phrase": parts[1].strip() if len(parts) > 1 else ""
                    }
                }
            elif "say(" in response:
                phrase = response.split("say(")[1].split(")")[0].strip('"')
                return {
                    "command": "say",
                    "parameters": {"phrase": phrase}
                }
            elif "ask(" in response:
                question = response.split("ask(")[1].split(")")[0].strip('"')
                return {
                    "command": "ask",
                    "parameters": {"question": question}
                }
            elif "search(" in response:
                query = response.split("search(")[1].split(")")[0].strip('"')
                return {
                    "command": "search",
                    "parameters": {"query": query}
                }
            
            raise ValueError(f"Unknown command in response: {response}")
        except Exception as e:
            print(f"Error extracting command: {e}")
            return {
                "command": "say",
                "parameters": {"phrase": "I'm sorry, I couldn't process that command."}
            }
