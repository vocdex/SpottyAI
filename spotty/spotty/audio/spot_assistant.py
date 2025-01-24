import os
import time
import queue
import threading
import requests
import wave
import pyaudio
import pygame
import numpy as np
from pvrecorder import PvRecorder
import pvporcupine
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from dotenv import load_dotenv


load_dotenv()

@dataclass
class AudioConfig:
    """Audio configuration settings"""
    format: int = pyaudio.paInt16
    channels: int = 1
    rate: int = 16000
    chunk: int = 1024
    temp_file: str = "output.wav"

class AudioManager:
    """Handles all audio-related operations"""
    def __init__(self, config: AudioConfig):
        self.config = config
        self.audio = pyaudio.PyAudio()
        self._init_pygame_mixer()

    def _init_pygame_mixer(self):
        """Initialize pygame mixer for audio feedback"""
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

    def play_feedback_sound(self, sound_type: str = "start"):
        """Play audio feedback sounds"""
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

@dataclass
class ChatMessage:
    role: str
    content: str
    metadata: Dict = field(default_factory=dict)

class OpenAIClient:
    """Handles all OpenAI API interactions"""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.chat_history : List[ChatMessage] = []
    
    def add_to_history(self, role: str, content: str, metadata: Dict = None):
        """Add a message to chat history with optional metadata"""
        self.chat_history.append(ChatMessage(
            role=role,
            content=content,
            metadata=metadata or {}
        ))
        
    def get_recent_context(self, n_messages: int = 5) -> List[Dict]:
        """Get the n most recent messages for context"""
        recent_messages = self.chat_history[-n_messages:] if self.chat_history else []
        return [{"role": msg.role if isinstance(msg, ChatMessage) else msg["role"],
                "content": msg.content if isinstance(msg, ChatMessage) else msg["content"]} 
                for msg in recent_messages]
    
    def clear_history(self):
        """Clear the chat history"""
        self.chat_history.clear()

    def transcribe_audio(self, file_path: str) -> str:
        """Transcribe audio using OpenAI Whisper API"""
        url = "https://api.openai.com/v1/audio/transcriptions"
        
        with open(file_path, "rb") as audio_file:
            files = {
                "file": (file_path, audio_file, "audio/wav"),
                "model": (None, "whisper-1"),
                "response_format": (None, "text"),
            }
            response = requests.post(url, headers=self.headers, files=files)
            response.raise_for_status()
            return response.text.strip()

    def chat_completion(self, user_input: str, system_prompt: str) -> str:
        """Get chat completion from OpenAI API"""
        url = "https://api.openai.com/v1/chat/completions"
        headers = {**self.headers, "Content-Type": "application/json"}
        try:
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(self.get_recent_context())
            messages.append({"role": "user", "content": user_input})
            
            
            payload = {"model": "gpt-4o-mini", 
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 100}
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            chat_response = response.json()["choices"][0]["message"]["content"]
            from datetime import datetime
            timestamp = datetime.now().isoformat()
            self.chat_history.append([
                {"role": "user", "content": user_input,
                "metadata": {"timestamp": timestamp,"type": "voice_input"}},
            ])
            self.chat_history.append([
                {"role": "assistant", "content": chat_response,
                "metadata": {"timestamp": timestamp,"type": "ai_response"}},
            ])
            return chat_response
        except Exception as e:
            print(f"Error in chat completion: {e}")
            return "I'm sorry, I couldn't process that command."

    def text_to_speech(self, text: str) -> Optional[str]:
        """Convert text to speech using OpenAI TTS API"""
        url = "https://api.openai.com/v1/audio/speech"
        headers = {**self.headers, "Content-Type": "application/json"}
        
        payload = {
            "model": "tts-1",
            "input": text,
            "voice": "shimmer"
        }
        
        output_file = "tts_output.mp3"
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            with open(output_file, "wb") as audio_file:
                audio_file.write(response.content)
            
            pygame.mixer.music.load(output_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
            return output_file
        except Exception as e:
            print(f"TTS error: {e}")
            return None

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

class VoiceAssistant:
    """Main voice assistant class that coordinates all components"""
    def __init__(self, 
                 system_prompt: str,
                 picovoice_access_key: str,
                 openai_api_key: str,
                 keyword_path: str,
                 audio_device_index: int = -1):
        
        self.system_prompt = system_prompt
        self.audio_config = AudioConfig()
        self.audio_manager = AudioManager(self.audio_config)
        self.openai_client = OpenAIClient(openai_api_key)
        self.command_parser = CommandParser()
        
        # Wake word detection setup
        self.porcupine = pvporcupine.create(
            access_key=picovoice_access_key,
            keyword_paths=[keyword_path],
            sensitivities=[0.5]
        )
        self.recorder = PvRecorder(
            device_index=audio_device_index,
            frame_length=self.porcupine.frame_length
        )
        
        # Threading and control
        self.wake_word_queue = queue.Queue()
        self.stop_detection_event = threading.Event()
        self.command_queue = None
        
    def _wake_word_detection_loop(self):
        """Continuous loop for detecting wake words"""
        try:
            self.recorder.start()
            print("Listening for wake word...")
            
            while not self.stop_detection_event.is_set():
                pcm = self.recorder.read()
                result = self.porcupine.process(pcm)
                
                if result >= 0:
                    print("Wake word detected! Preparing for interaction...")
                    self.wake_word_queue.put(True)
                    time.sleep(0.1)
        
        except Exception as e:
            print(f"Wake word detection error: {e}")
        
        finally:
            try:
                self.recorder.stop()
            except Exception as e:
                print(f"Error stopping recorder: {e}")

    def _conversation_loop(self):
        """Main conversation loop"""
        stop_recording_queue = queue.Queue()
        
        while not self.stop_detection_event.is_set():
            try:
                self.wake_word_queue.get(timeout=1)
                
                # Handle conversation flow
                self.audio_manager.play_feedback_sound("start")
                self.stop_detection_event.set()
                time.sleep(0.5)
                
                audio_file = self.audio_manager.record_audio(
                    max_recording_time=5,
                    stop_queue=stop_recording_queue
                )
                
                if audio_file:
                    self.audio_manager.play_feedback_sound("end")
                    print("\nTranscribing...")
                    
                    transcribed_text = self.openai_client.transcribe_audio(audio_file)
                    print(f">> You said: {transcribed_text}")
                    
                    print("Thinking...")
                    chat_response = self.openai_client.chat_completion(
                        transcribed_text,
                        self.system_prompt
                    )
                    print(f"🤖 Assistant: {chat_response}")
                    
                    if self.command_queue is not None:
                        cmd_dict = self.command_parser.extract_command(chat_response)
                        print(f"Queueing command: {cmd_dict}")
                        self.command_queue.put(cmd_dict)
                    
                    if not any(cmd in chat_response for cmd in ['navigate_to', 'search', 'ask']):
                        self.openai_client.text_to_speech(chat_response)
                
                # Reset for next interaction
                self.stop_detection_event.clear()
                self.recorder = PvRecorder(
                    device_index=-1,
                    frame_length=self.porcupine.frame_length
                )
                
                self.wake_word_thread = threading.Thread(target=self._wake_word_detection_loop)
                self.wake_word_thread.daemon = True
                self.wake_word_thread.start()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Conversation loop error: {e}")
                self.stop_detection_event.clear()

    def start(self):
        """Start the voice assistant"""
        self.stop_detection_event.clear()
        
        self.wake_word_thread = threading.Thread(target=self._wake_word_detection_loop)
        self.wake_word_thread.daemon = True
        self.wake_word_thread.start()
        
        self.conversation_thread = threading.Thread(target=self._conversation_loop)
        self.conversation_thread.daemon = True
        self.conversation_thread.start()

    def stop(self):
        """Stop the voice assistant and clean up resources"""
        self.stop_detection_event.set()
        time.sleep(1)
        
        if hasattr(self, 'wake_word_thread'):
            self.wake_word_thread.join()
        
        if hasattr(self, 'conversation_thread'):
            self.conversation_thread.join()
        
        try:
            self.recorder.stop()
            self.recorder.delete()
            self.porcupine.delete()
        except Exception as e:
            print(f"Cleanup error: {e}")
        
        print("Voice Assistant stopped.")

# Example usage
if __name__ == "__main__":
    # Load environment variables
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PICOVOICE_ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY")
    from spotty.audio import system_prompt_assistant

    # System prompt for the assistant
    SYSTEM_PROMPT = system_prompt_assistant
    assistant = VoiceAssistant(
        system_prompt=SYSTEM_PROMPT,
        picovoice_access_key=PICOVOICE_ACCESS_KEY,
        openai_api_key=OPENAI_API_KEY,
        keyword_path="/Users/shuk/Desktop/spot/practical-seminar-mobile-robotics/spotty/assets/hey_spot_version_02/Hey-Spot_en_mac_v3_0_0.ppn"
    )
    
    try:
        assistant.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        assistant.stop()