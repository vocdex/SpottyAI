"""This script is knowledgable about SPOT and can parse user commands into actionable JSON tasks for controlling SPOT."""

import os
import pyaudio
import wave
import requests
import threading
import queue
import argparse
import subprocess
import time
import pvporcupine
import numpy as np
from scipy import signal
import sounddevice as sd
from pvrecorder import PvRecorder
import platform
from system_prompts import  system_prompt_robin
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

system_prompt=system_prompt_robin
# Optional local model imports
try:
    from whispercpp import Whisper
    from llama_cpp import Llama
    WHISPER_LOCAL_AVAILABLE = True
    LLAMA_LOCAL_AVAILABLE = True
except ImportError:
    WHISPER_LOCAL_AVAILABLE = False
    LLAMA_LOCAL_AVAILABLE = False

# Configuration and API settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PICOVOICE_ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY")

# Audio settings
RATE = 16000
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
TEMP_FILE = "output.wav"

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Chat history
chat_history = []
stop_recording_queue = queue.Queue()

def record_audio(max_recording_time=5, stop_queue=None):
    """
    Records audio from the microphone with more flexible stopping options.
    
    :param max_recording_time: Maximum recording duration in seconds
    :param stop_queue: Optional queue to signal stopping the recording
    :return: Path to the recorded audio file
    """
    print("Recording...")
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []
    
    start_time = time.time()
    
    try:
        while True:
            # Check if we've exceeded max recording time
            if time.time() - start_time > max_recording_time:
                print("Max recording time reached.")
                break
            
            # Check if stop is signaled via queue
            if stop_queue and not stop_queue.empty():
                stop_queue.get()
                print("Recording stopped by signal.")
                break
            
            # Read audio data
            data = stream.read(CHUNK)
            frames.append(data)
            
            # Optional: Add a small sleep to prevent tight looping
            time.sleep(0.01)
    
    except Exception as e:
        print(f"Recording error: {e}")
    
    finally:
        # Clean up audio stream
        stream.stop_stream()
        stream.close()
        
        # Save recorded audio
        if frames:
            with wave.open(TEMP_FILE, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(audio.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
            
            print(f"Recording saved. Duration: {time.time() - start_time:.2f} seconds")
            return TEMP_FILE
        else:
            print("No audio recorded.")
            return None

def transcribe_audio_openai(file_path):
    """Transcribes audio using OpenAI Whisper API."""
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key is not set.")
    
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    
    with open(file_path, "rb") as audio_file:
        files = {
            "file": (file_path, audio_file, "audio/wav"),
            "model": (None, "whisper-1"),
            "response_format": (None, "text"),
        }
        response = requests.post(url, headers=headers, files=files)
        response.raise_for_status()
        return response.text.strip()

def transcribe_audio_local(file_path, model_size='tiny'):
    """Transcribes audio using local Whisper model."""
    if not WHISPER_LOCAL_AVAILABLE:
        raise ImportError("Local Whisper library (whispercpp) is not installed.")
    
    w = Whisper(model_size)
    result = w.transcribe(file_path)
    transcribed_text = w.extract_text(result)
    print(f"Transcribed text: {transcribed_text}")
    
    # Ensure the output is a clean string, not a list
    if isinstance(transcribed_text, list):
        # Join the list of transcriptions
        transcribed_text = " ".join(transcribed_text)
    
    return transcribed_text

def chat_with_openai(user_input):
    """Sends the transcribed input to OpenAI Chat API and returns the response."""
    global chat_history
    
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key is not set.")
    
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    messages = [
        {"role": "system", "content": system_prompt},
        *chat_history,
        {"role": "user", "content": user_input},
    ]
    
    payload = {"model": "gpt-4o-mini", "messages": messages}
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    
    chat_response = response.json()["choices"][0]["message"]["content"]
    chat_history.extend([{"role": "user", "content": user_input}, {"role": "assistant", "content": chat_response}])
    
    return chat_response



def chat_with_local_llama(user_input, model_path="./models/7B/llama-model.gguf"):
    """Sends the transcribed input to local LLaMA model and returns the response."""
    global chat_history
    
    if not LLAMA_LOCAL_AVAILABLE:
        raise ImportError("Local LLaMA library (llama-cpp-python) is not installed.") 

    # Construct the prompt with chat history
    prompt = f"System: {system_prompt}\n"
    prompt += "".join([f"{'Human' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}\n" for msg in chat_history])
    prompt += f"Human: {user_input}\nAssistant: "
    
    
    # Load the LLaMA model (only if not already loaded)
    if not hasattr(chat_with_local_llama, 'llm'):
        chat_with_local_llama.llm = Llama(
            model_path=model_path,
            n_ctx=4096,  # Adjust context window as needed
            n_gpu_layers=-1,  # Uncomment for GPU acceleration
            verbose=False
        )
    
    # Generate response
    output = chat_with_local_llama.llm(
        prompt, 
        max_tokens=256,  # Adjust as needed
        stop=["Human:", "\n"],
        echo=False
    )
    
    # Extract and clean the response
    chat_response = output['choices'][0]['text'].strip()
    
    # Update chat history
    chat_history.extend([{"role": "user", "content": user_input}, {"role": "assistant", "content": chat_response}])
    
    return chat_response

# TTS functions

def create_modern_beep(type="start", volume=0.3):
    """
    Create a modern notification sound similar to Alexa/Siri.
    
    :param type: "start" for recording start sound, "end" for recording end sound
    :param volume: Volume of the sound (0.0 to 1.0)
    :return: NumPy array of audio samples
    """
    sample_rate = 44100
    
    if type == "start":
        # Create an upward sweep
        duration = 0.15
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        # Main frequencies for the sweep
        f1, f2 = 880, 1760  # A5 to A6
        sweep = volume * (
            np.sin(2 * np.pi * (f1 + (f2-f1) * t/duration) * t) +
            0.3 * np.sin(4 * np.pi * (f1 + (f2-f1) * t/duration) * t)
        )
        
        # Smooth envelope
        envelope = np.exp(-4 * (t - duration/2) ** 2 / (duration/2) ** 2)
        
    else:  # end sound
        # Create a downward sweep
        duration = 0.15
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        # Main frequencies for the sweep
        f1, f2 = 1760, 880  # A6 to A5
        sweep = volume * (
            np.sin(2 * np.pi * (f1 + (f2-f1) * t/duration) * t) +
            0.3 * np.sin(4 * np.pi * (f1 + (f2-f1) * t/duration) * t)
        )
        
        # Smooth envelope with faster decay
        envelope = np.exp(-6 * (t - duration/4) ** 2 / (duration/2) ** 2)
    
    # Apply envelope and smoothing
    sound = (sweep * envelope).astype(np.float32)
    
    # Apply a gentle low-pass filter
    from scipy import signal
    b, a = signal.butter(2, 0.1)
    sound = signal.filtfilt(b, a, sound)
    
    return sound

def play_modern_beep(type="start", volume=0.3, blocking=True):
    """
    Play a modern notification sound.
    
    :param type: "start" for recording start sound, "end" for recording end sound
    :param volume: Volume of the sound (0.0 to 1.0)
    :param blocking: Whether to block until sound is finished
    """
    try:
        beep = create_modern_beep(type, volume)
        if blocking:
            sd.play(beep, samplerate=44100)
            sd.wait()
        else:
            sd.play(beep, samplerate=44100)
    except Exception as e:
        print(f"Error playing notification sound: {e}")

        
def text_to_speech(text,tts):
    """
    Converts text to speech using OpenAI TTS API.
    
    :param text: Text to be converted to speech
    :return: Path to the generated audio file
    """
    if not OPENAI_API_KEY or tts != 'openai':
        print("OpenAI API key is not set. Falling back to system TTS.")
        # Fallback to existing system TTS method
        system = platform.system()
        
        if system == "Darwin":  # macOS
            voices = ["Samantha"]
            voice = voices[len(chat_history) % len(voices)]
            subprocess.run(["say", "-v", voice, text])
        elif system == "Linux":
            subprocess.run(["espeak", text])
        elif system == "Windows":
            subprocess.run(["powershell", "-Command", f"Add-Type -AssemblyName System.speech; $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; $speak.Speak('{text}')"])
        else:
            print("Text-to-speech not supported on this system.")
        return None

    # OpenAI TTS API endpoint
    url = "https://api.openai.com/v1/audio/speech"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Payload for TTS request
    payload = {
        "model": "tts-1",  # You can also use "tts-1-hd" for higher quality
        "input": text,
        "voice": "shimmer"  # Options: alloy, echo, fable, onyx, nova, shimmer
    }
    
    # Output file path
    output_file = "tts_output.mp3"
    
    try:
        # Make the API request
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        # Save the audio file
        with open(output_file, "wb") as audio_file:
            audio_file.write(response.content)
        
        # Play the audio file using system-specific method
        system = platform.system()
        if system == "Darwin":  # macOS
            subprocess.run(["afplay", output_file])
        elif system == "Linux":
            subprocess.run(["aplay", output_file])
        elif system == "Windows":
            subprocess.run(["start", output_file], shell=True)
        
        return output_file
    
    except Exception as e:
        print(f"OpenAI TTS error: {e}")
        # Fallback to system TTS if OpenAI TTS fails
        return None



class WakeWordConversationAgent:
    def __init__(self, 
                 access_key, 
                 keyword_path, 
                 transcription_method='openai', 
                 inference_method='openai', 
                 local_whisper_model='tiny', 
                 local_llama_model=None,
                 tts='openai',
                 audio_device_index=-1):
        """
        Initialize the Conversation Agent with Wake Word Detection.
        
        :param access_key: PicoVoice access key
        :param keyword_path: Path to the wake word model
        :param transcription_method: 'openai' or 'local'
        :param inference_method: 'openai' or 'local'
        :param local_whisper_model: Size of local Whisper model
        :param local_llama_model: Path to local LLaMA model
        :param tts: Text-to-speech method: 'openai' or 'local'
        :param audio_device_index: Audio input device index
        """
        # Wake Word Detection Setup
        self.porcupine = pvporcupine.create(
            access_key=access_key,
            keyword_paths=[keyword_path],
            sensitivities=[0.5]
        )
        self.recorder = PvRecorder(
            device_index=audio_device_index, 
            frame_length=self.porcupine.frame_length
        )
        self.audio_device_index = audio_device_index
        
        # Configuration for transcription and chat
        self.transcription_method = transcription_method
        self.inference_method = inference_method
        self.local_whisper_model = local_whisper_model
        self.local_llama_model = local_llama_model
        self.tts = tts
        
        # Threading and control
        self.is_running = False
        self.wake_word_thread = None
        self.stop_event = threading.Event()
        
        # Queues for thread communication
        self.wake_word_queue = queue.Queue()
        self.stop_detection_event = threading.Event()

    
    def _wake_word_detection_loop(self):
        """
        Continuous loop for detecting wake words.
        Uses threading Event for more controlled stopping.
        """
        try:
            self.recorder.start()
            print("Listening for wake word...")
            
            while not self.stop_detection_event.is_set():
                pcm = self.recorder.read()
                result = self.porcupine.process(pcm)
                
                if result >= 0:
                    print("Wake word detected! Preparing for interaction...")
                    self.wake_word_queue.put(True)
                    time.sleep(0.1) # Reduce CPU usage when no wake word detected
        
        except Exception as e:
            print(f"Wake word detection error: {e}")
        
        finally:
            try:
                # Safely attempt to stop the recorder
                self.recorder.stop()
            except Exception as stop_error:
                print(f"Error stopping recorder: {stop_error}")

    
    def _conversation_loop(self):
        while not self.stop_event.is_set():
            try:
                # Wait for wake word detection
                self.wake_word_queue.get(timeout=1)
                
                # Play start recording beep (higher pitch)
                play_modern_beep(type="start", volume=1.0)
                
                # Stop the PvRecorder to prevent audio overlap
                self.stop_detection_event.set()
                # Give some time for the recorder to stop
                time.sleep(0.5)
            
                # Record audio with 10-second max time and optional stop queue
                record_audio(max_recording_time=10, stop_queue=stop_recording_queue)
                
                # Play end recording beep (lower pitch)
                play_modern_beep(type="end", volume=1.0)
                
                # Transcribe audio
                transcribe_func = (
                    transcribe_audio_local if self.transcription_method == 'local' 
                    else transcribe_audio_openai
                )
                transcribed_text = (
                    transcribe_func(TEMP_FILE, self.local_whisper_model) 
                    if self.transcription_method == 'local' 
                    else transcribe_func(TEMP_FILE)
                )
                print(f">> You said: {transcribed_text}")
                
                # Generate chat response
                chat_func = (
                    chat_with_local_llama if self.inference_method == 'local' 
                    else chat_with_openai
                )
                chat_response = (
                    chat_func(transcribed_text, self.local_llama_model) 
                    if self.inference_method == 'local' 
                    else chat_func(transcribed_text)
                )
                print(f"🦙 Assistant said: {chat_response}")
                
                # Text to speech
                text_to_speech(chat_response, tts=self.tts)

                # Reset for next wake word detection
                self.stop_detection_event.clear()
                try:
                    self.recorder.stop()
                except:
                    pass
                # Create a new recorder instance
                self.recorder = PvRecorder(
                    device_index=self.audio_device_index, 
                    frame_length=self.porcupine.frame_length
                )
                 # Restart wake word detection
                self.wake_word_thread = threading.Thread(target=self._wake_word_detection_loop)
                self.wake_word_thread.daemon = True
                self.wake_word_thread.start()
            except queue.Empty:
                # No wake word detected, continue listening
                continue
            except Exception as e:
                print(f"Conversation loop error: {e}")
                # Ensure the detection can restart
                self.stop_detection_event.clear()
    
    def start(self):
        """
        Start wake word detection and conversation threads.
        """
        self.stop_detection_event.clear()
        
        # Start wake word detection thread
        self.wake_word_thread = threading.Thread(target=self._wake_word_detection_loop)
        self.wake_word_thread.daemon = True
        self.wake_word_thread.start()
        
        # Start conversation thread
        self.conversation_thread = threading.Thread(target=self._conversation_loop)
        self.conversation_thread.daemon = True
        self.conversation_thread.start()
    
    def stop(self):
        """
        Stop all threads and clean up resources.
        """
        # Set stop events
        self.stop_detection_event.set()
        
        # Give threads time to stop
        time.sleep(1)
        
        # Join threads
        if hasattr(self, 'wake_word_thread'):
            self.wake_word_thread.join()
        
        if hasattr(self, 'conversation_thread'):
            self.conversation_thread.join()
        
        # Clean up resources
        try:
            self.recorder.stop()
            self.recorder.delete()
        except Exception as e:
            print(f"Error during cleanup: {e}")
        
        try:
            self.porcupine.delete()
        except Exception as e:
            print(f"Error deleting porcupine: {e}")
        
        print("Conversation Agent stopped.")
    
def get_abs_path(relative_path:str):
    """This function is used to convert relative path to absolute path."""
    if not os.path.isabs(relative_path):
        return os.path.abspath(relative_path)
    return relative_path

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Wake Word Conversation Agent")
    parser.add_argument('--keyword-model', required=False, default="./hey_spot_version_02/Hey-Spot_en_mac_v3_0_0.ppn", help='Path to wake word model')
    parser.add_argument('--transcribe', choices=['openai', 'local'], default='openai')
    parser.add_argument('--whisper-model', choices=['tiny', 'base', 'small', 'medium', 'large'], default='tiny')
    parser.add_argument('--chat', choices=['openai', 'local'], default='openai')
    parser.add_argument('--llama-model', type=str, default="./models/mistral-7b-instruct-v0.1.Q4_K_S.gguf")
    parser.add_argument('--tts', type=str, choices=['openai', 'local'], default='openai')
    parser.add_argument('--audio-device-index', type=int, default=-1)
    
    args = parser.parse_args()
    
    # Initialize and start the conversation agent
    agent = WakeWordConversationAgent(
        access_key=PICOVOICE_ACCESS_KEY,
        keyword_path=get_abs_path(args.keyword_model),
        transcription_method=args.transcribe,
        inference_method=args.chat,
        local_whisper_model=args.whisper_model,
        local_llama_model=get_abs_path(args.llama_model),
        tts = args.tts,
        audio_device_index=args.audio_device_index
    )
    
    try:
        agent.start()
        # Keep the main thread running
        agent.wake_word_thread.join()
    except KeyboardInterrupt:
        print("\nStopping Conversation Agent...")
    finally:
        agent.stop()

if __name__ == "__main__":
    main()