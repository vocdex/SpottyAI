"""This script is knowledgable about SPOT and can parse user commands into actionable JSON tasks for controlling SPOT."""
import os
import pyaudio
import wave
import json
import requests
import threading
import queue
import argparse
import subprocess
import platform

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
API_KEY = os.getenv("OPENAI_API_KEY")

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
stop_recording = queue.Queue()

system_prompt = """You are a voice-based assistant for Spot, a robot dog. Your job is to:
1. Parse user commands into actionable JSON tasks for controlling Spot.
2. If the user mixes unrelated or irrelevant input, ignore the irrelevant parts and extract actionable tasks.
3. For compound commands (multiple tasks in one input), split them into individual tasks.
Respond with a **list of JSON commands**. If no actionable task is found, respond with: [{"action": "none"}].
Examples:
- "Go to the kitchen and find the mug." 
→ [{"action": "navigate_to", "location": "kitchen"}, {"action": "find_object", "object": "mug"}]
- "Move 3 meters forward, then rotate 90 degrees clockwise." 
→ [{"action": "move", "direction": "forward", "distance": 3.0}, {"action": "rotate", "direction": "clockwise", "angle": 90}]
- "Does Spot like pizza? Also, go to the office."
→ [{"action": "navigate_to", "location": "office"}]
- "Sit down, please!" 
→ [{"action": "posture", "state": "sit"}]

Actionable tasks:
- "navigate_to": {"location": "room_name"}
- "find_object": {"object": "object_name"}
- "move": {"direction": "forward/backward/left/right", "distance": float}
- "rotate": {"direction": "clockwise/counterclockwise", "angle": float}
- "posture": {"state": "sit/stand/lie_down"}
- "none": No actionable task found
Valid room names: ["kitchen", "living_room", "bedroom", "bathroom", "office"]. Ignore all other room names.
Non-actionable tasks: ["dance", "bark", "play_dead", "fetch", "roll_over", "shake_hands"]

Strictly follow these guidelines:
- Only generate JSON-formatted responses
- Extract only actionable robotic commands
- Ignore non-actionable conversation
- Be precise and concise in task interpretation

"""

def record_audio():
    """Records audio from the microphone and saves it to TEMP_FILE."""
    print("Recording... Press Enter to stop.")
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []
    
    def listen_for_stop():
        input()  # Wait for Enter key
        stop_recording.put(True)
    
    # Start a thread to listen for the stop signal
    stop_thread = threading.Thread(target=listen_for_stop)
    stop_thread.daemon = True
    stop_thread.start()
    
    try:
        while stop_recording.empty():
            data = stream.read(CHUNK)
            frames.append(data)
    finally:
        stream.stop_stream()
        stream.close()
    
    with wave.open(TEMP_FILE, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    
    print("Recording stopped. Processing audio...")
    return TEMP_FILE

def transcribe_audio_openai(file_path):
    """Transcribes audio using OpenAI Whisper API."""
    if not API_KEY:
        raise ValueError("OpenAI API key is not set.")
    
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
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
    
    if not API_KEY:
        raise ValueError("OpenAI API key is not set.")
    
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
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
    
      
    # System role for Spot robot dog assistant
 

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

def text_to_speech(text):
    """Converts text to speech using system-specific method."""
    system = platform.system()
    
    if system == "Darwin":  # macOS
        # Use say command with different voices
        voices = [ "Samantha"]
        voice = voices[len(chat_history) % len(voices)]  # Cycle through voices
        subprocess.run(["say", "-v", voice, text])
    elif system == "Linux":
        # Fallback to espeak on Linux
        subprocess.run(["espeak", text])
    elif system == "Windows":
        # Fallback to PowerShell Say-Speech on Windows
        subprocess.run(["powershell", "-Command", f"Add-Type -AssemblyName System.speech; $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; $speak.Speak('{text}')"])
    else:
        print("Text-to-speech not supported on this system.")

def main(transcription_method='openai', inference_method='openai', local_whisper_model='tiny', local_llama_model=None):
    """Main function to control recording, transcription, and chat."""
    transcribe_func = transcribe_audio_openai if transcription_method == 'openai' else transcribe_audio_local
    
    chat_func = chat_with_openai if inference_method == 'openai' else chat_with_local_llama
    
    try:
        while True:
            input("Press Enter to start recording...")
            stop_recording.queue.clear()  # Clear any previous stop signals
            record_audio()
            
            try:
                # Transcribe audio
                transcribed_text = (
                    transcribe_func(TEMP_FILE, local_whisper_model) 
                    if transcription_method == 'local' 
                    else transcribe_func(TEMP_FILE)
                )
                print(f">> You said: {transcribed_text}")
                
                # Generate chat response
                chat_response = (
                    chat_func(transcribed_text, local_llama_model) 
                    if inference_method == 'local' 
                    else chat_func(transcribed_text)
                )
                print(f"🦙 Assistant said: {chat_response}")
                # Text to speech
                text_to_speech(chat_response)
                
            except Exception as e:
                print(f"Error: {e}")
            
            print("Press Enter to speak again, or Ctrl+C to exit.")
    
    except KeyboardInterrupt:
        print("\nExiting application...")
    finally:
        audio.terminate()

def parse_arguments():
    """Parse command-line arguments for transcription and inference methods."""
    parser = argparse.ArgumentParser(description="Conversation AI Agent with Flexible Transcription and Inference")
    
    # Transcription method selection
    parser.add_argument('--transcribe-method', 
                        choices=['openai', 'local'], 
                        default='local', 
                        help='Choose transcription method: OpenAI API or local Whisper')
    parser.add_argument('--whisper-model', 
                        choices=['tiny', 'base', 'small', 'medium', 'large'], 
                        default='tiny', 
                        help='Select local Whisper model size (only for local method)')
    
    # Inference method selection
    parser.add_argument('--inference-method', 
                        choices=['openai', 'local'], 
                        default='local', 
                        help='Choose inference method: OpenAI API or local LLaMA')
    parser.add_argument('--llama-model', 
                        type=str, 
                        default="./models/7B/mistral-7b-instruct-v0.2.Q4_K_M.gguf", 
                        help='Path to local LLaMA model')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Check and handle method availability
    if args.transcribe_method == 'local' and not WHISPER_LOCAL_AVAILABLE:
        print("Local Whisper library not found. Falling back to OpenAI API.")
        args.transcribe_method = 'openai'
    
    if args.inference_method == 'local' and not LLAMA_LOCAL_AVAILABLE:
        print("Local LLaMA library not found. Falling back to OpenAI API.")
        args.inference_method = 'openai'
    
    # Run the main application
    main(
        transcription_method=args.transcribe_method, 
        inference_method=args.inference_method,
        local_whisper_model=args.whisper_model,
        local_llama_model=args.llama_model
    )
