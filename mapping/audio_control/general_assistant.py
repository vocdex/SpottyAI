"""This script interacts with the user through audio input and output, using OpenAI's Chat API and Whisper for transcription.
It doesn't know anything about SPOT, but it can be used as a standalone assistant for general conversations."""
import os
import pyaudio
import wave
import requests
import threading
import queue
import argparse
import subprocess
import platform

# Optional local Whisper import
try:
    from whispercpp import Whisper
    WHISPER_LOCAL_AVAILABLE = True
except ImportError:
    WHISPER_LOCAL_AVAILABLE = False

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
    
    # Ensure the output is a clean string, not a list
    if isinstance(transcribed_text, list):
        transcribed_text = transcribed_text[0].strip()
    
    return transcribed_text

def chat_with_openai(user_input):
    """Sends the transcribed input to OpenAI Chat API and returns the response."""
    global chat_history
    
    if not API_KEY:
        raise ValueError("OpenAI API key is not set.")
    
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant providing concise responses in at most two sentences."},
        *chat_history,
        {"role": "user", "content": user_input},
    ]
    
    payload = {"model": "gpt-3.5-turbo", "messages": messages}
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    
    chat_response = response.json()["choices"][0]["message"]["content"]
    chat_history.extend([{"role": "user", "content": user_input}, {"role": "assistant", "content": chat_response}])
    
    return chat_response

def text_to_speech(text):
    """Converts text to speech using system-specific method."""
    system = platform.system()
    
    if system == "Darwin":  # macOS
        # Use say command with different voices
        voices = ["Alex", "Samantha", "Karen"]
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

def main(transcription_method='openai', local_model_size='tiny'):
    """Main function to control recording, transcription, and chat."""
    transcribe_func = transcribe_audio_openai if transcription_method == 'openai' else transcribe_audio_local
    
    try:
        while True:
            input("Press Enter to start recording...")
            stop_recording.queue.clear()  # Clear any previous stop signals
            record_audio()
            
            try:
                transcribed_text = transcribe_func(TEMP_FILE, local_model_size) if transcription_method == 'local' else transcribe_func(TEMP_FILE)
                print(f">> You said: {transcribed_text}")
                chat_response = chat_with_openai(transcribed_text)
                print(f">> Assistant said: {chat_response}")
                
                text_to_speech(chat_response)
                
            except Exception as e:
                print(f"Transcription error: {e}")
            
            print("Press Enter to speak again, or Ctrl+C to exit.")
    
    except KeyboardInterrupt:
        print("\nExiting application...")
    finally:
        audio.terminate()

def parse_arguments():
    """Parse command-line arguments for transcription method and local model size."""
    parser = argparse.ArgumentParser(description="Conversation AI Agent with Flexible Transcription")
    parser.add_argument('--method', choices=['openai', 'local'], default='openai', 
                        help='Choose transcription method: OpenAI API or local Whisper')
    parser.add_argument('--model', choices=['tiny', 'base', 'small', 'medium', 'large'], default='tiny', 
                        help='Select local Whisper model size (only for local method)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    if args.method == 'local' and not WHISPER_LOCAL_AVAILABLE:
        print("Local Whisper library not found. Falling back to OpenAI API.")
        args.method = 'openai'
    
    main(transcription_method=args.method, local_model_size=args.model)