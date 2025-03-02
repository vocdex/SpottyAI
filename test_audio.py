import os
import queue
import select
import sys
import termios
import time
import tty

from spotty import KEYWORD_PATH
from spotty.audio.robot_interface import (
    AudioConfig,
    AudioManager,
    ChatClient,
    VoiceInterface,
    VoiceInterfaceConfig,
    WakeWordConfig,
    WakeWordDetector,
)


def test_audio_recording():
    # Initialize configuration
    config = AudioConfig()

    # Create AudioManager instance
    audio_manager = AudioManager(config)

    try:
        # Test recording
        print("\nStarting recording test (5 seconds)...")
        audio_manager.play_feedback_sound("start")

        # Create stop queue for potentially stopping recording early
        stop_queue = queue.Queue()

        # Record audio
        recorded_file = audio_manager.record_audio(max_recording_time=10, stop_queue=stop_queue)

        audio_manager.play_feedback_sound("stop")

        if recorded_file:
            print(f"Test successful! Audio saved to: {recorded_file}")
        else:
            print("Test failed: No audio file was created")

    except Exception as e:
        print(f"Test failed with error: {e}")
    finally:
        # Cleanup
        audio_manager.audio.terminate()


def test_chat_completion():
    system_prompt = """You are Spot, a helpful and friendly robot assistant.
    Keep your responses concise and natural. When asked to perform an action,
    respond with specific commands like 'say(message)' or 'ask(question)'."""
    chat_client = ChatClient(system_prompt=system_prompt, max_context_length=10)
    audio_manager = AudioManager(AudioConfig())

    print("\nChat Interface Test")
    print("Commands:")
    print("- Press 'r' to start recording")
    print("- Press 'h' to see chat history")
    print("- Press 'q' to quit")
    print("-" * 50)

    while True:
        try:
            # Wait for keyboard command
            command = input("\nPress 'r' to record, 'h' for history, 'q' to quit: ").lower()

            if command == "q":
                print("Ending chat session.")
                break

            elif command == "h":
                print("\nChat History:")
                for msg in chat_client.chat_history:
                    role = msg["role"].capitalize()
                    content = msg["content"]
                    timestamp = msg.get("timestamp", "N/A")
                    print(f"{role} ({timestamp}): {content}")
                continue

            elif command == "r":
                # Record and process audio
                audio_manager.play_feedback_sound("start")
                audio_file = audio_manager.record_audio(max_recording_time=6)
                audio_manager.play_feedback_sound("stop")

                if audio_file:
                    user_input = chat_client.speech_to_text(audio_file)
                    print(f"\nYou said: {user_input}")

                    # Get response from chat client
                    response = chat_client.chat_completion(user_input)
                    print(f"\nSpot: {response}")

                    # TTS
                    audio_file = chat_client.text_to_speech(response)
                    if audio_file:
                        audio_manager.play_audio(audio_file)

            else:
                print("Invalid command. Please try again.")

        except KeyboardInterrupt:
            print("\nChat session interrupted.")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue


def test_voice_interface():
    system_prompt = """You are Spot, a helpful and friendly robot assistant.
    Keep your responses concise and natural. When asked to perform an action,
    respond with specific commands like 'say(message)' or 'ask(question)'."""

    # Initialize components
    chat_client = ChatClient(system_prompt=system_prompt, max_context_length=10)
    audio_manager = AudioManager(AudioConfig())

    # Initialize wake word detector
    wake_word_config = WakeWordConfig(access_key=os.getenv("PICOVOICE_ACCESS_KEY"), keyword_path=KEYWORD_PATH)
    wake_detector = WakeWordDetector(wake_word_config)

    print("\nVoice Interface Test")
    print("Commands:")
    print("- Say 'Hey Spot' to start recording")
    print("- Press 'h' to see chat history")
    print("- Press 'q' to quit")
    print("-" * 50)

    def handle_wake_word():
        """Callback for wake word detection"""
        audio_manager.play_feedback_sound("start")

    # Start wake word detection
    wake_detector.start(callback=handle_wake_word)

    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    while True:
        try:
            # Check for keyboard commands
            if select.select([sys.stdin], [], [], 0.1)[0]:
                command = sys.stdin.read(1).lower()
                if command == "q":
                    print("Ending chat session.")
                    break

                elif command == "h":
                    print("\nChat History:")
                    for msg in chat_client.chat_history:
                        role = msg["role"].capitalize()
                        content = msg["content"]
                        timestamp = msg.get("timestamp", "N/A")
                        print(f"{role} ({timestamp}): {content}")
                    continue

            # Check if wake word was detected
            try:
                wake_detector.wake_word_queue.get_nowait()
                print("\n Starting recording...")
                wake_detector.stop()

                # Record and process audio
                audio_file = audio_manager.record_audio(max_recording_time=6)
                audio_manager.play_feedback_sound("stop")

                if audio_file:
                    user_input = chat_client.speech_to_text(audio_file)
                    print(f"\nYou said: {user_input}")

                    # Get response from chat client
                    response = chat_client.chat_completion(user_input)
                    print(f"\nSpot: {response}")

                    # TTS
                    audio_file = chat_client.text_to_speech(response)
                    if audio_file:
                        audio_manager.play_audio(audio_file)

                # Reset wake word detection
                wake_detector.reset()
                wake_detector.start(callback=handle_wake_word)

            except queue.Empty:
                continue

        except KeyboardInterrupt:
            print("\nChat session interrupted.")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue

    # Cleanup
    wake_detector.stop()
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def test_voice_interface_no_key():
    system_prompt = """You are Spot, a helpful and friendly robot assistant.
    Keep your responses concise and natural. When asked to perform an action,
    respond with specific commands like 'say(message)' or 'ask(question)'."""

    # Initialize components
    chat_client = ChatClient(system_prompt=system_prompt, max_context_length=10)
    audio_manager = AudioManager(AudioConfig())

    # Initialize wake word detector
    wake_word_config = WakeWordConfig(access_key=os.getenv("PICOVOICE_ACCESS_KEY"), keyword_path=KEYWORD_PATH)
    wake_detector = WakeWordDetector(wake_word_config)

    print("\nVoice Interface Test")
    print("Say 'Hey Spot' to start a conversation")
    print("-" * 50)

    def handle_wake_word():
        """Callback for wake word detection"""
        audio_manager.play_feedback_sound("start")

    try:
        # Start wake word detection
        wake_detector.start(callback=handle_wake_word)

        while True:
            try:
                # Wait for wake word detection
                wake_detector.wake_word_queue.get()

                # Stop wake word detection during recording
                wake_detector.stop()

                # Give time to stop wake word detection
                time.sleep(0.5)

                # Record and process audio
                audio_file = audio_manager.record_audio(max_recording_time=6)
                audio_manager.play_feedback_sound("stop")
                time.sleep(0.3)

                if audio_file:
                    user_input = chat_client.speech_to_text(audio_file)
                    print(f"\nYou said: {user_input}")

                    # Get response from chat client
                    response = chat_client.chat_completion(user_input)
                    print(f"\nSpot: {response}")

                    # TTS
                    audio_file = chat_client.text_to_speech(response)
                    if audio_file:
                        audio_manager.play_audio(audio_file)
                        time.sleep(0.5)
                    # Give time for audio resouces to be released
                    time.sleep(0.5)
                # Reset wake word detection
                wake_detector = WakeWordDetector(wake_word_config)
                wake_detector.start(callback=handle_wake_word)

            except Exception as e:
                print(f"\nError during conversation: {e}")
                # Reset wake word detection on error
                wake_detector = WakeWordDetector(wake_word_config)
                wake_detector.start(callback=handle_wake_word)
                print("\nListening for wake word...")

    except KeyboardInterrupt:
        print("\nStopping voice interface...")
    finally:
        wake_detector.stop()
        time.sleep(0.5)
        print("Voice interface stopped.")


if __name__ == "__main__":
    # test_chat_completion()
    # test_audio_recording()
    # test_voice_interface()
    # test_voice_interface_no_key()
    config = VoiceInterfaceConfig(max_recording_time=6)
    voice_interface = VoiceInterface(config)
    try:
        voice_interface.run()
    except KeyboardInterrupt:
        print("\nStopping voice interface...")
    finally:
        voice_interface.cleanup()
        print("Voice interface stopped.")
