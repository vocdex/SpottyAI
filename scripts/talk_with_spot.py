#                                                                               
# Copyright (c) 2025, FAU, Shukrullo Nazirjonov
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#                                                                               
import argparse
import os

from dotenv import load_dotenv

from spotty import ASSETS_PATH
from spotty.audio import (
    WakeWordConversationAgent,
    initialize_pygame_mixer,
    system_prompt_assistant,
)
from spotty.utils.common_utils import get_abs_path

load_dotenv()

# Configuration and API settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PICOVOICE_ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY")


def main():
    keyword_model = os.path.join(ASSETS_PATH, "hey_spot_version_02/Hey-Spot_en_mac_v3_0_0.ppn")
    # Parse arguments
    parser = argparse.ArgumentParser(description="Wake Word Conversation Agent")
    parser.add_argument("--keyword-model", required=False, default=keyword_model, help="Path to wake word model")
    parser.add_argument("--transcribe", choices=["openai", "local"], default="openai")
    parser.add_argument("--whisper-model", choices=["tiny", "base", "small", "medium", "large"], default="tiny")
    parser.add_argument("--chat", choices=["openai", "local"], default="openai")
    parser.add_argument("--llama-model", type=str, default="./models/mistral-7b-instruct-v0.1.Q4_K_S.gguf")
    parser.add_argument("--tts", type=str, choices=["openai", "local"], default="openai")
    parser.add_argument("--audio-device-index", type=int, default=-1)

    args = parser.parse_args()
    initialize_pygame_mixer()

    agent = WakeWordConversationAgent(
        access_key=PICOVOICE_ACCESS_KEY,
        system_prompt=system_prompt_assistant,
        keyword_path=get_abs_path(args.keyword_model),
        transcription_method=args.transcribe,
        inference_method=args.chat,
        local_whisper_model=args.whisper_model,
        local_llama_model=get_abs_path(args.llama_model),
        tts=args.tts,
        audio_device_index=args.audio_device_index,
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
