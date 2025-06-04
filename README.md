# Spotty: Voice Interface for Boston Dynamics Spot
Natural language interface for Spot robot with vision, navigation, and contextual awareness.



<p align="center">
  <img src="assets/spot_logo.png", alt="Spotty Logo" width="400" height="450">
</p>


## Video Demo
[![Spotty Demo Video](https://img.youtube.com/vi/7-Tha8riGnU/maxresdefault.jpg)](https://youtu.be/7-Tha8riGnU?t=117)

More videos available at [GDrive](https://drive.google.com/drive/folders/1Y0DPO2_XnNbGx1GQ6T4V72DM6P8rJYkU?usp=sharing):
- Visual Question Answering
- Navigating to kitchen
- Navigating to KUKA robot
- Navigating to trash can

## Features

**Voice Control**
- Wake word activation ("Hey Spot")
- Speech-to-text via OpenAI Whisper
- Text-to-speech responses
- Conversation memory

**Navigation**
- GraphNav integration with waypoint labeling
- Location-based commands ("Go to kitchen")
- Object search and navigation
- Automatic scene understanding via GPT-4o-mini + CLIP

**Vision**
- Scene description and visual Q&A
- Object detection and environment mapping
- Multimodal RAG system for location context

## Architecture

- **UnifiedSpotInterface**: Main orchestrator
- **GraphNav Interface**: Map recording and navigation
- **Audio Interface**: Wake word detection, speech processing
- **RAG Annotation**: Location/object knowledge base
- **Vision System**: Camera processing and interpretation

Uses OpenAI GPT-4o-mini, Whisper, TTS, CLIP, and FAISS vector database.

## Setup

### Prerequisites
- Boston Dynamics Spot robot
- Python 3.8+
- Boston Dynamics SDK
- OpenAI and Picovoice API keys

### Installation

```bash
git clone https://github.com/vocdex/SpottyAI.git
cd spotty
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-optional.txt
pip install -e .

export OPENAI_API_KEY="your_key"
export PICOVOICE_ACCESS_KEY="your_key"
```

### Map Setup

1. **Record environment map:**
   ```bash
   python scripts/recording_command_line.py --download-filepath /path/to/map ROBOT_IP
   ```

2. **Auto-label waypoints:**
   ```bash
   python scripts/label_waypoints.py --map-path /path/to/map --label-method clip --prompts kitchen hallway office
   ```

3. **Create RAG database:**
   ```bash
   python scripts/label_with_rag.py --map-path /path/to/map --vector-db-path /path/to/database --maybe-label
   ```

4. **Visualize setup:**
   ```bash
   python scripts/visualize_map.py --map-path /path/to/map --rag-path /path/to/database
   ```

### Run

```bash
python main_interface.py
```

## Usage

**Voice Commands:**
- "Hey Spot" → activate
- "Go to the kitchen" → navigate to location
- "Find a chair(a mug, a plant, etc.)" → search and navigate to object
- "What do you see?" → describe surroundings
- "Stand up" / "Sit down" → basic robot control

## Configuration

- `spotty/audio/system_prompts.py` - Assistant personality
- `spotty/vision/vision_assistant.py` - Vision settings
- `spotty/utils/robot_utils.py` - Robot connection

Custom wake words: Create at https://console.picovoice.ai/, update KEYWORD_PATH in `spotty/__init__.py`

## Project Structure

```
spotty/
├── assets/             # Maps, databases, wake words
├── scripts/            # Setup utilities
├── spotty/
│   ├── annotation/     # Waypoint tools
│   ├── audio/          # Speech processing
│   ├── mapping/        # Navigation
│   ├── vision/         # Computer vision
│   └── utils/          # Shared utilities
└── main_interface.py   # Entry point
```

Pre-recorded assets( maps, voice activation model, RAG database): [Download from Google Drive](https://drive.google.com/drive/folders/121bVTZ4XUPne3RY7Yg7RoRAkBqK9336t)

## License

MIT License
## Disclaimer
This project is an open-source implementation of [Boston Dynamics Demo](https://youtu.be/djzOBZUFzTw) and is not affiliated with Boston Dynamics. It is intended for educational and research purposes only. Use at your own risk.
## Acknowledgements
- Big thanks to Automatic Control Chair at FAU for providing the robot for my semester project
- Many thanks to Boston Dynamics’ engineers for their work on Spot SDK
- HuggingFace, OpenAI, Facebook Research