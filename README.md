# Spotty: Intelligent Interface for Boston Dynamics Spot Robot

Spotty is a multimodal system that enhances Boston Dynamics' Spot robot with natural language interaction, contextual awareness, and advanced navigation capabilities. It combines computer vision, speech recognition, and language understanding to create a more intuitive and helpful robotic assistant.

## 🌟 Features

### 🗣️ Voice Interaction
- **Wake Word Detection**: Activate Spot with "Hey Spot" using Porcupine wake word detector
- **Speech-to-Text**: Convert voice commands to text using OpenAI Whisper
- **Text-to-Speech**: Natural voice responses with OpenAI TTS
- **Conversational Memory**: Maintain context across interactions

### 🧠 Contextual Understanding
- **Multimodal RAG System**: Retrieve and generate answers based on robot's location and visual context
- **Vector Database**: Store and query spatial information efficiently 
- **Location-Based Responses**: Provide context-aware information relevant to the robot's current position
- **Object and Scene Understanding**: Recognize objects and environments using GPT vision models

### 🗺️ Enhanced Navigation
- **GraphNav Integration**: Navigate complex environments using Boston Dynamics' GraphNav system
- **Waypoint Labeling**: Automatically or manually label waypoints (e.g., "kitchen", "office")
- **Location Queries**: Navigate to locations by name (e.g., "Go to the kitchen")
- **Object Search**: Find and navigate to objects (e.g., "Find the coffee mug")

### 👁️ Visual Intelligence
- **Scene Description**: Describe what the robot sees using vision-language models
- **Visual Question Answering**: Answer questions about the robot's surroundings
- **Object Detection**: Identify objects in the environment
- **Environment Mapping**: Build and maintain a semantic map of the environment

## 🏗️ System Architecture

Spotty consists of several integrated components that work together to provide a cohesive interaction experience:

1. **Unified Spot Interface**: Core component that orchestrates all subsystems
2. **GraphNav Interface**: Handles map recording, localization, and navigation
3. **Audio Interface**: Manages wake word detection, speech recognition, and audio output
4. **RAG Annotation**: Maintains knowledge base about locations and objects
5. **Vision System**: Processes camera feeds and interprets visual information

All components leverage modern AI services:
- **OpenAI GPT-4o-mini**: Natural language understanding and generation
- **OpenAI Whisper & TTS**: Speech processing
- **CLIP**: Visual-language understanding
- **FAISS**: Vector database for efficient similarity search

## 🚀 Getting Started

### Prerequisites

- Boston Dynamics Spot robot
- Python 3.8+
- Boston Dynamics SDK
- API keys for OpenAI and Picovoice

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/vocdex/SpottyAI.git
   cd spotty
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate 
   ```

3. Install the package and dependencies:
   ```bash
   pip install -e .
   ```

4. Set up environment variables:
   ```bash
   export OPENAI_API_KEY="your_openai_api_key"
   export PICOVOICE_ACCESS_KEY="your_picovoice_access_key"
   ```

### Map Creation and Navigation

Before using Spotty's navigation features, you'll need to create a map of your environment:

1. **Record a map**:
   ```bash
   python /scripts/recording_command_line.py --download-filepath /path/to/save/map ROBOT_IP
   ```
   Follow the command line prompts to start recording, move the robot around the space, and stop recording. This will also record camera images as waypoint snapshots. If you have a pre-recorded map, you can skip this step.

2. **Label waypoints**:
   ```bash
   python /scripts/label_waypoints.py --map-path /path/to/map --label-method clip --prompts kitchen hallway office
   ```
   This uses CLIP to automatically label waypoint locations based on visual context from recorded waypoint snapshot images.
   You need to provide a list of locations to label based on your environment.

3. **Create a RAG database**:
   ```bash
   python scripts/label_with_rag.py --map-path /path/to/map --vector-db-path /path/to/database --maybe-label
   ```
   This will create a vector database for efficient waypoint search and retrieval. It will detect visible objects in the waypoint snapshots and generate a short description of the scene using GPT-4o-mini.

4. **View the map**:
   ```bash
   python scripts/view_map.py --map-path /path/to/map
   ```
5. **Visualize the map with waypoint snapshot information**:
   ```bash
   python scripts/visualize_map.py --map-path /path/to/map  --rag-path /path/to/database
   ```

### Running Spotty

Run the main interface:

```bash
python main_interface.py
```

## 📚 Usage Examples

### Basic Voice Interaction

1. Say "Hey Spot" to activate the wake word detection
2. Ask a question or give a command:
   - "What do you see around you?"
   - "Go to the kitchen"
   - "Find a chair"
   - "Tell me about this room"

### Navigation Commands

- **Go to a labeled location**: "Take me to the kitchen"
- **Find an object**: "Can you find the coffee machine?"
- **Return to base**: "Go back to your charging station"
- **Stand/Sit**: "Stand up" or "Sit down"

### Visual Queries

- **Scene description**: "What can you see?"
- **Object identification**: "What objects are on the table?"
- **Environment questions**: "Is there anyone in the hallway?"
- **Spatial questions**: "How many chairs are in this room?"

## 🔧 Advanced Configuration

### System Customization

Modify the following files to customize behavior:

- `spotty/audio/system_prompts.py`: Change the assistant's personality and capabilities
- `spotty/vision/vision_assistant.py`: Adjust vision system configuration
- `spotty/utils/robot_utils.py`: Configure robot connection settings

### Creating Custom Wake Words

You can create custom wake words using Picovoice Console:

1. Visit https://console.picovoice.ai/
2. Create a new wake word
3. Download the .ppn file
4. Update the KEYWORD_PATH in `spotty/__init__.py`

## 🛠️ Development

### Project Structure

```
spotty/
├── assets/             # Wake words, maps, and vector databases
├── scripts/            # Command-line utilities
├── spotty/             # Main package
│   ├── annotation/     # Waypoint annotation tools
│   ├── audio/          # Audio processing components
│   ├── mapping/        # Navigation components
│   ├── utils/          # Utility functions
│   ├── vision/         # Vision components
│   └── __init__.py     # Package initialization
├── README.md           # This file
└── setup.py            # Package installation
```
### Pre-recorded Maps and Databases
To use pre-recorded maps and databases, download the assets folder from Google Drive and place it in the root directory of the project:
[Spotty Assets](https://drive.google.com/drive/folders/121bVTZ4XUPne3RY7Yg7RoRAkBqK9336t?usp=sharing)

### Adding New Capabilities

To extend Spotty with new features:

1. Develop your component in the appropriate subdirectory
2. Integrate it with the UnifiedSpotInterface in main_interface.py
3. Update prompts and command handlers as needed

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Boston Dynamics for the Spot robot and SDK
- OpenAI for GPT-4o-mini, Whisper, and TTS
- Picovoice for wake word detection
- The open-source community for various libraries and tools

## 📧 Contact

For questions, suggestions, or collaborations, please [open an issue](https://github.com/vocdex/SpottyAI/issues) or contact the maintainers directly.