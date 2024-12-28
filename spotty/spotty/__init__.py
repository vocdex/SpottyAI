from pathlib import Path
import os
from .audio_control import WakeWordConversationAgent
from .movement import Movement, GraphNavInterface
from .utils import common, graph_nav, robot


SPOTTY_ROOT = Path(__file__).parent.parent
ASSETS_PATH = os.path.join(SPOTTY_ROOT, 'assets')
MAP_PATH = os.path.join(ASSETS_PATH, 'map')
RAG_DB_PATH = os.path.join(ASSETS_PATH, 'rag_db')


# Ensure paths exist to prevent runtime errors
if not os.path.exists(ASSETS_PATH):
    raise FileNotFoundError(f"Assets folder not found at {ASSETS_PATH}")

print(f"SPOTTY_ROOT: {SPOTTY_ROOT}")
print(f"ASSETS_PATH: {ASSETS_PATH}")