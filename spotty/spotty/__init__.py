from pathlib import Path
import os

SPOTTY_ROOT = Path(__file__).parent.parent
ASSETS_PATH = os.path.join(SPOTTY_ROOT, 'assets')
MAP_PATH = os.path.join(ASSETS_PATH, 'maps')
RAG_DB_PATH = os.path.join(ASSETS_PATH, 'database')
KEYWORD_PATH = os.path.join(ASSETS_PATH, 'hey_spot_porcupine/Hey-Spot_en_mac_v3_0_0.ppn')


# Ensure paths exist to prevent runtime errors
if not os.path.exists(ASSETS_PATH):
    raise FileNotFoundError(f"Assets folder not found at {ASSETS_PATH}")
