#                                                                               
# Copyright (c) 2025, FAU, Shukrullo Nazirjonov
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#                                                                               
import os
from pathlib import Path

SPOTTY_ROOT = Path(__file__).parent.parent
ASSETS_PATH = os.path.join(SPOTTY_ROOT, "assets")
MAP_PATH = os.path.join(ASSETS_PATH, "maps")
RAG_DB_PATH = os.path.join(ASSETS_PATH, "database")
# Get the platform (linux, mac) to load the correct keyword file
platform = "linux" if os.name == "posix" else "mac"
print(f"Platform: {platform}")
KEYWORD_PATH = os.path.join(ASSETS_PATH, f"hey_spot_porcupine/Hey-Spot_en_{platform}_v3_0_0.ppn")


# Ensure paths exist to prevent runtime errors
if not os.path.exists(ASSETS_PATH):
    raise FileNotFoundError(f"Assets folder not found at {ASSETS_PATH}")


ROTATION_ANGLE = {
    "back_fisheye_image": 0,
    "frontleft_fisheye_image": -90,
    "frontright_fisheye_image": -90,
    "left_fisheye_image": 0,
    "right_fisheye_image": 180,
}
