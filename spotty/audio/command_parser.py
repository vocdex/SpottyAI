#                                                                               
# Copyright (c) 2025, FAU, Shukrullo Nazirjonov
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#                                                                               
"""Parses assistant responses into structured commands. For each function calling defined in the system prompt,
there should be a corresponding command defined in this file. The command should be extracted from the API response
"""
# spotty/audio/command_parser.py
from typing import Any, Dict, Optional


class CommandParser:
    """Parses assistant responses into structured commands"""

    @staticmethod
    def extract_command(response: str) -> Dict[str, Any]:
        """Extract command and parameters from response"""
        try:
            if "navigate_to(" in response:
                parts = response.split("navigate_to(")[1].split(")")[0].split(",")
                return {
                    "command": "navigate_to",
                    "parameters": {
                        "waypoint_id": parts[0].strip("\"'"),
                        "phrase": parts[1].strip("\"'") if len(parts) > 1 else None,
                    },
                }
            elif "describe_scene(" in response:
                query = response.split("describe_scene(")[1].split(")")[0].strip("\"'")
                return {"command": "describe_scene", "parameters": {"query": query}}
            elif "say(" in response:
                phrase = response.split("say(")[1].split(")")[0].strip("\"'")
                return {"command": "say", "parameters": {"phrase": phrase}}
            elif "ask(" in response:
                question = response.split("ask(")[1].split(")")[0].strip("\"'")
                return {"command": "ask", "parameters": {"question": question}}
            elif "search(" in response:
                query = response.split("search(")[1].split(")")[0].strip("\"'")
                return {"command": "search", "parameters": {"query": query}}
            elif "sit(" in response:
                return {"command": "sit", "parameters": {}}
            elif "stand(" in response:
                return {"command": "stand", "parameters": {}}

            # Default case for unrecognized commands
            return {"command": "say", "parameters": {"phrase": "I'm not sure how to handle that request."}}

        except Exception as e:
            return {"command": "say", "parameters": {"phrase": f"I had trouble processing that command: {str(e)}"}}
