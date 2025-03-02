"""Parses assistant responses into structured commands. For each function calling defined in the system prompt,
there should be a corresponding command defined in this file. The command should be extracted from the API response
"""
from typing import Any, Dict


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
                        "waypoint_id": parts[0].strip(),
                        "phrase": parts[1].strip() if len(parts) > 1 else "",
                    },
                }
            elif "say(" in response:
                phrase = response.split("say(")[1].split(")")[0].strip('"')
                return {"command": "say", "parameters": {"phrase": phrase}}
            elif "ask(" in response:
                question = response.split("ask(")[1].split(")")[0].strip('"')
                return {"command": "ask", "parameters": {"question": question}}
            elif "search(" in response:
                query = response.split("search(")[1].split(")")[0].strip('"')
                return {"command": "search", "parameters": {"query": query}}

            raise ValueError(f"Unknown command in response: {response}")
        except Exception as e:
            print(f"Error extracting command: {e}")
            return {"command": "say", "parameters": {"phrase": "I'm sorry, I couldn't process that command."}}
