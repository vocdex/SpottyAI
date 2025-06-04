#                                                                               
# Copyright (c) 2025, FAU, Shukrullo Nazirjonov
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#                                                                               
import base64
import logging
from typing import Dict

from openai import OpenAI


class VQAHandler:
    """Handles Visual Question Answering using vision-language models"""

    def __init__(self, openai_client: OpenAI, logger=None):
        self.openai_client = openai_client
        self.logger = logger or logging.getLogger(__name__)

    def process_query(self, query: str, images: Dict[str, bytes], model: str = "gpt-4o-mini") -> str:
        """Process a visual query using the available images"""
        try:
            if not images:
                return "I don't have access to camera images"

            messages = [
                {
                    "role": "system",
                    "content": """
                    You are assisting a robot in analyzing its environment through two camera feeds
                    that show the left and right front views. These images together form a wider
                    view of the scene. When describing what you see:

                    1. Provide a single, coherent description that combines information from both views
                    2. Focus on spatial relationships between objects across both images
                    3. Avoid mentioning "left camera" or "right camera" unless specifically asked
                    4. Describe the scene as if you're looking at it from the robot's perspective
                    5. If you see the same object in both images, mention it only once
                    Be creative and have fun with your responses!
                    Be concise(2-3 sentences)
                    Remember to address the user's specific query in your response.
                    """,
                }
            ]

            user_content = [{"type": "text", "text": query}]

            for source, img in images.items():
                try:
                    base64_image = base64.b64encode(img).decode("utf-8")
                    user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
                except Exception as e:
                    self.logger.error(f"Error processing image from {source}: {str(e)}")
                    continue

            messages.append({"role": "user", "content": user_content})

            response = self.openai_client.chat.completions.create(model=model, messages=messages)

            return response.choices[0].message.content

        except Exception as e:
            self.logger.error(f"Error in VQA processing: {str(e)}")
            return "I had trouble analyzing the images. Please try again."
