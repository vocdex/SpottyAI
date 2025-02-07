from dataclasses import dataclass
import time
import cv2
from PIL import Image
import io
import os
import base64
from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn.api.geometry_pb2 import SE2Velocity, SE2VelocityLimit, Vec2
from bosdyn import geometry
from openai import OpenAI

@dataclass
class VisionConfig:
    """Configuration for vision system"""
    image_folder: str = "captured_images"
    max_x_vel: float = 0.5
    max_y_vel: float = 0.5
    max_ang_vel: float = 2.0
    system_prompt: str = """
    You are a voice-based assistant for Spot, a robot dog in a robotics lab building.
    Your job is to:
        1. Answer any questions about what you see
        2. Refer to the environment as a room/space rather than an image
        3. Consider that you are looking at a stitched panoramic view from Spot's perspective(front cameras)
        
    Guidelines:
    - Mention which part of the building you are in. (kitchen, office, hallway, entrance)
    - Use natural, conversational language
    - Be friendly and informative
    """

class VisionSystem:
    """Handles Spot's visual perception and scene description capabilities"""
    
    def __init__(self, robot, command_client, image_client, config: VisionConfig = VisionConfig()):
        self.robot = robot
        self.command_client = command_client
        self.image_client = image_client
        self.config = config
        self.openai_client = OpenAI()
        
        # Define robot head postures for panoramic capture
        self.postures = [
            ("front", 0.0, 0.0, 0.0),
            ("front_left", 0.0, 0.0, -1.0),
            ("front_right", 0.0, 0.0, 1.0),
            ("top_right", 0.0, -1.0, 1.0),
            ("top_front", 0.0, -1.0, 0.0),
            ("top_left", 0.0, -1.0, -1.0),
        ]
        
        # Ensure image folder exists
        os.makedirs(self.config.image_folder, exist_ok=True)

    
    def describe_scene(self, prompt: str) -> str:
        """
        Capture and describe the current scene
        
        Args:
            prompt: User's question about the scene
            
        Returns:
            str: AI-generated description of the scene
        """
        try:
            # Stand and stabilize
            cmd = RobotCommandBuilder.synchro_stand_command()
            self.command_client.robot_command(command=cmd)
            time.sleep(1)
            
            # Capture panoramic view
            self._capture_panorama()
            
            # Stitch images
            stitched_image_path = "stitched_image.jpg"
            self._stitch_images(self.config.image_folder, stitched_image_path)
            
            # Get AI description
            return self._get_scene_description(stitched_image_path, prompt)
            
        except Exception as e:
            return f"I had trouble processing the visual information: {str(e)}"
            
    def _capture_panorama(self):
        """Capture panoramic images using different head positions"""
        speed_limit = SE2VelocityLimit(
            max_vel=SE2Velocity(
                linear=Vec2(x=self.config.max_x_vel, y=self.config.max_y_vel),
                angular=self.config.max_ang_vel
            )
        )
        
        for posture_name, roll, pitch, yaw in self.postures:
            # Position head
            rotation = geometry.EulerZXY(yaw=yaw, pitch=pitch, roll=roll)
            cmd = RobotCommandBuilder.synchro_stand_command(footprint_R_body=rotation)
            self.command_client.robot_command(command=cmd, timeout=3.0)
            time.sleep(2)
            
            # Capture from both cameras
            for camera, suffix in [("frontleft_fisheye_image", "FL"), ("frontright_fisheye_image", "FR")]:
                self._capture_single_image(camera, f"{self.config.image_folder}/{posture_name}_{suffix}.jpg")

    def _capture_single_image(self, image_source: str, save_path: str):
        """Capture single image from specified camera"""
        image_response = self.image_client.get_image_from_sources([image_source])
        image = Image.open(io.BytesIO(image_response[0].shot.image.data))
        image.save(save_path)

    def _stitch_images(self, image_folder: str, output_path: str):
        """Stitch multiple images into panorama"""
        image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.jpg')]
        images = [cv2.imread(path) for path in image_paths]
        
        if any(img is None for img in images):
            raise ValueError("Failed to load one or more images")
            
        images = [cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) for img in images]
        stitcher = cv2.Stitcher_create()
        status, stitched = stitcher.stitch(images)
        
        if status != cv2.Stitcher_OK:
            raise RuntimeError("Failed to stitch images")
            
        cv2.imwrite(output_path, stitched)

    def _get_scene_description(self, image_path: str, prompt: str) -> str:
        """Get AI description of the scene"""
        with open(image_path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode('utf-8')
            
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": self.config.system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        )
        
        return response.choices[0].message.content