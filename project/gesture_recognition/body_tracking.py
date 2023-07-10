"""
class BodyTracking:
Contains the function track_human(self, pose_landmarks) for tracking a human
appearing on the frontright fisheye camera and changing the orientation of Spot,
so that Spot keeps facing the human directly.

For small movements of the person in the picture:
Spot is twisting his body with its feet remaining in the same place
(function command_yaw_angle(self, commanded_angle) is used for that).

For bigger movements of the person:
The robot is turning his whole body by moving its feet
(function command_turn(self, commanded_angle) is used for that).

If no person was detected, Spot turns around until a human appears on camera. If a human was
detected in a former step, the direction of "turning around" is set according to which side
this person left the video.
"""

from bosdyn.client.robot_command import RobotCommandBuilder
import time
from bosdyn.client.frame_helpers import (
    ODOM_FRAME_NAME,
    BODY_FRAME_NAME,
    get_se2_a_tform_b,
)
from bosdyn.client import math_helpers
import bosdyn.geometry
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus
from enum import Enum


class HumanLocation(Enum):
    PERSON_ON_RIGHT = 1
    PERSON_ON_LEFT = 0


ROTATION_ANGLE = 0.4
PITCH_ROTATION_ANGLE = 0.4


class BodyTracking:
    def __init__(self, robot, config, command_client, robot_state_client):
        self.left_hip_x = 0.0
        self.right_hip_x = 0.0
        self.robot = robot
        self.command_client = command_client
        self.robot_state_client = robot_state_client
        self.current_angle = 0
        self.hip_middle_shifted = 0
        self.time_last_movement = time.time()
        self.human_location = HumanLocation.PERSON_ON_LEFT
        self.config = config

    def command_yaw_angle(self, commanded_angle):
        """
        For small movements of the person in the picture, Spot is twisting his body with its feet remaining in the same place
        """
        footprint_R_body = bosdyn.geometry.EulerZXY(
            yaw=commanded_angle, roll=0.0, pitch=-PITCH_ROTATION_ANGLE
        )
        cmd = RobotCommandBuilder.synchro_stand_command(
            body_height=0.0, footprint_R_body=footprint_R_body
        )
        self.command_client.robot_command(cmd)
        self.time_last_movement = time.time()
        self.current_angle = commanded_angle

    def command_turn(self, commanded_angle):
        """
        For bigger movements of the person the robot is turning his whole body by moving its feet
        """
        transforms = (
            self.robot_state_client.get_robot_state().kinematic_state.transforms_snapshot
        )
        body_tform_goal = math_helpers.SE2Pose(x=0, y=0, angle=commanded_angle)
        # We do not want to command this goal in body frame because the body will move, thus shifting
        # our goal. Instead, we transform this offset to get the goal position in the output frame
        # (which will be either odom or vision).
        out_tform_body = get_se2_a_tform_b(transforms, ODOM_FRAME_NAME, BODY_FRAME_NAME)
        out_tform_goal = out_tform_body * body_tform_goal

        robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=out_tform_goal.x,
            goal_y=out_tform_goal.y,
            goal_heading=out_tform_goal.angle,
            frame_name=ODOM_FRAME_NAME,
            params=RobotCommandBuilder.mobility_params(stair_hint=False),
        )

        end_time = 10.0
        cmd_id = self.command_client.robot_command(
            lease=None, command=robot_cmd, end_time_secs=time.time() + end_time
        )
        self.current_angle = 0
        # Wait until the robot has reached the goal.
        while True:
            feedback = self.command_client.robot_command_feedback(cmd_id)
            mobility_feedback = (
                feedback.feedback.synchronized_feedback.mobility_command_feedback
            )

            if mobility_feedback.status != RobotCommandFeedbackStatus.STATUS_PROCESSING:
                print("Failed to reach the goal")
                self.time_last_movement = time.time()
                return False

            traj_feedback = mobility_feedback.se2_trajectory_feedback
            if (
                traj_feedback.status == traj_feedback.STATUS_AT_GOAL
                and traj_feedback.body_movement_status
                == traj_feedback.BODY_STATUS_SETTLED
            ):
                # command Spot to face upwards again
                footprint_R_body = bosdyn.geometry.EulerZXY(
                    yaw=0.0, roll=0.0, pitch=-PITCH_ROTATION_ANGLE
                )
                cmd = RobotCommandBuilder.synchro_stand_command(
                    body_height=0.0, footprint_R_body=footprint_R_body
                )
                self.command_client.robot_command(cmd)
                self.time_last_movement = time.time()
                return True

    def get_human_location(self, pose_landmarks) -> HumanLocation:
        """
        record whether a recorded person is located to the right or to the left side of Spot
        """
        if pose_landmarks:
            self.left_hip_x = pose_landmarks.landmark[23].x
            self.right_hip_x = pose_landmarks.landmark[24].x
            # calculate x-coordinate of human in the video from hip-coordinate
            hip_middle = (self.left_hip_x + self.right_hip_x) / 2
            # shift the received coordinate to get positive and negative values
            # *(-1) to adapt the angle according ot the orientation of the pitch angle of Spot's local frame
            self.hip_middle_shifted = (hip_middle - 0.5) * (-1)

            # record whether a recorded person is located to the right or to the left side of Spot
            if self.hip_middle_shifted > 0:
                # Person is on the right
                self.human_location = HumanLocation.PERSON_ON_RIGHT
            else:
                # Person is on the left
                self.human_location = HumanLocation.PERSON_ON_LEFT

        return self.human_location

    def center_human(self, allow_new_stand=True):
        # person was recorded on the camera
        # adjust the current commanded angle of Spot
        self.current_angle = self.current_angle + 0.3 * self.hip_middle_shifted

        # small movements by a recorded person cause Spot to twist its body (feet stay on the ground)
        # to keep facing the person
        if not allow_new_stand:
            self.current_angle = max(-1, min(self.current_angle, 1))
            self.command_yaw_angle(self.current_angle)
        else:
            if abs(self.current_angle) < 1:
                self.command_yaw_angle(self.current_angle)
            else:
                self.command_turn(0.2 * self.current_angle)

    def search_human(self, allow_new_stand=True):
        # No person was recorded on camera -> Spot searches for a person
        # print(self.time_last_movement)
        # print((time.time() - self.time_last_movement))
        if (time.time() - self.time_last_movement) > 3.0:
            if self.human_location == HumanLocation.PERSON_ON_RIGHT:
                commanded_angle = (
                    ROTATION_ANGLE  # person left the picture on the right side
                )
            else:
                commanded_angle = (
                    -ROTATION_ANGLE
                )  # person left the picture on the left side

            if not allow_new_stand:
                # if deviation is small enough, only the upper body rotates
                self.command_yaw_angle(commanded_angle)
                self.current_angle = commanded_angle
            else:
                if abs(self.current_angle) < 0.3:
                    # if deviation is small enough, only the upper body rotates
                    self.command_yaw_angle(commanded_angle)
                    self.current_angle = commanded_angle
                else:
                    # if deviation is too large command a full body rotation with feet repositioning
                    self.command_turn(1.5 * commanded_angle)
