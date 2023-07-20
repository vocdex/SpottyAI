"""
robot interaction: all movement commands to the robot are handled here
handle Boston Dynamic API
"""

import numpy as np
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
import bosdyn.client
import bosdyn.geometry

# for trajectory command
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus
from bosdyn.client.frame_helpers import *

import time

DEFAULT_BODY_PITCH_ANGLE = -15  # degree

MAX_WALKING_SPEED = 0.5  # 0.5   # m/s
WALK_CMD_DURATION = 2  # 0.6   # seconds
MAX_TURNING_SPEED = 20  # 20     # deg/sec
TURN_CMD_DURATION = 2


class RobotInteraction:
    def __init__(self, robot, command_client, state_client):
        self.robot = robot
        self.command_client = command_client
        self.state_client = state_client
        self.robot_enabled = True
        if robot is None or command_client is None:
            self.robot_enabled = False

        self.time_last_walking_cmd = 0
        self.time_last_turning_cmd = 0
        self.time_last_push_up_cmd = 0

        self.yaw = 0.0

    @classmethod
    def no_robot_action(cls):
        """
        constructor that creates a robot interaction object, that does no robot motion,
        e.g. for use with webcam without robot
        Returns:
            RobotInteraction object
        """
        obj = RobotInteraction(None, None)
        return obj

    def stop(self):
        """
        Stop robot as fast as possible
        Command to stop with minimal motion. If the robot is walking, it will transition to stand.
        If the robot is standing or sitting, it will do nothing.
        """
        if self.robot_enabled:
            cmd = RobotCommandBuilder.stop_command()
            self.command_client.robot_command(cmd)

    def stand(self, pitch_angle: float = DEFAULT_BODY_PITCH_ANGLE):
        """
        lets the robot stand
        Args:
            pitch_angle: angle of the robot in degree, with negative values the robot looks up
        """
        if self.robot_enabled:
            print("Start stand command")
            footprint_R_body = bosdyn.geometry.EulerZXY(
                yaw=self.yaw, roll=0.0, pitch=pitch_angle * np.pi / 180
            )
            cmd = RobotCommandBuilder.synchro_stand_command(
                body_height=0.0, footprint_R_body=footprint_R_body
            )
            self.command_client.robot_command(cmd)
            time.sleep(0.5)

    def push_up(self, push_up_angle):
        """
        mimic push-ups with spot
        Args:
            push_up_angle: angle of human push-up, that should be mimicked
        """
        if self.robot_enabled:
            if (time.time() - self.time_last_push_up_cmd) > 0.1:
                self.time_last_push_up_cmd = time.time()
                # extract offset
                # push_up_angle -= 10

                body_HEIGHT = (push_up_angle - 30) / 90 + 0.1

                footprint_R_body = bosdyn.geometry.EulerZXY(
                    yaw=self.yaw,
                    roll=0.0,
                    pitch=((push_up_angle * (-0.85)) * np.pi / 180),
                )
                cmd = RobotCommandBuilder.synchro_stand_command(
                    body_height=body_HEIGHT, footprint_R_body=footprint_R_body
                )
                self.command_client.robot_command(cmd)

    def walk(self, speed: float = MAX_WALKING_SPEED):
        """
        move robot forward, used for waving gesture
        Args:
            speed: speed that the robot should move with: positive: forward, negative: backwards
        """
        if self.robot_enabled:
            speed_bounded = speed
            if speed > MAX_WALKING_SPEED:
                speed_bounded = MAX_WALKING_SPEED
            elif speed < -MAX_WALKING_SPEED:
                speed_bounded = -MAX_WALKING_SPEED

            current_time = time.time()
            if (current_time - self.time_last_walking_cmd) > WALK_CMD_DURATION + 2:
                self.time_last_walking_cmd = time.time()
                # move robot forward
                # footprint_R_body = bosdyn.geometry.EulerZXY(yaw=self.yaw, roll=0.0, pitch=DEFAULT_BODY_PITCH_ANGLE*np.pi/180)
                # params = RobotCommandBuilder.mobility_params(body_height=0,
                #                                              footprint_R_body=footprint_R_body)
                cmd = RobotCommandBuilder.synchro_velocity_command(
                    speed_bounded, 0, 0
                )  # , params=params)
                self.command_client.robot_command(
                    cmd, end_time_secs=current_time + WALK_CMD_DURATION
                )

                time.sleep(WALK_CMD_DURATION)
                self.stand()

    def turn(self, direction: bool, angular_speed: float = MAX_TURNING_SPEED):
        """
        turn robot around itself
        Args:
            direction: True for left and False for right
            angular_speed: turning speed in degree/s

        """
        if self.robot_enabled:
            angular_speed_bounded = angular_speed
            if angular_speed > MAX_TURNING_SPEED:
                angular_speed_bounded = MAX_TURNING_SPEED
            if direction:
                angular_speed_bounded = -angular_speed_bounded  # turn

            current_time = time.time()
            if (current_time - self.time_last_turning_cmd) > TURN_CMD_DURATION + 2:
                self.time_last_turning_cmd = time.time()
                # footprint_R_body = bosdyn.geometry.EulerZXY(yaw=self.yaw, roll=0.0, pitch=DEFAULT_BODY_PITCH_ANGLE*np.pi/180)
                # params = RobotCommandBuilder.mobility_params(body_height=0,
                #                                              footprint_R_body=footprint_R_body)
                cmd = RobotCommandBuilder.synchro_velocity_command(
                    0, 0, angular_speed_bounded * np.pi / 180
                )  # , params=params)
                self.command_client.robot_command(
                    cmd, end_time_secs=current_time + TURN_CMD_DURATION
                )

                time.sleep(TURN_CMD_DURATION)
                self.stand()

    def disable_movement(self):
        self.robot_enabled = False

    def enable_movement(self):
        self.robot_enabled = True

    def set_current_yaw(self, yaw):
        self.yaw = yaw

    def turn_360(self, direction):
        if direction == "left":
            x = 1
        elif direction == "right":
            x = -1

        # command to turn with 40 deg/sec
        cmd = RobotCommandBuilder.synchro_velocity_command(0, 0, x * 40 * np.pi / 180)

        transforms = (
            self.state_client.get_robot_state().kinematic_state.transforms_snapshot
        )
        out_tform_body = get_se2_a_tform_b(transforms, ODOM_FRAME_NAME, BODY_FRAME_NAME)
        angle_start = out_tform_body.angle
        start_time = time.time()
        self.command_client.robot_command(cmd, end_time_secs=start_time + 5)
        flag_180 = False
        while True:
            transforms = (
                self.state_client.get_robot_state().kinematic_state.transforms_snapshot
            )
            out_tform_body = get_se2_a_tform_b(
                transforms, ODOM_FRAME_NAME, BODY_FRAME_NAME
            )
            angle_now = out_tform_body.angle
            # print(f"start:{angle_start}\t now:{angle_now}")
            if abs(abs(angle_now - angle_start) - np.pi) < 5 / 180 * np.pi:
                self.command_client.robot_command(cmd, end_time_secs=time.time() + 5)
                flag_180 = True
                # print("180")
            if (
                flag_180
                and abs(angle_now - angle_start) < 5 / 180 * np.pi
                or abs(abs(angle_now - angle_start) - 2 * np.pi) < 5 / 180 * np.pi
            ):
                # print("Leave angle")
                break
            if time.time() > start_time + 10:
                # print("Leave time")
                break

        self.stand()

        # Turn for 5 seconds with 40 deg/sec (≈ 180°):
        # self.command_client.robot_command(cmd, end_time_secs=time.time() + 5)
        # self.relative_move(0, 0, x * 2/3 * np.pi)
        # self.relative_move(0, 0, x * 2 / 3 * np.pi)
        # self.relative_move(0, 0, x * 2 / 3 * np.pi)
        # time.sleep(5)

        # Turn for further 5 seconds with 40 deg/sec (≈ 180°):
        # self.command_client.robot_command(cmd, end_time_secs=time.time() + 5)
        # self.relative_move(0, 0, x * np.pi)
        # time.sleep(5)

    def relative_move(self, x, y, theta, stairs=False):
        if self.robot_enabled:
            # Command the robot to go to the goal point in the specified frame. The command will stop at the
            # new position.
            # from movement planner -> get startup frame
            transforms = (
                self.state_client.get_robot_state().kinematic_state.transforms_snapshot
            )
            out_tform_body = get_se2_a_tform_b(
                transforms, ODOM_FRAME_NAME, BODY_FRAME_NAME
            )

            body_tform_goal = math_helpers.SE2Pose(x, y, theta)
            out_tform_goal = out_tform_body * body_tform_goal

            robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
                goal_x=out_tform_goal.x,
                goal_y=out_tform_goal.y,
                goal_heading=out_tform_goal.angle,
                frame_name=ODOM_FRAME_NAME,
                params=RobotCommandBuilder.mobility_params(stair_hint=stairs),
            )
            end_time = 10.0
            cmd_id = self.command_client.robot_command(
                lease=None, command=robot_cmd, end_time_secs=time.time() + end_time
            )
            # Wait until the robot has reached the goal.
            while True:
                feedback = self.command_client.robot_command_feedback(cmd_id)
                mobility_feedback = (
                    feedback.feedback.synchronized_feedback.mobility_command_feedback
                )
                if (
                    mobility_feedback.status
                    != RobotCommandFeedbackStatus.STATUS_PROCESSING
                ):
                    print("Failed to reach the goal")
                    return False
                traj_feedback = mobility_feedback.se2_trajectory_feedback
                if (
                    traj_feedback.status == traj_feedback.STATUS_AT_GOAL
                    and traj_feedback.body_movement_status
                    == traj_feedback.BODY_STATUS_SETTLED
                ):
                    print("Arrived at the goal.")
                    return True

    def wiggle(self, wiggles):
        body_HEIGHT = 0


        footprint_R_body = bosdyn.geometry.EulerZXY(
            yaw=0,
            roll=0.0,
            pitch=0,
        )

        cmd = RobotCommandBuilder.synchro_stand_command(
            body_height=body_HEIGHT, footprint_R_body=footprint_R_body
        )

        self.command_client.robot_command(cmd)

        time.sleep(0.5)

        for t in range(0, wiggles*2):
            footprint_R_body.roll = (-1)**t * 20*np.pi/180

            print(footprint_R_body.roll)

            cmd = RobotCommandBuilder.synchro_stand_command(
                body_height=body_HEIGHT, footprint_R_body=footprint_R_body
            )

            self.command_client.robot_command(cmd)
            time.sleep(0.5)
