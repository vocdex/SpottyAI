import time
import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
import bosdyn.geometry

# from bosdyn.client.image import ImageClient # not used
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, ODOM_FRAME_NAME, get_se2_a_tform_b


def main():
    # Setup as in hello_spot
    bosdyn.client.util.setup_logging()
    sdk = bosdyn.client.create_standard_sdk('WalkaroundClient')
    robot = sdk.create_robot('192.168.80.3')
    # bosdyn.client.util.authenticate(robot)
    robot.authenticate("user", "c037gcf6n93f") # ausprobieren, ansonsten Zeile darüber
    robot.time_sync.wait_for_sync()

    assert not robot.is_estopped(), "Robot is estopped. Please use an external E-Stop client, " \
                                    "such as the estop SDK example, to configure E-Stop."

    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        # Powering robot on
        robot.logger.info("Powering on robot... This may take several seconds.")
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), "Robot power on failed."
        robot.logger.info("Robot powered on.")

        # TODO: If necessary: Selfright instead of stand up

        # Stand up
        robot.logger.info("Commanding robot to stand...")
        command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        blocking_stand(command_client, timeout_sec=10)
        robot.logger.info("Robot standing.")
        time.sleep(3)

        # TODO: Move somewhere
        robot.logger.info("Commanding robot to move 1 m in each direction and turn 90 deg.")
        # Building command:  goal_x_rt_body, goal_y_rt_body, goal_heading_rt_body
        # What is the frame_tree_snapshot and why is it needed?
        # Robot did not move, but the program ran through
        # cmd = RobotCommandBuilder. /
        # synchro_trajectory_command_in_body_frame(1, 1, 1.57, robot.get_frame_tree_snapshot())

        # Building command: v_x, v_y, v_rot
        # Trying the velocity command only with angular velocity for safety reasons
        # How long will the command be executed? -> until specified end time?
        # cmd = RobotCommandBuilder.synchro_velocity_command(0, 0, 1)

        # From the docs: "Issue a command to the robot synchronously." (An async version exists, too.)
        # Threw an exception: "The command was received after its max_duration had already passed."
        # The exceptions rise without the end time parameter, too.
        # command_client.robot_command(cmd, 3)  # End time for the command in secs: 3
        # command_client.robot_command(cmd)

        # (InvalidRequestError): Frame body is not allowed for body trajectory commands.
        # cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(1, 1, 1.57, BODY_FRAME_NAME)

        # Did not move anywhere, too
        # cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(1, 1, 1.57, ODOM_FRAME_NAME)

        # Falling back on the (deprecated) mobility commands:
        # cmd = RobotCommandBuilder.trajectory_command(1, 1, 1.57, ODOM_FRAME_NAME) # nothing
        # cmd = RobotCommandBuilder.velocity_command(1, 1, 1) # nothing

        # Trying to replicate the commands from the wasd example -> Works!

        # cmd = RobotCommandBuilder.synchro_velocity_command(v_x=0, v_y=0, v_rot=1)
        # VELOCITY_CMD_DURATION can not be more than 5 secs (Exception: End time is too far in the future)
        # Note: Even the duration was set to 5 sec, the robot did move less than 5 sec.
        end_time_secs = time.time() + 2  # --> VELOCITY_CMD_DURATION

        # cmd = RobotCommandBuilder.synchro_velocity_command(0.5, -0.5, 0)
        # cmd = RobotCommandBuilder. /
        # synchro_trajectory_command_in_body_frame(0, -2, -1.57, robot.get_frame_tree_snapshot())
        # Was passiert wenn der Endpunkt innerhalb der end_time niht erreicht wird? Vorzeitiges stehenbleiben?
        cmd = RobotCommandBuilder.synchro_trajectory_command_in_body_frame(1, 0, 0, robot.get_frame_tree_snapshot())
        command_client.robot_command(cmd, end_time_secs)
        robot.logger.info("Robot finished moving")
        time.sleep(3)  # It is necessary to wait for the robot to finish its movement!
        '''
        end_time_secs = time.time() + 2
        cmd = RobotCommandBuilder.synchro_velocity_command(-0.5, 0.5, 0)
        command_client.robot_command(cmd, end_time_secs)
        robot.logger.info("Robot finished moving")
        time.sleep(3)
        '''

        # TODO: Sit
        robot.logger.info("Commanding robot to sit.")
        cmd = RobotCommandBuilder.synchro_sit_command()
        command_client.robot_command(cmd)
        robot.logger.info("Robot is sitting.")
        time.sleep(3)

        robot.power_off(cut_immediately=False, timeout_sec=20)
        assert not robot.is_powered_on(), "Robot power off failed."
        robot.logger.info("Robot safely powered off.")


if __name__ == '__main__':
    main()
