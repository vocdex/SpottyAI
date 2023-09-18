import logging
import time

from project.robot_wrapper.robot_wrapper import SpotRobotWrapper


class Robot(SpotRobotWrapper):
    def __init__(self, config):
        super(Robot, self).__init__(config)

    def init_robot(self):
        # TODO: this part will be executed once during start up, any initialization should be done here
        if self.config.motors_on:
            self.stand_up()

        time.sleep(1)

        logging.info("Robot initialized")

    def loop_robot(self):
        # TODO: this is the part where your code is executed repeatedly, you can use this to control the robot or retrieve data continuously from the robot
        self.get_images()
        self.get_point_cloud()

        if self.config.dbg_mode:
            # TODO: if you want to execute stuff only in the debug mode do it here
            self.show_images()
            self.show_point_cloud()
