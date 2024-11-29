import logging
import time

from robot_wrapper.robot_wrapper import SpotRobotWrapper, Velocity2D
import numpy as np

class Robot(SpotRobotWrapper):
    def __init__(self, config):
        super(Robot, self).__init__(config)

    def init_robot(self):
        # TODO: this part will be executed once during start up, any initialization should be done here
        if self.motors_on:
            self.stand_up()

        logging.info("Robot initialized")

    def loop_robot(self):
        # TODO: this is the part where your code is executed repeatedly, you can use this to control the robot or retrieve data continuously from the robot
        # self.get_images()
        self.get_point_cloud()

        if self.config.dbg_mode:
            # TODO: if you want to execute stuff only in the debug mode do it here
            # self.show_images()
            self.show_point_cloud()
