import time

from project.robot_wrapper.robot_wrapper import SpotRobotWrapper


class Robot(SpotRobotWrapper):
    def __init__(self, config):
        super(Robot, self).__init__(config)

    def init_robot(self):
        # TODO: this part will be executed once during start up 
        # self.stand_up()
        time.sleep(1)

    def loop_robot(self):
        # TODO: this is part where your code is executed continuously
        self.get_images()

        if self.config.dbg_mode:
            pass
            # TODO: if you want to execute stuff only in the debug mode do it here

            # self.show_images()

        # self.show_point_cloud()
