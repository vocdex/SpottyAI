import time

from project.robot_wrapper.robot_wrapper import SpotRobotWrapper


class Robot(SpotRobotWrapper):
    def __init__(self, config):
        super(Robot, self).__init__(config)

    def init_stuff(self):
        # TODO: this part will be executed once during start up 
        self.stand_up()
        time.sleep(1)

    def loop_stuff(self):
        # TODO: this is part where your code is executed continuously

        if self.config.dbg_mode:
            # TODO: if you want to execute stuff only in the debug mode do it here
            pass
