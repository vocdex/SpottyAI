import time

from project.robot_wrapper.robot_wrapper import SpotRobotWrapper

from .image_processing import ImageProcessing

from .state_machine import GRSM


class Robot(SpotRobotWrapper):
    def __init__(self, config):
        super(Robot, self).__init__(config)
        self.grsm = None
        self.image_processing = None

    def init_stuff(self):

        self.stand_up()
        time.sleep(1)
        
        self.image_processing = ImageProcessing(self.robot, self.config)
        self.grsm = GRSM(self.robot, self.config, self.motion_client, self.state_client)

        time.sleep(1)

    def loop_stuff(self):
        pose_landmarks = self.image_processing.process_landmarks()[1]

        self.grsm.process_state(pose_landmarks)

        if self.config.dbg_mode:
            gesture = self.grsm.get_gesture()
            human_location = self.grsm.get_human_location()
            state_msg = self.grsm.get_state_msg()

            self.image_processing.annotate_gesture(gesture, human_location,state_msg)
            self.image_processing.show_image()
