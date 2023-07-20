import time

from statemachine import StateMachine, State
import logging


from project.gesture_recognition.gesture_recognition import GestureRecognition, GestureState
from project.gesture_recognition.body_tracking import BodyTracking, HumanLocation
from project.gesture_recognition.robot_interaction import RobotInteraction

class GRSM(StateMachine):
    """Gesture recognition state machine"""

    # states
    no_human = State(initial=True)
    human_no_gesture = State()
    gesture_recognized = State()
    gesture_action = State()

    # state transitions
    detected_human = no_human.to(human_no_gesture)
    lost_human = human_no_gesture.to(no_human)
    recognition = human_no_gesture.to(gesture_recognized)
    processing_gesture_action = gesture_recognized.to(gesture_action)
    gesture_aborted = gesture_recognized.to(no_human)
    completed_gesture_action = gesture_action.to(no_human)
    #

    def __init__(self, robot, config, motion_client, state_client):
        super(GRSM, self).__init__()

        self.config = config

        if not hasattr(self.config, "track_human__allow_new_stand"):
            self.config.track_human__allow_new_stand = False

        self.gesture_recognition = GestureRecognition()
        self.robot_interaction = RobotInteraction(robot, motion_client, state_client)

        self.body_tracking = BodyTracking(robot, config, motion_client, state_client)

        self.gesture = GestureState.NO_GESTURE
        self.human_location = HumanLocation.PERSON_ON_RIGHT

        self.state_msg = "starting up"

    # When entering states
    def on_enter_no_human(self):
        self.state_msg = "no human -> searching for target..."
        logging.info(self.state_msg)

    def on_enter_human_no_gesture(self):
        self.state_msg = "human found. Awaiting command ..."
        logging.info(self.state_msg)

    def on_enter_gesture_recognized(self):
        self.state_msg = "Hurray! Gesture was recognized."
        logging.info(self.state_msg)

    def on_enter_gesture_action(self):
        self.state_msg = "Performing action according to recognized {} gesture ...".format(self.get_gesture())
        logging.info(self.state_msg)

    def on_exit_gesture_action(self):
        logging.info("Gesture action complete.")

    def get_gesture(self) -> str:
        return self.gesture.name

    def get_human_location(self) -> str:
        return self.human_location.name

    def get_state_msg(self):
        return self.state_msg

    def process_state(self, pose_landmarks):
        if self.current_state == self.no_human:
            self.gesture_recognition.reset_gesture()
            self.gesture = self.gesture_recognition.gesture_state

            if pose_landmarks:
                self.detected_human()
                return

            # no human found -> search
            if self.config.track_human:
                self.body_tracking.search_human(self.config.track_human__allow_new_stand)
                return

        if self.current_state == self.human_no_gesture:
            self.gesture_recognition.reset_gesture()
            self.gesture = self.gesture_recognition.gesture_state

            if not pose_landmarks:
                self.lost_human()
                return

            # check if gesture can be recognized
            if self.gesture_recognition.get_gesture(pose_landmarks) != GestureState.NO_GESTURE:
                self.recognition()
                return

            #  if no gesture can be recognized track human -> center him
            if self.config.track_human:
                self.human_location = self.body_tracking.get_human_location(
                    pose_landmarks
                )
                self.body_tracking.center_human(self.config.track_human__allow_new_stand)

        if self.current_state == self.gesture_recognized:
            # initial gesture was recognized -> continue recognition and switch to processing if certain
            self.gesture = self.gesture_recognition.get_gesture(pose_landmarks)

            if self.gesture == GestureState.WAVING:
                self.processing_gesture_action()
            
            if self.gesture == GestureState.NO_GESTURE:
                self.gesture_aborted()

            # no final gesture was recognized -> continue
            return

        if self.current_state == self.gesture_action:

            self.robot_interaction.wiggle(3)

            self.completed_gesture_action()
            return
