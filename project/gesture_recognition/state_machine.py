from statemachine import StateMachine, State
import logging


from gesture_recognition.gesture_recognition import GestureRecognition, GestureState
from gesture_recognition.body_tracking import BodyTracking, HumanLocation


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
    completed_gesture_action = gesture_action.to(no_human)
    #

    def __init__(self, robot, config, motion_client, state_client):
        super(GRSM, self).__init__()

        self.config = config

        if not hasattr(self.config, "track_human__allow_new_stand"):
            self.config.track_human__allow_new_stand = False

        self.gesture_recognition = GestureRecognition()

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
        self.state_msg = "Performing action according to recognized gesture..."
        logging.info(self.state_msg)

    def after_gesture_action_complete(self):
        logging.info("Gesture action complete.")

    def get_gesture(self) -> str:
        return self.gesture.name

    def get_human_location(self) -> str:
        return self.human_location.name

    def get_state_msg(self):
        return self.state_msg
    def process_state(self, pose_landmarks):
        if self.current_state == self.no_human:
            if pose_landmarks:
                self.detected_human()
                return

            # no human found -> search
            if self.config.track_human:
                self.body_tracking.search_human(self.config.track_human__allow_new_stand)
                return

        if self.current_state == self.human_no_gesture:
            if not pose_landmarks:
                self.lost_human()
                return

            #  track human -> center him
            if self.config.track_human:
                self.human_location = self.body_tracking.get_human_location(
                    pose_landmarks
                )
                self.body_tracking.center_human(self.config.track_human__allow_new_stand)
                return

            if (
                self.gesture_recognition.get_gesture(pose_landmarks, self.gesture)
                == GestureState.NO_GESTURE
            ):
                return

            self.recognition()
            return

        if self.current_state == self.gesture_recognized:
            # TODO process more complex gestures

            self.processing_gesture_action()
            return

        if self.current_state == self.gesture_action:
            # TODO block loop and do action

            self.completed_gesture_action()
            return
