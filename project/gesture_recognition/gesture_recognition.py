import logging

from project.gesture_recognition.pose_points_data import PosePoints
from project.gesture_recognition.waving_recognition import WavingRecognizer

from enum import Enum


class GestureState(Enum):
    """Enum for the state machine"""

    NO_GESTURE = 0
    PUSH_UP = 1
    WAVING_START = 2
    WAVING = 3
    WALK = 4
    CIRCLE_START = 5
    HELICOPTER = 7


FALSE_DETECTION_THRESHOLD = 30  # iterations


class GestureRecognition:
    def __init__(self):
        """
        constructor to create GestureRecognition object
        """
        self.gesture_state = GestureState.NO_GESTURE
        # numpy arrays for body marks [x, y, z]
        self.points = PosePoints()

        # recognizers
        self.waving_recognizer = WavingRecognizer(self.points)
        self.false_detection = 0

    def get_gesture(self, pose_landmarks) -> GestureState:
        if pose_landmarks:
            # Save pose points (3D) to np.array
            # this is done in function of the dataclass
            self.points.new_data(pose_landmarks)

            self.waving_recognizer.calc_waving_stuff()

            # transition from NO_GESTURE to WAVING_START
            if self.gesture_state == GestureState.NO_GESTURE:
                # initial waving recognized
                if self.waving_recognizer.check_waving():
                    self.false_detection = 0
                    self.waving_recognizer.waving_counter = 0
                    self.gesture_state = GestureState.WAVING_START
                    return self.gesture_state

            # transition from WAVING_START to WAVING and NO_GESTURE
            if self.gesture_state == GestureState.WAVING_START:

                # transition to waving
                if self.waving_recognizer.check_waving() and self.waving_recognizer.avg_hand_rel_speed > 0.2:
                    # check for waving motion -> when relative speed is high enough for 3 times go to waving state
                    self.waving_recognizer.waving_counter += 1
                    if self.waving_recognizer.waving_counter > 3:
                        self.gesture_state = GestureState.WAVING
                        return self.gesture_state

                # initial gesture recognition does not seem to be valid -> go back to NO_GESTURE
                if self.false_detection > FALSE_DETECTION_THRESHOLD:
                    self.gesture_state = GestureState.NO_GESTURE
                    return self.gesture_state

                self.false_detection += 1
                return GestureState.WAVING_START

            # transition from WAVING to NO_GESTURE
            if self.gesture_state == GestureState.WAVING:
                self.gesture_state = GestureState.NO_GESTURE
                return GestureState.NO_GESTURE

        return GestureState.NO_GESTURE

    def reset_gesture(self):
        self.gesture_state = GestureState.NO_GESTURE
