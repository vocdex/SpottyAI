from gesture_recognition.pose_points_data import PosePoints

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


class GestureRecognition:
    def __init__(self):
        """
        constructor to create GestureRecognition object

        Args:
            robot_interaction: object of robot interaction class, that handles the interaction with spot
        """
        # numpy arrays for body marks [x, y, z]
        self.points = PosePoints()

    def get_gesture(self, pose_landmarks, last_state: GestureState) -> GestureState:
        if pose_landmarks:
            # Save pose points (3D) to np.array
            # this is done in function of the dataclass
            self.points.new_data(pose_landmarks)

            # # calculate all vectors and angles needed for winking recognition
            # if self.state != State.CIRCLE_START and self.state != State.PUSH_UP:
            #     self.waving.calc_waving_stuff()
            #     # self.waving.waving_start_debug_print()

            # push_up_result = self.push_up.is_push_up()
            # # transition from no gesture to a gesture
            # if self.state == State.NO_GESTURE:
            #     # check for push-up state
            #     if push_up_result != self.push_up.PushUpResults.NO:
            #         # self.waving.waving_start_debug_print()
            #         self.state = State.PUSH_UP
            #         return "PUSH_UP"
            #     # if the arm is in the waving position, get to waving start state
            #     elif self.waving.is_waving_pos(self.state):
            #         self.state = State.WAVING_START
            #         return "WAVING_START"
            #     # elif self.circle.is_circle_position(self.state):
            #     #     self.state = State.CIRCLE_START
            #     #     return "CIRCLE_START"
            #     # elif self.helicopter(self.s, self.e):  # ckeck for starting position
            #     #     self.state = State.HELICOPTER
            #     #     self.s = 0  # Idee: Soll helfen, falls helicopter() ungeplant verlassen wurde. IDEE NICHT ÜBERPRÜFT
            #     #     return "HELICOPTER"
            #     return "NO_GESTURE"

            # # transition from PUSH_UP to NO_GESTURE
            # elif (
            #     self.state == State.PUSH_UP
            #     and push_up_result == self.push_up.PushUpResults.NO
            # ):
            #     self.state = State.NO_GESTURE
            #     return "NO_GESTURE"

            # # transition from WAVING_START
            # elif self.state == State.WAVING_START:
            #     # to NO_GESTURE

            #     if not self.waving.is_waving_pos(self.state):
            #         # check whether arm is still in waving position
            #         self.state = State.NO_GESTURE
            #         self.waving_counter = 0
            #         return "NO_GESTURE"

            #     # AND WAVING
            #     elif self.waving.avg_hand_rel_speed > 0.25:
            #         # check for waving motion -> when relative speed is high enough for 3 times go to waving state
            #         self.waving_counter = self.waving_counter + 1
            #         if self.waving_counter > 3:
            #             self.waving_counter = 0
            #             self.state = State.WAVING
            #             return "WAVING"

            #     return "WAVING_START"

            # # transition from WAVING to NO_GESTURE and WAVING_START and WALK
            # elif self.state == State.WAVING:
            #     print(f"Speed: {self.waving.avg_hand_rel_speed}")
            #     # check whether arm is still in waving position
            #     if not self.waving.is_waving_pos(self.state):
            #         self.state = State.NO_GESTURE
            #         return "NO_GESTURE"

            #     # check for waving motion
            #     elif self.waving.avg_hand_rel_speed < 0.25:
            #         self.state = State.WAVING_START
            #         print(f"Speed: {self.waving.avg_hand_rel_speed}")
            #         return "WAVING_START"

            #     return "WAVING"

        return GestureState.NO_GESTURE
