"""
contains everything, that is used for recognising waving gesture
"""
# TODO Improve waving recognition -> changes to often to no gesture instead of waving;
#  especially when in the distorted part of the image

import numpy as np
import time
from project.gesture_recognition.pose_points_data import PosePoints

SHOULDER_ANGLE_THRESHOLD = [45,140]
FOREARM_Y_NORM_THRESHOLD = -0.4
UPPERARM_X_NORM_THRESHOLD = 0.2
ELBOW_ANGLE_THRESHOLD = 120



class WavingRecognizer:
    def __init__(self, points: PosePoints):
        self.points = points

        # init winking variables
        self.previous_hand_rel_speed = []
        self.previous_winking_calc_time = 0
        self.previous_hand_shoulder_distance = 0
        self.left_forarm_vec = np.array([0, 0, 0])
        self.left_upperarm_vec = np.array([0, 0, 0])
        self.left_upperbody_vec = np.array([0, 0, 0])
        self.shoulder_vec = np.array([0, 0, 0])
        self.angle_left_elbow = 0
        self.angle_left_shoulder = 0
        self.avg_hand_rel_speed = 0
        #averaging for last waving pos check
        self.last_shoulder_angle = []
        self.last_forarm_y_norm = []
        self.last_upperarm_x_norm = []
        self.last_elbow_angle = []

        self.waving_counter = 0


    def calc_waving_stuff(self):
        """
        Calculate all necessary things for waving gesture
        """
        MAX_PREVIOUS_DATA = 10  # number of previous speeds that are averaged
        # vectors of body edges
        self.left_forarm_vec = self.points.left_wrist - self.points.left_elbow
        self.left_upperarm_vec = self.points.left_elbow - self.points.left_shoulder
        self.left_upperbody_vec = self.points.left_shoulder - self.points.left_hip
        self.shoulder_vec = self.points.right_shoulder - self.points.left_shoulder
        # angle calculations -> calc cosine from cos angle = a*b/(|a|+|b|) and arccos to angle
        # both vectors must point away from joint
        # TODO at the moment test with projection on xy-plane
        cos_angle_left_elbow = np.dot(self.left_forarm_vec[:2], - self.left_upperarm_vec[:2]) / (
                    np.linalg.norm(self.left_forarm_vec[:2]) * np.linalg.norm(self.left_upperarm_vec[:2]))
        self.angle_left_elbow = np.arccos(cos_angle_left_elbow) * 180 / np.pi
        cos_angle_left_shoulder = np.dot(- self.left_upperbody_vec, self.left_upperarm_vec) / (
                    np.linalg.norm(self.left_upperbody_vec) * np.linalg.norm(self.left_upperarm_vec))
        self.angle_left_shoulder = np.arccos(cos_angle_left_shoulder) * 180 / np.pi
        # angle left elbow is funky???
        """ Hand speed calc """
        new_time = time.time()  # get time in seconds (float) since program start
        # only empty list of previous hand speeds when the max data length is reached
        if len(self.previous_hand_rel_speed) >= MAX_PREVIOUS_DATA:
            self.previous_hand_rel_speed.pop(0)
        # Hand shoulder distance in x direction normalized to distance between left and right shoulder
        new_hand_shoulder_distance = (self.points.left_wrist[0] - self.points.left_shoulder[0]) / np.linalg.norm(self.shoulder_vec)
        # speed is the difference between last and new hand-shoulder distance divided by the time difference
        new_hand_rel_speed = np.abs(new_hand_shoulder_distance - self.previous_hand_shoulder_distance) / (new_time - self.previous_winking_calc_time)
        self.previous_hand_rel_speed.append(new_hand_rel_speed)
        # avg speed is calculated as norm of all previous speeds
        self.avg_hand_rel_speed = 0
        if self.previous_hand_rel_speed:
            self.avg_hand_rel_speed = 1 / len(self.previous_hand_rel_speed) * np.linalg.norm(self.previous_hand_rel_speed)
        # set previous values to new values
        self.previous_winking_calc_time = new_time
        self.previous_hand_shoulder_distance = new_hand_shoulder_distance
        #print(f"Dist: {new_hand_shoulder_distance}, Previous: {self.previous_hand_rel_speed}, norm: {self.avg_hand_rel_speed}")

        # Averaging for checking waving position
        self.last_shoulder_angle.append(self.angle_left_shoulder)
        self.last_elbow_angle.append(self.angle_left_elbow)
        self.last_forarm_y_norm.append(self.left_forarm_vec[1] / np.linalg.norm(self.left_forarm_vec))
        self.last_upperarm_x_norm.append(self.left_upperarm_vec[0] / np.linalg.norm(self.left_upperarm_vec))

        if len(self.last_shoulder_angle) > 3:
            self.last_shoulder_angle.pop(0)
            self.last_elbow_angle.pop(0)
            self.last_forarm_y_norm.pop(0)
            self.last_upperarm_x_norm.pop(0)

    def check_waving(self)->bool:
        if (SHOULDER_ANGLE_THRESHOLD[0] < np.average(np.array(self.last_shoulder_angle)) < SHOULDER_ANGLE_THRESHOLD[
            1] and  # shoulder angle check
                np.average(np.array(
                    self.last_forarm_y_norm)) < FOREARM_Y_NORM_THRESHOLD and  # check if forearm is pointed upwards
                np.average(np.array(
                    self.last_upperarm_x_norm)) > UPPERARM_X_NORM_THRESHOLD and  # check if upper arm is pointed to the left
                np.average(np.array(self.last_elbow_angle)) < ELBOW_ANGLE_THRESHOLD  # elbow angle check
        ):
            return True
        else:
            return False

    def waving_start_debug_print(self):
        # print(f"forarm vec: {self.left_forarm_vec} Forarm y: {self.left_forarm_vec[1] / np.linalg.norm(self.left_forarm_vec)}  Shoulder: {self.angle_left_shoulder}")
        # print(f"Previous: {self.previous_hand_rel_speed}, norm: {self.avg_hand_rel_speed}")

        print(f"Shoulder: {np.average(np.array(self.last_shoulder_angle))} ,",
              f"Forarm y: {np.average(np.array(self.last_forarm_y_norm))} ,",
              f"Upperarm x: {np.average(np.array(self.last_upperarm_x_norm))} ,"
              f"Elbow: {np.average(np.array(self.last_elbow_angle))}")
