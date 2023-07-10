"""
contains dataclass, that contains all pose points
"""

from dataclasses import dataclass
import numpy as np


# TODO all points

@dataclass
class PosePoints:
    """
    contains points and visibility of pose with mediapipe naming scheme
    as numpy arrays for body marks [x, y, z] and variables for visibility
    """
    left_foot_index = np.array([0, 0, 0])
    right_foot_index = np.array([0, 0, 0])
    left_shoulder = np.array([0, 0, 0])
    right_shoulder = np.array([0, 0, 0])
    left_wrist = np.array([0, 0, 0])
    right_wrist = np.array([0, 0, 0])
    left_elbow = np.array([0, 0, 0])
    right_elbow = np.array([0, 0, 0])
    left_hip = np.array([0, 0, 0])
    right_hip = np.array([0, 0, 0])

    right_wrist_visibility = 0
    left_wrist_visibility = 0
    right_foot_index_visibility = 0
    left_foot_index_visibility = 0
    right_shoulder_visibility = 0
    left_shoulder_visibility = 0

    def new_data(self, pose_landmarks):
        """
        Save pose points (3D) to np.array
        """
        # Save pose points (3D) to np.array
        self.left_foot_index = np.array(
            (pose_landmarks.landmark[31].x, pose_landmarks.landmark[31].y, pose_landmarks.landmark[31].z))
        right_foot_index = np.array(
            (pose_landmarks.landmark[32].x, pose_landmarks.landmark[32].y, pose_landmarks.landmark[32].z))
        self.left_shoulder = np.array(
            (pose_landmarks.landmark[11].x, pose_landmarks.landmark[11].y, pose_landmarks.landmark[11].z))
        self.right_shoulder = np.array(
            (pose_landmarks.landmark[12].x, pose_landmarks.landmark[12].y, pose_landmarks.landmark[12].z))
        self.left_wrist = np.array(
            (pose_landmarks.landmark[15].x, pose_landmarks.landmark[15].y, pose_landmarks.landmark[15].z))
        self.right_wrist = np.array(
            (pose_landmarks.landmark[16].x, pose_landmarks.landmark[16].y, pose_landmarks.landmark[16].z))
        self.left_elbow = np.array(
            (pose_landmarks.landmark[13].x, pose_landmarks.landmark[13].y, pose_landmarks.landmark[13].z))
        self.right_elbow = np.array(
            (pose_landmarks.landmark[14].x, pose_landmarks.landmark[14].y, pose_landmarks.landmark[14].z))
        self.left_hip = np.array(
            (pose_landmarks.landmark[23].x, pose_landmarks.landmark[23].y, pose_landmarks.landmark[23].z))
        self.right_hip = np.array(
            (pose_landmarks.landmark[24].x, pose_landmarks.landmark[24].y, pose_landmarks.landmark[24].z))

        self.right_wrist_visibility = pose_landmarks.landmark[16].visibility
        self.left_wrist_visibility = pose_landmarks.landmark[15].visibility
        self.right_foot_index_visibility = pose_landmarks.landmark[32].visibility
        self.left_foot_index_visibility = pose_landmarks.landmark[31].visibility
        self.right_shoulder_visibility = pose_landmarks.landmark[12].visibility
        self.left_shoulder_visibility = pose_landmarks.landmark[11].visibility

