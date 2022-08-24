# DISCLAIMER
# this code was provided by students of a previous semester.
# It should be seen as a possible way to process the mediapipe landmark data for a static case.
# You will need to develop your own process for recognizing dynamics gestures

import mediapipe as mp
import numpy as np

from dataclasses import dataclass


@dataclass
class mvc:
    """enum for possible detected gestures aka movement commands aka mvc """

    def __init__(self):
        pass
    NONE = 0
    HALT = 1
    COME = 2
    DOWN = 3
    UP = 4


def get_static_gesture(mpHandIn):
    """Processes a mediapipe hand and returns a possible gesture
  Args:
    mpHandIn: An mediapipe hand represented as numpy ndarray.

  Raises:
    ValueError: If the input hand is not of type mediapipe.framework.formats.landmark_pb2.LandmarkList

  Returns:
    an int, which can be interpreted as movementcommand of class mvc e.g. mvc.HALT
  """

    # TODO Umgang mit fehlenden Punkten an der Handbasis notwendig ??

    if (type(mpHandIn) != mp.framework.formats.landmark_pb2.LandmarkList and type(
            mpHandIn) != mp.framework.formats.landmark_pb2.NormalizedLandmarkList):
        raise ValueError('Input of gestrec is no mediapipe hand datatype')

    p0 = np.array((mpHandIn.landmark[0].x, mpHandIn.landmark[0].y, mpHandIn.landmark[0].z))  # hand/thumb base
    p2 = np.array(
        (mpHandIn.landmark[2].x, mpHandIn.landmark[2].y, mpHandIn.landmark[2].z))  # thumb the one 2 before the tip
    p3 = np.array(
        (mpHandIn.landmark[3].x, mpHandIn.landmark[3].y, mpHandIn.landmark[3].z))  # thumb the one before the tip
    p4 = np.array((mpHandIn.landmark[4].x, mpHandIn.landmark[4].y, mpHandIn.landmark[4].z))  # thumb tip
    p5 = np.array((mpHandIn.landmark[5].x, mpHandIn.landmark[5].y, mpHandIn.landmark[5].z))  # index base
    p6 = np.array((mpHandIn.landmark[6].x, mpHandIn.landmark[6].y, mpHandIn.landmark[6].z))  # index 2
    p7 = np.array((mpHandIn.landmark[7].x, mpHandIn.landmark[7].y, mpHandIn.landmark[7].z))  # index 3
    p8 = np.array((mpHandIn.landmark[8].x, mpHandIn.landmark[8].y, mpHandIn.landmark[8].z))  # index tip
    p9 = np.array((mpHandIn.landmark[9].x, mpHandIn.landmark[9].y, mpHandIn.landmark[9].z))  # middle base
    p12 = np.array((mpHandIn.landmark[12].x, mpHandIn.landmark[12].y, mpHandIn.landmark[12].z))  # middle tip
    p13 = np.array((mpHandIn.landmark[13].x, mpHandIn.landmark[13].y, mpHandIn.landmark[13].z))  # ring base
    p16 = np.array((mpHandIn.landmark[16].x, mpHandIn.landmark[16].y, mpHandIn.landmark[16].z))  # ring tip
    p17 = np.array((mpHandIn.landmark[17].x, mpHandIn.landmark[17].y, mpHandIn.landmark[17].z))  # small base
    P = np.stack((p0, p5, p9, p13, p17))
    PseuInv = np.matmul(np.linalg.inv(np.matmul(P.T, P)), P.T)
    normvek = np.matmul(PseuInv, np.ones(5))
    scalprd = np.dot(normvek, np.array((0, 0, 1))) / np.linalg.norm(normvek)
    # print(scalprd)

    # halt command: hand is flat an nearly orthogonal to the robots point of view
    if scalprd > 0.9:
        minfinlen = min(np.linalg.norm(p0[0:2] - p8[0:2]), np.linalg.norm(p0[0:2] - p12[0:2]),
                        np.linalg.norm(p0[0:2] - p16[0:2]))
        if minfinlen > 1.5 * np.linalg.norm(p0[0:2] - p5[0:2]):  # detect halt command
            print("HALT")
            return mvc.HALT

        # up command: thumb up
        maxfinlen = max(np.linalg.norm(p0[0:2] - p8[0:2]), np.linalg.norm(p0[0:2] - p12[0:2]),
                        np.linalg.norm(p0[0:2] - p16[0:2]))
        if (maxfinlen < 1.3 * np.linalg.norm(p0[0:2] - p5[0:2]) and (p3[1] - p4[1]) > 0.5 * (p2[1] - p3[1]) and (
                p3[1] - p4[1]) > 0.5 * np.linalg.norm(
            p3[0:2] - p2[0:2])):  # note to future self: the missing abs is correct
            # please remember: the coordinate system of images starts in the upper left corner
            print("UP")
            return mvc.UP

        # down command: index pointing downward
        maxmrlen = max(np.linalg.norm(p0[0:2] - p12[0:2]), np.linalg.norm(p0[0:2] - p16[0:2]))
        if maxmrlen < 1.25 * np.linalg.norm(p0[0:2] - p5[0:2]) and (p8[1] - p7[1]) > 0.5 * abs(p6[1] - p7[1]):
            print("DOWN")
            return mvc.DOWN

    # come command: index and thumb ring, proven by small distance between the two tips in 2D image
    if np.linalg.norm(p4[0:2] - p8[0:2]) < 0.15 * np.linalg.norm(p0[0:2] - p5[0:2]) and scalprd > 0.6:
        print("COME")
        return mvc.COME
    return mvc.NONE


