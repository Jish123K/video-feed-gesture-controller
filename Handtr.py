from typing import List, Tuple

import cvzone

import mediapipe as mp

import cv2

import numpy as np

# See "google.github.io/mediapipe/images/mobile/hand_landmarks.png" for point reference

FINGER_TIPS = [4, 8, 12, 16, 20]

MP_HANDS = mp.solutions.hands

class TrackerColors:

    """

    Color options for the HandTracker class.

    """

    def __init__(self, point_color: Tuple[int, int, int], connection_color: Tuple[int, int, int],

                 finger_tip_up_color: Tuple[int, int, int], finger_tip_down_color: Tuple[int, int, int]):

        self.point_color = point_color

        self.connection_color = connection_color

        self.finger_tip_up_color = finger_tip_up_color

        self.finger_tip_down_color = finger_tip_down_color

class HandTracker:

    """

    Class for processing an image and tracking the hands within it.

    """

    def __init__(self, colors: TrackerColors):

        self.colors = colors

        self.hand_count = 0

        self.finger_count = 0

        self.mp_hands = MP_HANDS(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        

    def process(self, image: np.ndarray):

        """

        Process the `image` for hands, manipulating the image with received information.

        """

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = self.mp_hands.process(image_rgb)

        self.hand_count = 0

        self.finger_count = 0

        if results.multi_hand_landmarks:

            self.hand_count = len(results.multi_hand_landmarks)

            height, width, _ = image.shape

            for hand_lm in results.multi_hand_landmarks:

                # Get average of hand's `MCP.y` positional values for calculating if finger is up

                MCPS_Y: List[float] = [

                    hand_lm.landmark[5].y, hand_lm.landmark[9].y, hand_lm.landmark[13].y, hand_lm.landmark[17].y]

                MCP_Y_AVG: float = sum(MCPS_Y) / len(MCPS_Y)

                # Draw hand landmarks and connections using cvzone library

                cvzone.cornerRect(image, (int(hand_lm.rect.xmin * width), int(hand_lm.rect.ymin * height)),

                                  (int(hand_lm.rect.xmax * width), int(hand_lm.rect.ymax * height)), 20, 2)

                # Draw finger tips using cvzone library

                for idx, lm in enumerate(hand_lm.landmark):

                    x, y = int(lm.x * width), int(lm.y * height)

                    is_up_color = self.colors.finger_tip_up_color

                    if idx in FINGER_TIPS:

                        if MCP_Y_AVG < lm.y:

                            is_up_color = self.colors.finger_tip_down_color

                        else:

                            self.finger_count += 1

                        cv2.circle(

                            img=image,

                            center=(x, y),

                            radius=10,

                            color=is_up_color,

                            thickness=2,

                            lineType=cv2.LINE_AA

                        )

