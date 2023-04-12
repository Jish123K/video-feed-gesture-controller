import cv2

from cvzone.HandTrackingModule import HandDetector

import time

if __name__ == '__main__':

    capture = cv2.VideoCapture(0)

    detector = HandDetector(detectionCon=0.8, maxHands=2)

    previous_time = 0

    while True:

        success, image = capture.read()

        hands, image = detector.findHands(image)

        # Calculate Frame Rate

        current_time = time.time()

        frame_rate = 1 / (current_time - previous_time)

        previous_time = current_time

        # Frame Rate Text

        cv2.putText(

            img=image,

            text=f"FrameRate: {round(frame_rate, 1)}",

            org=(10, 20),

            fontFace=cv2.FONT_HERSHEY_SIMPLEX,

            fontScale=0.5,

            color=(255, 255, 255),

            thickness=1,

        )

        # Hand Count Text

        cv2.putText(

            img=image,

            text=f"Hands: {len(hands)}",

            org=(10, 35),

            fontFace=cv2.FONT_HERSHEY_SIMPLEX,

            fontScale=0.5,

            color=(255, 255, 255),

            thickness=1,

        )

        # Finger Count Text

        finger_count = 0

        for hand in hands:

            finger_count += detector.fingersUp(hand)

        cv2.putText(

            img=image,

            text=f"Fingers: {finger_count}",

            org=(10, 50),

            fontFace=cv2.FONT_HERSHEY_SIMPLEX,

            fontScale=0.5,

            color=(255, 255, 255),

            thickness=1,

        )

        cv2.imshow("Hand Tracker", image)

        cv2.waitKey(1)

