import cv2
import mediapipe as mp
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
prev_x = None
prev_y = None

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            landmarks = hand_landmark.landmark
            x = landmarks[8].x 
            y = landmarks[8].y 

            if prev_x is not None and prev_y is not None:
                dx = x - prev_x
                dy = y - prev_y

                if abs(dx) > abs(dy): 
                    if dx > 0.1:
                        print("Swipe Right")
                        os.system("xdotool key super+Right")
                    elif dx < -0.1:
                        print("Swipe Left")
                        os.system("xdotool key super+Left")
                elif abs(dy) > abs(dx): 
                    if dy < -0.1:
                        print("Swipe Up")
                        os.system("xdotool key super+Up")

            prev_x, prev_y = x, y

    cv2.imshow("Gesture Control", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
