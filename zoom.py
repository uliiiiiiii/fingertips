import cv2
import mediapipe as mp
import os
from math import sqrt
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)
prev_distance = None  # To track zoom gesture distances
last_gesture_time = 0  # To track time between gestures

def calculate_distance(landmark1, landmark2):
    """Calculate Euclidean distance between two landmarks."""
    return sqrt((landmark1.x - landmark2.x)**2 + (landmark1.y - landmark2.y)**2)

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            landmarks = hand_landmark.landmark
            # Thumb tip (4) and index finger tip (8)
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            distance = calculate_distance(thumb_tip, index_tip)

            # Zoom Gesture Detection
            # if prev_distance is not None and time.time() - last_gesture_time > 0.3:  # Minimum time between gestures
            #     if distance < prev_distance - 0.05:  # Threshold for zoom-in
            #         print("Zoom In")
            #         os.system("xdotool key ctrl+plus")
            #         last_gesture_time = time.time() 
            #     elif distance > prev_distance + 0.05:  # Threshold for zoom-out
            #         print("Zoom Out")
            #         os.system("xdotool key ctrl+minus")
            #         last_gesture_time = time.time()  # Lock zoom gesture for 0.5s
            if distance > 0.1:
                print("Zoom Out")
                os.system("xdotool key ctrl+minus")
            elif distance < 0.03:
                print("Zoom In")
                os.system("xdotool key ctrl+plus")

            prev_distance = distance  # Update distance for next iteration

    cv2.imshow("Zoom Gesture Control", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
