import cv2
import mediapipe as mp
import subprocess
import time
from math import sqrt

class HandController:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Allow up to 2 hands
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.prev_zoom_distance = None  # For zoom gesture tracking
        self.last_zoom_time = 0
        self.is_dragging = False
        self.prev_hand_center = None
        self.smoothing_factor = 0.5

    def calculate_distance(self, landmark1, landmark2):
        """Calculate Euclidean distance between two landmarks."""
        return sqrt((landmark1.x - landmark2.x)**2 + (landmark1.y - landmark2.y)**2)

    def is_fist(self, landmarks):
        """Check if fingers are curled (fist gesture)."""
        finger_tips = [8, 12, 16, 20]
        finger_bases = [5, 9, 13, 17]
        is_curled = True
        for tip, base in zip(finger_tips, finger_bases):
            if landmarks[tip].y < landmarks[base].y or landmarks[tip].z < landmarks[base].z - 0.1:
                is_curled = False
                break
        return is_curled

    def get_hand_center(self, landmarks):
        """Calculate the hand center based on key landmarks."""
        palm_landmarks = [0, 5, 9, 13, 17]
        x_mean = sum(landmarks[i].x for i in palm_landmarks) / len(palm_landmarks)
        y_mean = sum(landmarks[i].y for i in palm_landmarks) / len(palm_landmarks)
        return (x_mean, y_mean)

    def execute_xdotool_command(self, command):
        """Execute xdotool command."""
        try:
            subprocess.run(command, shell=True)
        except subprocess.SubprocessError as e:
            print(f"Error executing xdotool command: {e}")

    def control_molecule(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        h, w, _ = frame.shape
        left_hand = None
        right_hand = None

        if results.multi_handedness:
            for idx, hand_handedness in enumerate(results.multi_handedness):
                label = hand_handedness.classification[0].label
                if label == "Left":
                    left_hand = results.multi_hand_landmarks[idx]
                elif label == "Right":
                    right_hand = results.multi_hand_landmarks[idx]

        # Zoom (Left Hand)
        if left_hand:
            thumb_tip = left_hand.landmark[4]
            index_tip = left_hand.landmark[8]
            distance = self.calculate_distance(thumb_tip, index_tip)

            if self.prev_zoom_distance is not None and time.time() - self.last_zoom_time > 0.3:
                if distance < self.prev_zoom_distance - 0.05:
                    print("Zoom In")
                    self.execute_xdotool_command("xdotool click 5")
                    self.last_zoom_time = time.time()
                elif distance > self.prev_zoom_distance + 0.05:
                    print("Zoom Out")
                    self.execute_xdotool_command("xdotool click 4")
                    self.last_zoom_time = time.time()

            self.prev_zoom_distance = distance

        # Rotate (Right Hand)
        if right_hand:
            landmarks = right_hand.landmark
            hand_center = self.get_hand_center(landmarks)
            fist_detected = self.is_fist(landmarks)

            cx, cy = int(hand_center[0] * w), int(hand_center[1] * h)
            color = (0, 255, 0) if fist_detected else (0, 0, 255)
            cv2.circle(frame, (cx, cy), 10, color, -1)

            if fist_detected:
                if not self.is_dragging:
                    self.execute_xdotool_command("xdotool mousedown 1")
                    self.is_dragging = True
                    self.prev_hand_center = hand_center
                elif self.prev_hand_center:
                    dx = (hand_center[0] - self.prev_hand_center[0]) * w * 4
                    dy = (hand_center[1] - self.prev_hand_center[1]) * h * 3
                    dx = dx * self.smoothing_factor
                    dy = dy * self.smoothing_factor

                    if abs(dx) > 1 or abs(dy) > 1:
                        self.execute_xdotool_command(f"xdotool mousemove_relative -- {int(-dx)} {int(dy)}")
                        print(f"Moving by: dx={int(dx)}, dy={int(dy)}")

                    self.prev_hand_center = hand_center
            elif self.is_dragging:
                self.execute_xdotool_command("xdotool mouseup 1")
                self.is_dragging = False
                self.prev_hand_center = None

        elif self.is_dragging:
            self.execute_xdotool_command("xdotool mouseup 1")
            self.is_dragging = False
            self.prev_hand_center = None

        return frame

def main():
    cap = cv2.VideoCapture(0)
    controller = HandController()

    print("Starting in 5 seconds. Switch to molecule viewer window...")
    print("Use your left hand for zoom and right hand for rotation.")
    print("Press 'q' to quit.")
    time.sleep(5)

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = controller.control_molecule(frame)
            small_frame = cv2.resize(frame, (640, 480))
            cv2.imshow('Hand Control', small_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        if controller.is_dragging:
            controller.execute_xdotool_command("xdotool mouseup 1")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()