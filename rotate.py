import cv2
import mediapipe as mp
import numpy as np
import subprocess
import time

class FistGrabController:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.is_dragging = False
        self.prev_hand_center = None
        self.smoothing_factor = 0.5
        
    def is_fist(self, landmarks):
        # Check if fingers are curled (typical fist gesture)
        finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
        finger_bases = [5, 9, 13, 17]  # Corresponding finger bases
        
        is_curled = True
        for tip, base in zip(finger_tips, finger_bases):
            # If tip is higher than base (y is smaller), finger is extended
            if landmarks[tip].y < landmarks[base].y:
                is_curled = False
                break
                
            # If tip is significantly in front of base, finger is extended
            if landmarks[tip].z < landmarks[base].z - 0.1:
                is_curled = False
                break
        
        return is_curled

    def get_hand_center(self, landmarks):
        # Use the center of palm as the control point
        palm_landmarks = [0, 5, 9, 13, 17]  # Wrist and finger bases
        x_mean = sum(landmarks[i].x for i in palm_landmarks) / len(palm_landmarks)
        y_mean = sum(landmarks[i].y for i in palm_landmarks) / len(palm_landmarks)
        return (x_mean, y_mean)

    def execute_xdotool_command(self, command):
        try:
            subprocess.run(command, shell=True)
        except subprocess.SubprocessError as e:
            print(f"Error executing xdotool command: {e}")

    def control_molecule(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        h, w, _ = frame.shape
        
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0].landmark
            hand_center = self.get_hand_center(landmarks)
            fist_detected = self.is_fist(landmarks)
            
            # Visualize hand center and state
            cx, cy = int(hand_center[0] * w), int(hand_center[1] * h)
            color = (0, 255, 0) if fist_detected else (0, 0, 255)
            cv2.circle(frame, (cx, cy), 10, color, -1)
            
            if fist_detected:
                if not self.is_dragging:
                    # Start drag
                    self.execute_xdotool_command("xdotool mousedown 1")
                    self.is_dragging = True
                    self.prev_hand_center = hand_center
                elif self.prev_hand_center:
                    # Calculate movement
                    dx = (hand_center[0] - self.prev_hand_center[0]) * w * 1.5
                    dy = (hand_center[1] - self.prev_hand_center[1]) * h * 1.5
                    
                    # Apply smoothing
                    dx = dx * self.smoothing_factor
                    dy = dy * self.smoothing_factor
                    
                    # Move if there's significant movement
                    if abs(dx) > 1 or abs(dy) > 1:
                        self.execute_xdotool_command(f"xdotool mousemove_relative -- {int(dx)} {int(dy)}")
                        print(f"Moving by: dx={int(dx)}, dy={int(dy)}")  # Debug output
                
                self.prev_hand_center = hand_center
            elif self.is_dragging:
                # Release drag
                self.execute_xdotool_command("xdotool mouseup 1")
                self.is_dragging = False
                self.prev_hand_center = None
        elif self.is_dragging:
            # Release if hand is lost
            self.execute_xdotool_command("xdotool mouseup 1")
            self.is_dragging = False
            self.prev_hand_center = None
        
        return frame

def main():
    cap = cv2.VideoCapture(0)
    controller = FistGrabController()
    
    print("Starting in 5 seconds. Switch to molecule viewer window...")
    print("Make a fist to grab and move the molecule")
    print("Press 'q' to quit")
    time.sleep(5)
    
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            frame = controller.control_molecule(frame)
            
            # Display the camera feed in a small window
            small_frame = cv2.resize(frame, (320, 240))
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