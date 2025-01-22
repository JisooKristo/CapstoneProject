# LateralRaises.py

import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
import sys

# Initialize MediaPipe Pose and Drawing utils
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Variable to set the correct angle range for lateral raises
correct_angle_min = 100  # Minimum angle for a valid raise (default 100)
correct_angle_max = 135  # Maximum angle for a valid raise (default 135)
incorrect_angle_threshold = 136  # Angle above which it's considered incorrect (default 136)

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point (hip)
    b = np.array(b)  # Mid point (shoulder)
    c = np.array(c)  # End point (wrist)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Function to close the application
def done_callback():
    global running
    running = False
    root.quit()

# Initialize variables for counting lateral raises
right_raise_count = 0
left_raise_count = 0
right_incorrect_count = 0
left_incorrect_count = 0
right_arm_state = 'down'  # Initial state is 'down'
left_arm_state = 'down'
right_arm_incorrect = False  # Track if right arm has gone into incorrect range
left_arm_incorrect = False  # Track if left arm has gone into incorrect range

# Initialize Tkinter window
root = tk.Tk()
root.title("Lateral Raise Counter")
root.geometry("720x960")

# Create a frame for the video feed
video_frame = tk.Frame(root)
video_frame.pack()

# Create label to display the video feed
video_label = Label(video_frame)
video_label.pack()

# Create labels for the right and left raise counts
right_raise_label = Label(root, text="Right Arm Raises: 0", font=("Helvetica", 12))
left_raise_label = Label(root, text="Left Arm Raises: 0", font=("Helvetica", 12))
right_incorrect_label = Label(root, text="Right Incorrect Raises: 0", font=("Helvetica", 12))
left_incorrect_label = Label(root, text="Left Incorrect Raises: 0", font=("Helvetica", 12))

# Start with the "Left" counter and labels visible
right_raise_label.pack_forget()  # Hide initially
left_raise_label.pack(pady=5)
right_incorrect_label.pack_forget()  # Hide initially
left_incorrect_label.pack(pady=5)

# Create "Done" button at the bottom of the window
done_button = tk.Button(root, text="Done", command=done_callback)
done_button.pack(side="bottom", pady=10)

# Start capturing video from the webcam
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("train model/lateral_raise.mp4")
running = True

# Initialize the MediaPipe Pose object globally
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Determine if left or right side was selected
selected_side = sys.argv[1] if len(sys.argv) > 1 else "left"  # Default to left if not provided

# Show relevant counter and hide others based on selected side
def update_curl_labels():
    if selected_side == "right":
        left_raise_label.pack_forget()  # Hide left raise count
        right_raise_label.pack(pady=5)  # Show right raise count
        left_incorrect_label.pack_forget()  # Hide left incorrect raise count
        right_incorrect_label.pack(pady=5)  # Show right incorrect raise count
    else:
        right_raise_label.pack_forget()  # Hide right raise count
        left_raise_label.pack(pady=5)  # Show left raise count
        right_incorrect_label.pack_forget()  # Hide right incorrect raise count
        left_incorrect_label.pack(pady=5)  # Show left incorrect raise count

# Initialize the selected side
update_curl_labels()

def show_frame():
    global right_raise_count, left_raise_count, right_incorrect_count, left_incorrect_count
    global right_arm_state, left_arm_state, right_arm_incorrect, left_arm_incorrect

    if not running:
        cap.release()
        cv2.destroyAllWindows()
        return

    ret, frame = cap.read()
    if not ret:
        return

    # Convert the BGR frame to RGB for processing
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Detect the pose and landmarks
    results = pose.process(image)

    # Convert back to BGR for OpenCV
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Get the landmark positions if detected
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Get coordinates for right arm (shoulder, hip, wrist)
        right_shoulder = [landmarks[12].x * frame.shape[1], landmarks[12].y * frame.shape[0]]
        right_hip = [landmarks[24].x * frame.shape[1], landmarks[24].y * frame.shape[0]]
        right_wrist = [landmarks[16].x * frame.shape[1], landmarks[16].y * frame.shape[0]]

        # Get coordinates for left arm (shoulder, hip, wrist)
        left_shoulder = [landmarks[11].x * frame.shape[1], landmarks[11].y * frame.shape[0]]
        left_hip = [landmarks[23].x * frame.shape[1], landmarks[23].y * frame.shape[0]]
        left_wrist = [landmarks[15].x * frame.shape[1], landmarks[15].y * frame.shape[0]]

        # Calculate angles for both arms
        right_arm_angle = calculate_angle(right_hip, right_shoulder, right_wrist)
        left_arm_angle = calculate_angle(left_hip, left_shoulder, left_wrist)

        # If right arm is selected, draw the right arm angle on the video frame
        if selected_side == "right":
            # Display the right arm angle on the video frame next to the shoulder
            cv2.putText(image, f"Angle: {int(right_arm_angle)}",
                        (int(right_shoulder[0]) + 10, int(right_shoulder[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # If left arm is selected, draw the left arm angle on the video frame
        if selected_side == "left":
            # Display the left arm angle on the video frame next to the shoulder
            cv2.putText(image, f"Angle: {int(left_arm_angle)}",
                        (int(left_shoulder[0]) + 10, int(left_shoulder[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Detect raises for the right arm
        if selected_side == "right":
            if right_arm_angle >= incorrect_angle_threshold and not right_arm_incorrect:  # Exceeds the correct range (incorrect rep)
                right_incorrect_count += 1
                right_incorrect_label.config(text=f"Right Incorrect Raises: {right_incorrect_count}")  # Update incorrect count
                right_arm_incorrect = True  # Mark as incorrect, prevent further correct counting
            elif correct_angle_min <= right_arm_angle < correct_angle_max and not right_arm_incorrect:  # Valid rep range (correct)
                if right_arm_state == 'down':  # The arm is moving up and within the range
                    right_arm_state = 'up'  # Track that the arm is up
            elif right_arm_angle < 20:  # Reset the state if the angle is very low (near the resting position)
                if right_arm_state == 'up' and not right_arm_incorrect:  # Only count as a valid rep if the arm is going down
                    right_raise_count += 1
                    right_raise_label.config(text=f"Right Arm Raises: {right_raise_count}")  # Update raise count
                right_arm_state = 'down'  # Ready for a new rep
                right_arm_incorrect = False  # Reset the incorrect rep flag

        # Detect raises for the left arm
        if selected_side == "left":
            if left_arm_angle >= incorrect_angle_threshold and not left_arm_incorrect:  # Exceeds the correct range (incorrect rep)
                left_incorrect_count += 1
                left_incorrect_label.config(text=f"Left Incorrect Raises: {left_incorrect_count}")  # Update incorrect count
                left_arm_incorrect = True  # Mark as incorrect
            elif correct_angle_min <= left_arm_angle < correct_angle_max and not left_arm_incorrect:  # Valid rep range (correct)
                if left_arm_state == 'down':  # The arm is moving up and within the range
                    left_arm_state = 'up'  # Track that the arm is up
            elif left_arm_angle < 20:  # Reset the state if the angle is very low (near the resting position)
                if left_arm_state == 'up' and not left_arm_incorrect:  # Only count as a valid rep if the arm is going down
                    left_raise_count += 1
                    left_raise_label.config(text=f"Left Arm Raises: {left_raise_count}")  # Update raise count
                left_arm_state = 'down'  # Ready for a new rep
                left_arm_incorrect = False  # Reset the incorrect rep flag

        # Draw landmarks on the image
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Convert the frame to an image format that Tkinter can use
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)

    # Update the label with the image
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    # Continue to call show_frame
    video_label.after(10, show_frame)

# Start showing frames
show_frame()

# Start Tkinter main loop
root.mainloop()
