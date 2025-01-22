# HamstringCurls.py

import cv2
import mediapipe as mp
import numpy as np
import sys
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk

# Initialize MediaPipe Pose and Drawing utils
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Set the target angle for hamstring curls (default 90 degrees)
target_angle = 50  # Change this value to whatever the desired angle is for the exercise
angle_tolerance = 5  # Allowable deviation from the target angle (in degrees)

# Function to calculate the angle between three points (using the old code calculation method)
def calculate_angle(a, b, c):
    a = np.array(a)  # Point A (e.g., hip)
    b = np.array(b)  # Point B (e.g., knee)
    c = np.array(c)  # Point C (e.g., ankle)

    # Calculate the vectors
    vector_1 = a - b
    vector_2 = c - b

    # Compute the angle between the two vectors using arctan2
    angle = np.arctan2(vector_2[1], vector_2[0]) - np.arctan2(vector_1[1], vector_1[0])

    # Convert angle to degrees
    angle = angle * 180.0 / np.pi

    # Normalize the angle to be between 0 and 360 degrees
    if angle < 0:
        angle += 360

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Function to close the application
def done_callback():
    global running
    running = False
    root.quit()

# Function to pause/resume video when spacebar is pressed
def toggle_pause(event):
    global paused
    paused = not paused  # Toggle paused state

# Initialize variables for counting hamstring curls
right_leg_curl_count = 0
left_leg_curl_count = 0
right_leg_state = None  # 'up' or 'down'
left_leg_state = None

paused = False  # Variable to track whether the video is paused

# Initialize Tkinter window
root = tk.Tk()
root.title("Hamstring Curl Detection")
root.geometry("720x960")

# Create a frame for the video feed
video_frame = tk.Frame(root)
video_frame.pack()

# Create label to display the video feed
video_label = Label(video_frame)
video_label.pack()

# Create labels for the right and left leg curl counts
right_leg_label = Label(root, text="Right Leg Curls: 0", font=("Helvetica", 12))
right_leg_label.pack(pady=5)

left_leg_label = Label(root, text="Left Leg Curls: 0", font=("Helvetica", 12))
left_leg_label.pack(pady=5)

# Create a label for displaying the target angle
target_angle_label = Label(root, text=f"Target Angle: {target_angle}°", font=("Helvetica", 12))
target_angle_label.pack(pady=5)

# Create "Done" button at the bottom of the window
done_button = tk.Button(root, text="Done", command=done_callback)
done_button.pack(side="bottom", pady=10)

# Bind spacebar to toggle pause/resume
root.bind("<space>", toggle_pause)

# Start capturing video from the webcam
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("train model/Hamstring_Curls.mp4")
running = True

# Initialize the MediaPipe Pose object globally
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Get side argument from the command line
selected_side = sys.argv[1] if len(sys.argv) > 1 else "right"

# Show only the label for the selected side
if selected_side == "right":
    left_leg_label.pack_forget()  # Hide left leg label
elif selected_side == "left":
    right_leg_label.pack_forget()  # Hide right leg label

def show_frame():
    global right_leg_curl_count, left_leg_curl_count, right_leg_state, left_leg_state

    if not running:
        cap.release()
        cv2.destroyAllWindows()
        return

    if not paused:  # Only process frames if not paused
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

            # Get coordinates for right side (hip, knee, ankle)
            right_hip = [landmarks[24].x * frame.shape[1], landmarks[24].y * frame.shape[0]]
            right_knee = [landmarks[26].x * frame.shape[1], landmarks[26].y * frame.shape[0]]
            right_ankle = [landmarks[28].x * frame.shape[1], landmarks[28].y * frame.shape[0]]

            # Get coordinates for left side (hip, knee, ankle)
            left_hip = [landmarks[23].x * frame.shape[1], landmarks[23].y * frame.shape[0]]
            left_knee = [landmarks[25].x * frame.shape[1], landmarks[25].y * frame.shape[0]]
            left_ankle = [landmarks[27].x * frame.shape[1], landmarks[27].y * frame.shape[0]]

            # Calculate the hamstring curl angle for both legs (hip, knee, ankle)
            right_leg_angle = calculate_angle(right_hip, right_knee, right_ankle)
            left_leg_angle = calculate_angle(left_hip, left_knee, left_ankle)

            # Display the angle at the knee for the selected leg
            if selected_side == "right":
                cv2.putText(image, f"Angle: {int(right_leg_angle)}",
                            (int(right_knee[0]) + 10, int(right_knee[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)  # Red text for right leg angle at right knee

                # Draw right hip, knee, and ankle
                cv2.circle(image, tuple(np.int32(right_hip)), 5, (0, 255, 0), -1)  # Green circle for right hip
                cv2.circle(image, tuple(np.int32(right_knee)), 5, (0, 255, 0), -1)  # Green circle for right knee
                cv2.circle(image, tuple(np.int32(right_ankle)), 5, (0, 255, 0), -1)  # Green circle for right ankle

                # Draw lines between right hip-knee and knee-ankle
                cv2.line(image, tuple(np.int32(right_hip)), tuple(np.int32(right_knee)), (0, 255, 0), 2)
                cv2.line(image, tuple(np.int32(right_knee)), tuple(np.int32(right_ankle)), (0, 255, 0), 2)

            elif selected_side == "left":
                cv2.putText(image, f"Angle: {int(left_leg_angle)}",
                            (int(left_knee[0]) + 10, int(left_knee[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)  # Red text for left leg angle at left knee

                # Draw left hip, knee, and ankle
                cv2.circle(image, tuple(np.int32(left_hip)), 5, (0, 255, 0), -1)  # Green circle for left hip
                cv2.circle(image, tuple(np.int32(left_knee)), 5, (0, 255, 0), -1)  # Green circle for left knee
                cv2.circle(image, tuple(np.int32(left_ankle)), 5, (0, 255, 0), -1)  # Green circle for left ankle

                # Draw lines between left hip-knee and knee-ankle
                cv2.line(image, tuple(np.int32(left_hip)), tuple(np.int32(left_knee)), (0, 255, 0), 2)
                cv2.line(image, tuple(np.int32(left_knee)), tuple(np.int32(left_ankle)), (0, 255, 0), 2)

            # Right leg curl detection with angle check
            if target_angle - angle_tolerance <= right_leg_angle <= target_angle + angle_tolerance:  # Leg up (target angle reached)
                right_leg_state = 'up'
            if 160 <= right_leg_angle <= 180:  # Leg down (straightened out)
                if right_leg_state == 'up':
                    right_leg_curl_count += 1
                    right_leg_label.config(text=f"Right Leg Curls: {right_leg_curl_count}")
                right_leg_state = 'down'

            # Left leg curl detection with angle check
            if target_angle - angle_tolerance <= left_leg_angle <= target_angle + angle_tolerance:  # Leg up (target angle reached)
                left_leg_state = 'up'
            if 160 <= left_leg_angle <= 180:  # Leg down (straightened out)
                if left_leg_state == 'up':
                    left_leg_curl_count += 1
                    left_leg_label.config(text=f"Left Leg Curls: {left_leg_curl_count}")
                left_leg_state = 'down'

        # Convert the OpenCV image to a format that can be displayed in Tkinter
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image = image.resize((640, 480))  # Resize to fit Tkinter window
        photo = ImageTk.PhotoImage(image=image)

        # Update the video label with the new image
        video_label.config(image=photo)
        video_label.image = photo

    # Call this function again after 10ms to update the frame
    video_label.after(10, show_frame)

# Start displaying the video frames
show_frame()

# Start the Tkinter main loop
root.mainloop()
