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

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point (shoulder)
    b = np.array(b)  # Mid point (elbow)
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

# Initialize variables for counting curls
right_curl_count = 0
left_curl_count = 0
right_arm_state = None  # 'up' or 'down'
left_arm_state = None

# Define threshold angles for curl detection
target_up_angle = 30   # Arm bent, e.g., when the user is at the top of the curl
target_down_angle = 160  # Arm extended, e.g., when the user is at the bottom of the curl
angle_tolerance = 5  # Tolerance to account for small variations in user movement

# Initialize Tkinter window
root = tk.Tk()
root.title("Pose Detection with Camera Feed")
root.geometry("720x960")

# Create a frame for the video feed
video_frame = tk.Frame(root)
video_frame.pack()

# Create label to display the video feed
video_label = Label(video_frame)
video_label.pack()

# Create labels for the right and left curl counts
right_curl_label = Label(root, text="Right Arm Curls: 0", font=("Helvetica", 12))
left_curl_label = Label(root, text="Left Arm Curls: 0", font=("Helvetica", 12))

# Start with the "Left" curl counter visible
right_curl_label.pack_forget()  # Hide initially
left_curl_label.pack(pady=5)

# Create "Done" button at the bottom of the window
done_button = tk.Button(root, text="Done", command=done_callback)
done_button.pack(side="bottom", pady=10)

# Start capturing video from the webcam
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("train model/Bicep_Curl.mp4")
running = True

# Initialize the MediaPipe Pose object globally
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Determine if left or right side was selected
selected_side = sys.argv[1] if len(sys.argv) > 1 else "left"  # Default to left if not provided

# Show relevant curl label based on selected side
def update_curl_labels():
    if selected_side == "right":
        left_curl_label.pack_forget()  # Hide left curl count
        right_curl_label.pack(pady=5)  # Show right curl count
    else:
        right_curl_label.pack_forget()  # Hide right curl count
        left_curl_label.pack(pady=5)  # Show left curl count

# Initialize the selected side
update_curl_labels()

def show_frame():
    global right_curl_count, left_curl_count, right_arm_state, left_arm_state

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

        # Get coordinates for right arm (shoulder, elbow, wrist)
        right_shoulder = [landmarks[12].x * frame.shape[1], landmarks[12].y * frame.shape[0]]
        right_elbow = [landmarks[14].x * frame.shape[1], landmarks[14].y * frame.shape[0]]
        right_wrist = [landmarks[16].x * frame.shape[1], landmarks[16].y * frame.shape[0]]

        # Get coordinates for left arm (shoulder, elbow, wrist)
        left_shoulder = [landmarks[11].x * frame.shape[1], landmarks[11].y * frame.shape[0]]
        left_elbow = [landmarks[13].x * frame.shape[1], landmarks[13].y * frame.shape[0]]
        left_wrist = [landmarks[15].x * frame.shape[1], landmarks[15].y * frame.shape[0]]

        # Calculate angles for both elbows
        right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

        # Display the elbow angles on the video frame
        if selected_side == "right":
            cv2.putText(image, str(int(right_elbow_angle)),
                        (int(right_elbow[0]), int(right_elbow[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)  # Black outline
            cv2.putText(image, str(int(right_elbow_angle)),
                        (int(right_elbow[0]), int(right_elbow[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # Green text
        elif selected_side == "left":
            cv2.putText(image, str(int(left_elbow_angle)),
                        (int(left_elbow[0]), int(left_elbow[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)  # Black outline
            cv2.putText(image, str(int(left_elbow_angle)),
                        (int(left_elbow[0]), int(left_elbow[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # Green text

        # Detect curls for the right arm with tolerance
        if selected_side == "right":
            if right_elbow_angle > target_down_angle - angle_tolerance:  # Arm extended (with tolerance)
                if right_arm_state == 'up':
                    right_curl_count += 1
                    right_curl_label.config(text=f"Right Arm Curls: {right_curl_count}")  # Update the right curl count label
                right_arm_state = 'down'
            if right_elbow_angle < target_up_angle + angle_tolerance:  # Arm bent (with tolerance)
                right_arm_state = 'up'

        # Detect curls for the left arm with tolerance
        if selected_side == "left":
            if left_elbow_angle > target_down_angle - angle_tolerance:  # Arm extended (with tolerance)
                if left_arm_state == 'up':
                    left_curl_count += 1
                    left_curl_label.config(text=f"Left Arm Curls: {left_curl_count}")  # Update the left curl count label
                left_arm_state = 'down'
            if left_elbow_angle < target_up_angle + angle_tolerance:  # Arm bent (with tolerance)
                left_arm_state = 'up'

        # Draw landmarks on the image for the selected side
        if selected_side == "right":
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        elif selected_side == "left":
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
