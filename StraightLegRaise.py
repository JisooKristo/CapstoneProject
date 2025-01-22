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

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # Shoulder
    b = np.array(b)  # Hip
    c = np.array(c)  # Ankle

    # Calculate the vectors
    vector_1 = a - b  # Shoulder to hip
    vector_2 = c - b  # Ankle to hip

    # Compute the angle between the two vectors using arctan2
    angle = np.arctan2(vector_2[1], vector_2[0]) - np.arctan2(vector_1[1], vector_1[0])

    # Convert angle to degrees
    angle = angle * 180.0 / np.pi

    # Normalize the angle to be between 0 and 360 degrees
    if angle < 0:
        angle += 360

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

# Initialize variables for counting straight leg raises
right_leg_raise_count = 0
left_leg_raise_count = 0

# Initialize leg up and down state variables
right_leg_up = False
left_leg_up = False
right_leg_down = False
left_leg_down = False

# Timer variables
timer = 8  # Changeable timer for each rep
timer_started = False  # Whether the timer has started

paused = False  # Variable to track whether the video is paused

# Define target angles and tolerance
target_angle_up = 190
target_angle_down = 170
angle_tolerance = 5

# Initialize Tkinter window
root = tk.Tk()
root.title("Straight Leg Raise Detection")
root.geometry("720x960")

# Create a frame for the video feed
video_frame = tk.Frame(root)
video_frame.pack()

# Create label to display the video feed
video_label = Label(video_frame)
video_label.pack()

# Create labels for the right and left leg raise counts
right_leg_label = Label(root, text="Right Leg Raises: 0", font=("Helvetica", 12))
right_leg_label.pack(pady=5)

left_leg_label = Label(root, text="Left Leg Raises: 0", font=("Helvetica", 12))
left_leg_label.pack(pady=5)

# Create a label to display the countdown timer
timer_label = Label(root, text=f"Timer: {timer}", font=("Helvetica", 14))
timer_label.pack(pady=5)

# Create "Done" button at the bottom of the window
done_button = tk.Button(root, text="Done", command=done_callback)
done_button.pack(side="bottom", pady=10)

# Bind spacebar to toggle pause/resume
root.bind("<space>", toggle_pause)

# Start capturing video from the webcam
cap = cv2.VideoCapture("train model/Straight_Leg_Raises.mp4")
# cap = cv2.VideoCapture(0)

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

# Add the feedback label for the text
feedback_label = Label(root, text="", font=("Helvetica", 14), fg="red")
feedback_label.pack(pady=5)

def show_frame():
    global right_leg_raise_count, left_leg_raise_count
    global right_leg_up, left_leg_up, right_leg_down, left_leg_down
    global timer_started
    global paused

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

            # Get coordinates for right side (shoulder, hip, ankle)
            right_shoulder = [landmarks[12].x * frame.shape[1], landmarks[12].y * frame.shape[0]]
            right_hip = [landmarks[24].x * frame.shape[1], landmarks[24].y * frame.shape[0]]
            right_ankle = [landmarks[28].x * frame.shape[1], landmarks[28].y * frame.shape[0]]

            # Get coordinates for left side (shoulder, hip, ankle)
            left_shoulder = [landmarks[11].x * frame.shape[1], landmarks[11].y * frame.shape[0]]
            left_hip = [landmarks[23].x * frame.shape[1], landmarks[23].y * frame.shape[0]]
            left_ankle = [landmarks[27].x * frame.shape[1], landmarks[27].y * frame.shape[0]]

            # Calculate the leg raise angle for both legs (between shoulder, hip, and ankle)
            right_leg_angle = calculate_angle(right_shoulder, right_hip, right_ankle)
            left_leg_angle = calculate_angle(left_shoulder, left_hip, left_ankle)

            # Display the angle at the hip for the selected leg
            if selected_side == "right":
                cv2.putText(image, f"Angle: {int(right_leg_angle)}",
                            (int(right_hip[0]) + 10, int(right_hip[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)  # Red text for right leg angle at right hip

                # Draw the right leg landmarks (shoulder, hip, and ankle) for the right side only
                cv2.circle(image, tuple(np.int32(right_shoulder)), 5, (0, 255, 0), -1)  # Green circle for right shoulder
                cv2.circle(image, tuple(np.int32(right_hip)), 5, (0, 255, 0), -1)  # Green circle for right hip
                cv2.circle(image, tuple(np.int32(right_ankle)), 5, (0, 255, 0), -1)  # Green circle for right ankle

                # Draw lines for the right side landmarks
                cv2.line(image, tuple(np.int32(right_shoulder)), tuple(np.int32(right_hip)), (0, 255, 0), 2)
                cv2.line(image, tuple(np.int32(right_hip)), tuple(np.int32(right_ankle)), (0, 255, 0), 2)

                # Check if right leg is up and down for counting
                if target_angle_up - angle_tolerance <= right_leg_angle <= target_angle_up + angle_tolerance:
                    if not right_leg_up:
                        right_leg_up = True
                        if not timer_started:
                            timer_started = True
                            update_timer()  # Start non-blocking timer countdown

                if target_angle_down - angle_tolerance <= right_leg_angle <= target_angle_down + angle_tolerance:
                    if right_leg_up:  # Only count down if the leg was previously up
                        right_leg_down = True

                if right_leg_down:
                    if timer == 0:
                        right_leg_raise_count += 1
                        right_leg_label.config(text=f"Right Leg Raises: {right_leg_raise_count}")
                        reset_leg_states()  # Reset the leg states for next rep
                    feedback_label.config(text="")  # Clear the feedback when rep is counted or leg is down

                # Show feedback when timer is 0 and leg is not down yet
                if timer == 0 and not right_leg_down:
                    feedback_label.config(text="Bring Down the Leg")

            elif selected_side == "left":
                cv2.putText(image, f"Angle: {int(left_leg_angle)}",
                            (int(left_hip[0]) + 10, int(left_hip[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)  # Red text for left leg angle at left hip

                # Draw the left leg landmarks (shoulder, hip, and ankle) for the left side only
                cv2.circle(image, tuple(np.int32(left_shoulder)), 5, (0, 255, 0), -1)  # Green circle for left shoulder
                cv2.circle(image, tuple(np.int32(left_hip)), 5, (0, 255, 0), -1)  # Green circle for left hip
                cv2.circle(image, tuple(np.int32(left_ankle)), 5, (0, 255, 0), -1)  # Green circle for left ankle

                # Draw lines for the left side landmarks
                cv2.line(image, tuple(np.int32(left_shoulder)), tuple(np.int32(left_hip)), (0, 255, 0), 2)
                cv2.line(image, tuple(np.int32(left_hip)), tuple(np.int32(left_ankle)), (0, 255, 0), 2)

                # Check if left leg is up and down for counting
                if target_angle_up - angle_tolerance <= left_leg_angle <= target_angle_up + angle_tolerance:
                    if not left_leg_up:
                        left_leg_up = True
                        if not timer_started:
                            timer_started = True
                            update_timer()  # Start non-blocking timer countdown

                if target_angle_down - angle_tolerance <= left_leg_angle <= target_angle_down + angle_tolerance:
                    if left_leg_up:  # Only count down if the leg was previously up
                        left_leg_down = True

                # For the left leg:
                if left_leg_down:
                    if timer == 0:
                        left_leg_raise_count += 1
                        left_leg_label.config(text=f"Left Leg Raises: {left_leg_raise_count}")
                        reset_leg_states()  # Reset the leg states for next rep
                    feedback_label.config(text="")  # Clear the feedback when rep is counted or leg is down

                # Show feedback when timer is 0 and leg is not down yet
                if timer == 0 and not left_leg_down:
                    feedback_label.config(text="Bring Down the Leg")

        # Convert the frame to an image format that Tkinter can use
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)

        # Update the label with the image
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

    # Continue to call show_frame
    video_label.after(10, show_frame)






def reset_leg_states():
    global right_leg_up, left_leg_up, right_leg_down, left_leg_down, timer_started
    right_leg_up = False
    left_leg_up = False
    right_leg_down = False
    left_leg_down = False
    timer_started = False

def update_timer():
    global timer, timer_started
    if timer_started and timer > 0:
        timer -= 1
        timer_label.config(text=f"Timer: {timer}")  # Update timer label
        root.after(1000, update_timer)  # Call update_timer() every 1 second

# Start showing frames
show_frame()

# Start Tkinter main loop
root.mainloop()
