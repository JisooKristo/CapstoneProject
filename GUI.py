# GUI.py

import tkinter as tk
from tkinter import messagebox
import os
import subprocess

# Function to execute the selected exercise script
def execute_script(script_name, side):
    try:
        # Pass the side as an argument to the script
        subprocess.run(['python', script_name, side], check=True)
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"Failed to run {script_name}: {str(e)}")

# Function to display the exercise options based on the selected body part
def show_exercises(body_part):
    exercise_var.set('')  # Reset the selected exercise
    exercise_menu['menu'].delete(0, 'end')  # Clear current exercise options

    # Add exercises based on body part selection
    if body_part == "Lower Body":
        for exercise in lower_body_exercises:
            exercise_menu['menu'].add_command(label=exercise, command=tk._setit(exercise_var, exercise))
    elif body_part == "Upper Body":
        for exercise in upper_body_exercises:
            exercise_menu['menu'].add_command(label=exercise, command=tk._setit(exercise_var, exercise))

    # Hide the side selection and start button initially
    side_menu.grid_forget()
    start_button.grid_forget()

# Function to show side selection (left or right) after choosing an exercise
def on_exercise_selected(*args):
    selected_exercise = exercise_var.get()
    if selected_exercise:
        side_menu.grid(row=4, column=0, pady=20)  # Show side selection
    else:
        side_menu.grid_forget()
    start_button.grid_forget()  # Hide the start button until side is selected

# Function to show the start button once the side is selected
def on_side_selected(*args):
    selected_side = side_var.get()
    if selected_side != "Select Side":
        start_button.grid(row=5, column=0, pady=40)  # Show the start button
    else:
        start_button.grid_forget()

# Function to handle exercise and side selection and execution
def on_start_exercise():
    selected_exercise = exercise_var.get()
    selected_side = side_var.get()

    if not selected_exercise:
        messagebox.showwarning("Warning", "Please select an exercise!")
        return

    if selected_side == "Select Side":
        messagebox.showwarning("Warning", "Please select Left or Right!")
        return

    # Mapping exercises to their corresponding script filenames
    script_mapping = {
        "Straight Leg Raise": "StraightLegRaise.py",
        "Hamstring Curls": "HamstringCurls.py",
        "Bicep Curls": "BicepCurls.py",
        "Lateral Raises": "LateralRaises.py"
    }

    script_file = script_mapping.get(selected_exercise)
    if script_file and os.path.exists(script_file):
        execute_script(script_file, selected_side.lower())
    else:
        messagebox.showerror("Error", f"Script for {selected_exercise} not found!")

# Define the list of exercises for each body part
lower_body_exercises = ["Straight Leg Raise", "Hamstring Curls"]
upper_body_exercises = ["Bicep Curls", "Lateral Raises"]

# Create the main window
root = tk.Tk()
root.title("Exercise Selector")

# Set the size of the window to 720x480
root.geometry("720x480")

# Configure padding for all widgets inside the main window
root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure([0, 1, 2, 3, 4, 5], weight=1)

# Body part selection
body_part_var = tk.StringVar(value="Select Body Part")
tk.Label(root, text="Select Body Part:", font=("Arial", 14)).grid(row=0, column=0, pady=20, padx=10)
body_part_menu = tk.OptionMenu(root, body_part_var, "Lower Body", "Upper Body", command=show_exercises)
body_part_menu.config(width=30, font=("Arial", 12))
body_part_menu.grid(row=1, column=0, pady=10)

# Exercise selection
exercise_var = tk.StringVar(value="Select Exercise")
exercise_var.trace('w', on_exercise_selected)  # Track when an exercise is selected
tk.Label(root, text="Select Exercise:", font=("Arial", 14)).grid(row=2, column=0, pady=20, padx=10)
exercise_menu = tk.OptionMenu(root, exercise_var, "Select Exercise")
exercise_menu.config(width=30, font=("Arial", 12))
exercise_menu.grid(row=3, column=0, pady=10)

# Side selection (left or right)
side_var = tk.StringVar(value="Select Side")
side_var.trace('w', on_side_selected)  # Track when the side is selected
side_label = tk.Label(root, text="Select Side (Left or Right):", font=("Arial", 14))
side_label.grid(row=4, column=0, pady=20, padx=10)
side_menu = tk.OptionMenu(root, side_var, "Left", "Right")
side_menu.config(width=30, font=("Arial", 12))

# Initially hide the side selection and start button
side_menu.grid_forget()

# Start exercise button
start_button = tk.Button(root, text="Start Exercise", font=("Arial", 14), width=20, command=on_start_exercise)

# Initially hide the start button
start_button.grid_forget()

# Run the main loop
root.mainloop()
