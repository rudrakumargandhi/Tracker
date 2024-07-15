import json
import numpy as np
import cv2
from scipy.signal import find_peaks

# Read the coordinates from the JSON file
with open("pathmain.json", "r") as f:
    coordinates = json.load(f)

# Extract the x and y values from the coordinates
x_values = np.array([coord["x"] for coord in coordinates])
y_values = np.array([coord["y"] for coord in coordinates])

# Predict the amplitude of the wave function
amplitude = (np.max(y_values) - np.min(y_values)) / 2

# Find the peaks to calculate the period
peaks, _ = find_peaks(y_values)
if len(peaks) > 1:
    peak_distances = np.diff(peaks)
    average_peak_distance = np.mean(peak_distances)
    period = average_peak_distance
else:
    period = len(x_values) / (2 * np.pi)  # Fallback if peaks not found

# Create a window with OpenCV
cv2.namedWindow("Drawing Circle", cv2.WINDOW_NORMAL)

# Create a black image
img = np.zeros((512, 512, 3), dtype=np.uint8)

# Draw a circle with the amplitude as diameter and period as speed
color = (255, 255, 255)
center = (256, 256)
radius = int((amplitude*10)/2)
thickness = 2
speed = int(period * 25)  # Control speed of drawing based on period

angle_step = 1  # Angle step for drawing the ellipse
current_angle = 0

for _ in range(0, 360, angle_step):
    next_angle = current_angle + angle_step
    cv2.ellipse(img, center, (radius, radius), 0, current_angle, next_angle, color, thickness)
    cv2.imshow('Drawing Circle', img)
    cv2.waitKey(speed)  # Control speed of drawing
    current_angle = next_angle

# Display the final circle
cv2.imshow('Drawing Circle', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Release resources
cv2.destroyAllWindows()
