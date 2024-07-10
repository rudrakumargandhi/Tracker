import cv2
import json
import numpy as np

# Load the recorded path
with open("path.json", "r") as f:
    path = json.load(f)

# Initialize the video capture to get video properties
video_path = "C:\\Users\\USER\\pyproj\\test2.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Release the capture since we only need properties
cap.release()

# Initialize the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('Tracer.mp4', fourcc, fps, (frame_width, frame_height))

cube_size = 50

for pos in path:
    # Create a blank frame (black background)
    frame = np.zeros((frame_height, frame_width, 3), np.uint8)

    x, y = pos

    # Draw a cube
    cv2.rectangle(frame, (x - cube_size // 2, y - cube_size // 2),
                  (x + cube_size // 2, y + cube_size // 2), (0, 255, 0), -1)

    # Write the frame to the output video
    out.write(frame)

out.release()
cv2.destroyAllWindows()
