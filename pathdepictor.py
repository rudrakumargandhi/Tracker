import cv2
import json
import numpy as np

# Initialize the video capture
video_path = "C:\\Users\\USER\\Downloads\\testmain.mp4"
cap = cv2.VideoCapture(video_path)

# Initialize the object tracker
tracker = cv2.TrackerKCF_create()

# Read the first frame
ret, frame = cap.read()
if not ret:
    print("Failed to read video")
    exit()

# Select the bounding box
bbox = cv2.selectROI("Tracking", frame, False)
cv2.destroyWindow("Tracking")

# Initialize the tracker with the first frame and bounding box
ok = tracker.init(frame, bbox)

path = []
frame_height, frame_width = frame.shape[:2]

# Initialize the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_with_path.mp4', fourcc, int(cap.get(cv2.CAP_PROP_FPS)), (frame_width, frame_height))

while True:
    # Read a new frame
    ret, frame = cap.read()
    if not ret:
        break

    # Update the tracker
    ok, bbox = tracker.update(frame)

    if ok:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        center = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        path.append(center)
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
    else:
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Create a blank frame for drawing the path
    path_frame = np.zeros((frame_height, frame_width, 3), np.uint8)

    # Draw the path as a continuous line on the blank frame
    if len(path) > 1:
        for i in range(1, len(path)):
            cv2.line(path_frame, path[i - 1], path[i], (0, 255, 0), 5)

    # Combine the original frame with the path frame
    combined_frame = cv2.addWeighted(frame, 0.7, path_frame, 0.3, 0)

    # Write the combined frame to the output video
    out.write(combined_frame)

    # Display the combined frame
    cv2.imshow("Tracking", combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save the path to a file
with open("path.json", "w") as f:
    json.dump(path, f)

cap.release()
out.release()
cv2.destroyAllWindows()
