import cv2
import json

# Initialize the video capture
video_path = "C:\\Users\\USER\\pyproj\\test2.mp4"
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
        path.append(((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
    else:
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save the path to a file
with open("path.json", "w") as f:
    json.dump(path, f)

cap.release()
cv2.destroyAllWindows()
