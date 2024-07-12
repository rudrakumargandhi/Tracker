import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance


# Function to calculate the centroid of a cluster
def calculate_centroid(cluster):
    return np.mean(cluster, axis=0)


# Function to calculate the radius of a circle for a cluster
def calculate_radius(cluster, centroid):
    distances = np.linalg.norm(cluster - centroid, axis=1)
    return np.max(distances)


# Function to cluster points
def cluster_points(path, cluster_size):
    clusters = []
    for i in range(0, len(path), cluster_size):
        cluster = path[i:i + cluster_size]
        if len(cluster) > 0:
            clusters.append(cluster)
    return clusters


# Load the recorded path
def load_path(file_path):
    with open(file_path, "r") as f:
        return np.array(json.load(f))


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
out = cv2.VideoWriter('output_with_circles.mp4', fourcc, int(cap.get(cv2.CAP_PROP_FPS)), (frame_width, frame_height))

# Define the number of points per cluster
cluster_size = 500

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
    else:
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Calculate the clusters and update circles periodically
    if len(path) % cluster_size == 0:
        clusters = cluster_points(np.array(path), cluster_size)
        centroids = np.array([calculate_centroid(cluster) for cluster in clusters])
        radii = np.array([calculate_radius(cluster, centroid) for cluster, centroid in zip(clusters, centroids)])

    # Draw circles on the frame
    if len(path) >= cluster_size:
        for centroid, radius in zip(centroids, radii):
            cv2.circle(frame, (int(centroid[0]), int(centroid[1])), int(radius), (255, 0, 0), 2, cv2.LINE_AA)

    # Write the frame to the output video
    out.write(frame)

    # Display the frame
    cv2.imshow("Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save the path to a file
with open("pathsecond.json", "w") as f:
    json.dump(path, f)

cap.release()
out.release()
cv2.destroyAllWindows()
