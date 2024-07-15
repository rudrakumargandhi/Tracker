import json
import numpy as np
import cv2
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


# Function to determine the maximum bounding box
def get_max_bounding_box(path, cluster_size, padding=50):
    clusters = cluster_points(path, cluster_size)
    centroids = np.array([calculate_centroid(cluster) for cluster in clusters])
    radii = np.array([calculate_radius(cluster, centroid) for cluster, centroid in zip(clusters, centroids)])

    min_x = int(np.min(centroids[:, 0] - radii)) - padding
    max_x = int(np.max(centroids[:, 0] + radii)) + padding
    min_y = int(np.min(centroids[:, 1] - radii)) - padding
    max_y = int(np.max(centroids[:, 1] + radii)) + padding

    return min_x, max_x, min_y, max_y


# Main function to generate the circles on a black background
def generate_circles(path_file, output_video, cluster_size=250):
    # Load the path
    path = load_path(path_file)

    # Define video properties
    frame_width = 1920
    frame_height = 1080
    fps = 30

    # Determine the maximum bounding box
    min_x, max_x, min_y, max_y = get_max_bounding_box(path, cluster_size)

    # Ensure the bounding box is within the frame limits
    min_x = max(min_x, 0)
    min_y = max(min_y, 0)
    max_x = min(max_x, frame_width)
    max_y = min(max_y, frame_height)

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    # Initialize empty frame
    frame = np.zeros((frame_height, frame_width, 3), np.uint8)

    clusters = []
    centroids = []
    radii = []

    for i in range(len(path)):
        # Add point to the path
        current_path = path[:i + 1]

        # Calculate clusters and update circles periodically
        if len(current_path) % cluster_size == 0 or i == len(path) - 1:
            clusters = cluster_points(current_path, cluster_size)
            centroids = np.array([calculate_centroid(cluster) for cluster in clusters])
            radii = np.array([calculate_radius(cluster, centroid) for cluster, centroid in zip(clusters, centroids)])

        # Create a black frame
        frame = np.zeros((frame_height, frame_width, 3), np.uint8)

        # Draw circles
        for centroid, radius in zip(centroids, radii):
            cv2.circle(frame, (int(centroid[0]), int(centroid[1])), int(radius), (255, 0, 0), 2, cv2.LINE_AA)

        # Draw the path
        for j in range(1, len(current_path)):
            cv2.line(frame, tuple(current_path[j - 1]), tuple(current_path[j]), (0, 255, 0), 2, cv2.LINE_AA)

        # Extract the region of interest and resize to fit the output frame
        roi = frame[min_y:max_y, min_x:max_x]
        zoomed_frame = cv2.resize(roi, (frame_width, frame_height))

        # Write the zoomed frame to the output video
        out.write(zoomed_frame)

    out.release()
    print(f"Output video saved as {output_video}")


# Example usage
if __name__ == "__main__":
    generate_circles("pathsecond.json", "zoomed.mp4")
