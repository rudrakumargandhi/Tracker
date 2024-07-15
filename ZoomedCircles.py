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


# Main function to generate the circles on a black background
def generate_circles(path_file, output_video, cluster_size=250):
    # Load the path
    path = load_path(path_file)

    # Define video properties
    frame_width = 1920
    frame_height = 1080
    fps = 30

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


        # Write the zoomed frame to the output video
        out.write(frame)

    out.release()
    print(f"Output video saved as {output_video}")


# Example usage
if __name__ == "__main__":
    generate_circles("pathsecond.json", "output_of_zoomed_circles.mp4")
