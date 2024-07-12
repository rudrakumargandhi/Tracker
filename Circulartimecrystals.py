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

# Function to cluster points and avoid completely overlapping circles
def cluster_points(path, cluster_size):
    clusters = []
    for i in range(0, len(path), cluster_size):
        cluster = path[i:i + cluster_size]
        if len(cluster) > 0:
            clusters.append(cluster)
    return clusters

# Function to adjust circles to avoid complete overlap
"""def adjust_circles(centroids, radii):
    adjusted_centroids = centroids.copy()
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            dist = distance.euclidean(centroids[i], centroids[j])
            if dist < radii[i] + radii[j]:
                overlap = radii[i] + radii[j] - dist
                direction = (centroids[j] - centroids[i]) / dist
                adjustment = direction * (overlap / 2 + 1)
                adjusted_centroids[j] += adjustment
                adjusted_centroids[i] -= adjustment
    return adjusted_centroids """

# Load the recorded path
with open("path.json", "r") as f:
    path = json.load(f)

# Convert the path to a numpy array for easier manipulation
path = np.array(path)

# Define the number of points per cluster
cluster_size = 500

# Calculate the clusters
clusters = cluster_points(path, cluster_size)

# Calculate the centroids and radii of the clusters
centroids = np.array([calculate_centroid(cluster) for cluster in clusters])
radii = np.array([calculate_radius(cluster, centroid) for cluster, centroid in zip(clusters, centroids)])

# Adjust the centroids to avoid completely overlapping circles
#adjusted_centroids = adjust_circles(centroids, radii)

# Plot the circles
fig, ax = plt.subplots()

for centroid, radius in zip(centroids, radii):
    circle = plt.Circle((centroid[0], centroid[1]), radius, color='b', alpha=0.3)
    ax.add_artist(circle)

# Ensure the plot scales correctly
ax.set_xlim(path[:, 0].min() - 500, path[:, 0].max() + 500)
ax.set_ylim(path[:, 1].min() - 500, path[:, 1].max() + 500)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('2D Path Representation with Circles')
ax.set_aspect('equal', 'box')

plt.show()
