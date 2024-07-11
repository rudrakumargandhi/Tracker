import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Function to calculate the centroid of a cluster
def calculate_centroid(cluster):
    return np.mean(cluster, axis=0)


# Function to calculate the radius of a sphere for a cluster
def calculate_radius(cluster, centroid):
    distances = np.linalg.norm(cluster - centroid, axis=1)
    return np.max(distances)


# Load the recorded path
with open("path.json", "r") as f:
    path = json.load(f)

# Convert the path to a numpy array for easier manipulation
path = np.array(path)

# Define the number of points per cluster
cluster_size = 500

# Prepare the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Counters for clusters and spheres
num_clusters = 0
num_spheres = 0

# Loop through the path in clusters
for i in range(0, len(path), cluster_size):
    cluster = path[i:i + cluster_size]
    if len(cluster) > 1:
        num_clusters += 1
        centroid = calculate_centroid(cluster)
        radius = calculate_radius(cluster, centroid)

        # Generate the sphere's surface
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x = radius * np.cos(u) * np.sin(v) + centroid[0]
        y = radius * np.sin(u) * np.sin(v) + centroid[1]
        z = radius * np.cos(v)

        ax.plot_surface(x, y, z, color='b', alpha=0.3)
        num_spheres += 1

# Print the number of clusters and spheres
print(f'Number of clusters: {num_clusters}')
print(f'Number of Time Crystals: {num_spheres}')

# Setting the labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Path Representation with Spheres')

# Show the plot
plt.show()
