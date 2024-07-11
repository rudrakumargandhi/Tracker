import json
import numpy as np
# Function to calculate the clusters from the path
def cluster_points(path, cluster_size):
    clusters = []
    for i in range(0, len(path), cluster_size):
        cluster = path[i:i + cluster_size]
        if len(cluster) > 0:
            clusters.append(cluster.tolist())
    return clusters

# Load the recorded path
with open("path.json", "r") as f:
    path = json.load(f)

# Convert the path to a numpy array for easier manipulation
path = np.array(path)

# Define the number of points per cluster
cluster_size = 500

# Calculate the clusters
clusters = cluster_points(path, cluster_size)

# Save the clusters to a file
with open("clusters.json", "w") as f:
    json.dump(clusters, f)

print(f'Number of clusters: {len(clusters)}')
for idx, cluster in enumerate(clusters):
    print(f'Cluster {idx + 1}: {cluster}')
