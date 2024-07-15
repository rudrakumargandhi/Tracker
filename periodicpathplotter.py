import json
import matplotlib.pyplot as plt

# Read the coordinates from the JSON file
with open("pathmain.json", "r") as f:
    coordinates = json.load(f)

# Extract the x and y values from the coordinates
x_values = [coord["x"] for coord in coordinates]
y_values = [coord["y"] for coord in coordinates]

# Create the plot
plt.plot(x_values, y_values)

# Set the title and labels
plt.title("Periodic Function Wave")
plt.xlabel("X")
plt.ylabel("Y")

# Show the plot
plt.show()