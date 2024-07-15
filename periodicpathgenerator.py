import json
import math

# Define the function parameters
amplitude = 10
period = 10
num_points = 1000  # number of points to generate

# Generate the coordinates
x_values = [i * 0.1 for i in range(int(num_points / 0.1))]  # increment by 0.1
y_values = [amplitude * math.sin((math.pi/180)*x * period) for x in x_values]  # corrected the sine function

# Create a list of dictionaries to store the coordinates
coordinates = [{"x": x, "y": y} for x, y in zip(x_values, y_values)]

# Write the coordinates to a JSON file
with open("pathmain.json", "w") as f:
    json.dump(coordinates, f, indent=4)