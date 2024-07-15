import numpy as np
import matplotlib.pyplot as plt
import cv2


def plot_wave(A, x):
    # Generate x values
    x_vals = np.linspace(0, 2 * np.pi, 1000)
    # Generate y values using A sin(x)
    y_vals = A * np.sin(x_vals)

    # Plot the wave
    plt.plot(x_vals, y_vals)
    plt.title(f'Wave Function: {A} sin(x)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()


def draw_circle(A, x):
    # Ensure A is an integer for diameter
    diameter = int(A)
    radius = diameter // 2
    image_size = diameter + 10  # Adding padding for visibility

    # Ensure image_size is an integer
    image_size = int(image_size)

    img = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    center = (image_size // 2, image_size // 2)
    color = (255, 255, 255)  # White color
    thickness = 2  # Thickness of the circle
    speed = abs(x)  # Control speed of drawing based on x

    angle_step = 1 if x > 0 else -1  # Clockwise if x > 0, else anti-clockwise
    current_angle = 0 if x > 0 else 360

    for _ in range(0, 360):
        next_angle = current_angle + angle_step
        cv2.ellipse(img, center, (radius, radius), 0, current_angle, next_angle, color, thickness)
        cv2.imshow('Drawing Circle', img)
        cv2.waitKey(int(speed))  # Control speed of drawing
        current_angle = next_angle

    # Display the final circle
    cv2.imshow('Final Circle', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Input variables
A = float(input("Enter the amplitude (A): "))
x = float(input("Enter the parameter (x): "))

# Plot the wave function
plot_wave(A, x)

# Draw the circle with specified properties
draw_circle(A, x)


