import cv2
import numpy as np

# Function to detect the object frame
def detect_object(frame):
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range of blue color in HSV
    blue_lower = np.array([100, 50, 50])
    blue_upper = np.array([140, 255, 255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, blue_lower, blue_upper)

    # Morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    # Draw a rectangle around the object frame
    if max_contour is not None:
        x, y, w, h = cv2.boundingRect(max_contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame

# Function to track the object frame
def track_object(cap, frame_count):
    # Initialize the object frame
    object_frame = None

    # Track the object frame across frames
    for i in range(frame_count):
        ret, frame = cap.read()
        if ret:
            frame = detect_object(frame)
            if object_frame is None:
                object_frame = frame
            else:
                # Calculate the difference between the current frame and the object frame
                diff = cv2.absdiff(object_frame, frame)
                # Convert the difference image to grayscale
                diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                # Threshold the difference to get only the moving parts
                _, diff_gray = cv2.threshold(diff_gray, 10, 255, cv2.THRESH_BINARY)
                # Find contours in the thresholded difference
                contours, _ = cv2.findContours(diff_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # Draw a rectangle around the moving object
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 1000:
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.imshow('Tracking', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            break

    return frame

# Main function
def main():
    # Open the video file
    cap = cv2.VideoCapture("C:\\Users\\USER\\Downloads\\testmain.mp4")

    # Read the first frame to get the frame count
    ret, frame = cap.read()
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Track the object frame
    tracked_frame = track_object(cap, frame_count)

    # Display the tracked frame
    if tracked_frame is not None:
        cv2.imshow('Tracked Frame', tracked_frame)
        cv2.waitKey(0)
    else:
        print("Error: Tracked frame is empty.")

    # Release the video capture and close the window
    cap.release()
    cv2.destroyAllWindows()

# Run the main function
if __name__ == '__main__':
    main()