import os
import cv2

# Function to save an image from the camera
def save_image(frame, filename="captured_image.png"):
    cv2.imwrite(filename, frame)
    print(f"Image saved to {filename}")


# Open the camera connection
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    print("Press 'q' to exit the video window.")
    print("Press 'p' to save the current frame as an pattern image.")
    print("Press 'i' to save the current frame as an anomali image.")
    print("Press 'r' to change camera parameters.")
    while True:
        # Capture frames frame by frame
        ret, frame = cap.read()

        # If the frame is read correctly, ret is True
        if not ret:
            print("Error: Could not read frame.")
            break

        # Define the region of interest (ROI) as a centered square
        frame_height, frame_width = frame.shape[:2]
        roi_size = 55*7  # Size of the square
        roi_start_x = frame_width // 2 - roi_size // 2
        roi_start_y = frame_height // 2 - roi_size // 2
        roi_end_x = roi_start_x + roi_size
        roi_end_y = roi_start_y + roi_size

        # Extract the ROI from the frame
        roi = frame[roi_start_y:roi_end_y, roi_start_x:roi_end_x]
        roi_without_border = roi.copy()
        # Display the original frame with the ROI square
        frame[roi_start_y:roi_end_y, roi_start_x:roi_end_x] = roi
        cv2.rectangle(frame, (roi_start_x, roi_start_y), (roi_end_x, roi_end_y), (255, 0, 0), 2)    

        cv2.imshow('Original Frame', frame)


        # Exit the loop if 'q' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            pattern_image_count = len([name for name in os.listdir('Images') if name.startswith("pattern")])
            save_image(roi_without_border, f"Images/pattern_{pattern_image_count + 1}.png")
        elif key == ord('i'):
            pattern_image_count_2 = len([name for name in os.listdir('Images') if name.startswith("anomali")])
            print(f"Anomali image count: {pattern_image_count_2}")
            save_image(roi_without_border, f"Images/anomali_{pattern_image_count_2 + 1}.png")
        elif key == ord('r'): 
            # Now set to new values
            cap.set(cv2.CAP_PROP_BRIGHTNESS, 79)
            cap.set(cv2.CAP_PROP_CONTRAST, 2)
            cap.set(cv2.CAP_PROP_GAIN, -1)
            cap.set(cv2.CAP_PROP_EXPOSURE, -6)
        elif key == ord('t'): 
            # Now set to new values
            cap.set(cv2.CAP_PROP_BRIGHTNESS, 80)
            cap.set(cv2.CAP_PROP_CONTRAST, 2)
            cap.set(cv2.CAP_PROP_GAIN, -1)
            cap.set(cv2.CAP_PROP_EXPOSURE, -6)

    cap.release()
    cv2.destroyAllWindows()