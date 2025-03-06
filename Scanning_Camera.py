import ezdxf
import cv2
import numpy as np
import os

# Function to save coordinates to a DXF file 
def save_to_dxf(corners, filename="output.dxf"):
    doc = ezdxf.new(dxfversion='R2010')
    msp = doc.modelspace()
    
    for point in corners:
        x, y = point.ravel()  # Convert from array to simple coordinates
        mirrored_point = (float(x), float(-y))  # Convert to float and mirror y
        msp.add_point(mirrored_point, dxfattribs={'color': 1})
    
    doc.saveas(filename)
    print(f"Coordinates saved to {filename}")

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
    print("Press 's' to save coordinates to a DXF file.")
    print("Press 'i' to save the current frame as an image.")
    print("Press 'o' to save the current frame as an anomali image.")
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
        roi_size = 200  # Size of the square
        roi_start_x = frame_width // 2 - roi_size // 2
        roi_start_y = frame_height // 2 - roi_size // 2
        roi_end_x = roi_start_x + roi_size
        roi_end_y = roi_start_y + roi_size

        # Extract the ROI from the frame
        roi = frame[roi_start_y:roi_end_y, roi_start_x:roi_end_x]

        # Convert the ROI to grayscale
        gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray_frame, 9, 75, 75)
        
        # Apply adaptive thresholding with MEAN method (less blurry than Gaussian)
        thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY_INV, 15, 2)
        
        # Improve edge detection
        edges = cv2.Canny(filtered, 50, 150)
        
        # Morphological operations to clean up edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area to remove noise
        min_area = 100
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        # Create a copy of the ROI for visualization
        display_roi = roi.copy()
        roi_without_border = roi.copy()
        # Draw contours on the display image
        cv2.drawContours(display_roi, valid_contours, -1, (0, 255, 0), 1)
        
        # Store all detected corner points
        all_corners = []
        
        for contour in valid_contours:
            # Use a smaller epsilon for more accurate corner detection
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Alternative: Use Harris corner detector for more precise corner detection
            corners = cv2.goodFeaturesToTrack(gray_frame, 
                                             mask=cv2.drawContours(np.zeros_like(gray_frame), [contour], 0, 255, -1),
                                             maxCorners=25,
                                             qualityLevel=0.01,
                                             minDistance=10)
            
            if corners is not None:
                all_corners.extend(corners)
                for corner in corners:
                    x, y = corner.ravel()
                    cv2.circle(display_roi, (int(x), int(y)), 3, (0, 0, 255), -1)
            
            # Also draw the approximated polygon corners
            for point in approx:
                x, y = point[0]
                cv2.circle(display_roi, (x, y), 5, (255, 0, 0), -1)  # Blue points
                all_corners.append(np.array([[x, y]], dtype=np.float32))

        # Display the original frame with the ROI square
        frame[roi_start_y:roi_end_y, roi_start_x:roi_end_x] = display_roi
        cv2.rectangle(frame, (roi_start_x, roi_start_y), (roi_end_x, roi_end_y), (255, 0, 0), 2)
        
        # Display the edge image for debugging
        edge_display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        thresh_display = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        gray_display = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
        
        # Stack displays horizontally
        top_row = np.concat([display_roi, edge_display])
        bottom_row = np.concat([thresh_display, gray_display])
        combined = np.concat([top_row, bottom_row])
        
        # Resize for display
        combined = cv2.resize(combined, (0, 0), fx=1, fy=1)
        
        cv2.imshow('Original Frame', frame)
        cv2.imshow('Processing Steps', combined)

        # Exit the loop if 'q' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and all_corners:
            save_to_dxf(all_corners)
        elif key == ord('i'):
            pattern_image_count = len([name for name in os.listdir('Images') if name.startswith("pattern")])
            save_image(roi_without_border, f"Images/pattern_{pattern_image_count + 1}.png")
        elif key == ord('o'):
            pattern_image_count = len([name for name in os.listdir('Images') if name.startswith("anomali")])
            save_image(roi_without_border, f"Images/anomali_{pattern_image_count + 1}.png")
        elif key == ord('r'):
            # Print current camera parameters
            print("Current camera parameter values:")
            brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
            contrast = cap.get(cv2.CAP_PROP_CONTRAST)
            gain = cap.get(cv2.CAP_PROP_GAIN)
            exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
            
            print(f"Brightness: {brightness}")
            print(f"Contrast: {contrast}")
            print(f"Gain: {gain}")
            print(f"Exposure: {exposure}")
            
            # Now set to new values
            print("Setting camera to new values...")
            cap.set(cv2.CAP_PROP_BRIGHTNESS, 20)
            cap.set(cv2.CAP_PROP_CONTRAST, 5)
            cap.set(cv2.CAP_PROP_GAIN, -1)
            cap.set(cv2.CAP_PROP_EXPOSURE, -6)
    # Release the capture when everything is done
    cap.release()
    cv2.destroyAllWindows()