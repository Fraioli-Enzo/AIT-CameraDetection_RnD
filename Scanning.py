import ezdxf
import cv2

# Function to save coordinates to a DXF file 
def save_to_dxf(contours, filename="output.dxf"):
    doc = ezdxf.new(dxfversion='R2010')
    msp = doc.modelspace()
    
    for contour in contours:

        # # Add lines of the contours with mirrored coordinates
        # for i in range(len(contour)):
        #     start_point = contour[i][0]
        #     end_point = contour[(i + 1) % len(contour)][0]
        #     mirrored_start_point = (start_point[0], -start_point[1])
        #     mirrored_end_point = (end_point[0], -end_point[1])
        #     msp.add_line(start=mirrored_start_point, end=mirrored_end_point)

        # Add points of interest
        for point in contour:
            x, y = point[0]
            mirrored_point = (x, -y)
            msp.add_point(location=mirrored_point, dxfattribs={'color': 1})
    
    doc.saveas(filename)
    print(f"Coordinates saved to {filename}")

# Open the camera connection
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    print("Press 'q' to exit the video window.")
    print("Press 's' to save coordinates to a DXF file.")

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

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 25, 16)

        # Edge detection with Canny
        edges = cv2.Canny(gray_frame, 100, 200)

        # Morphological transformation to detect blobs
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Find contours of the detected blobs
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours and points of interest directly on the original image
        for contour in contours:
            # Calculate the polygonal approximation of the contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Draw the points of interest at the vertices of the polygon
            for point in approx:
                x, y = point[0]
                cv2.circle(frame[roi_start_y:roi_end_y, roi_start_x:roi_end_x], (x, y), 4, (0, 0, 255), -1)  # Red points

        # Display the original frame with the ROI square and detected points
        cv2.rectangle(frame, (roi_start_x, roi_start_y), (roi_end_x, roi_end_y), (255, 0, 0), 2)
        cv2.imshow('Original Frame', frame)

        # Exit the loop if 'q' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_to_dxf(contours)

    # Release the capture when everything is done
    cap.release()
    cv2.destroyAllWindows()