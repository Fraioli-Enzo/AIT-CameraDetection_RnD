import ezdxf
import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog

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

# Function to process an image
def process_image(image_path):
    # Read the image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    # Define the region of interest (ROI) as the entire image or a specific area
    frame_height, frame_width = frame.shape[:2]
    
    # You can process the entire image
    roi = frame.copy()
    
    # Or define a centered ROI if you want to keep that functionality
    roi_size = min(frame_height, frame_width) - 20  # Size of the square
    roi_start_x = frame_width // 2 - roi_size // 2
    roi_start_y = frame_height // 2 - roi_size // 2
    roi_end_x = roi_start_x + roi_size
    roi_end_y = roi_start_y + roi_size
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

    # Display the edge image for debugging
    edge_display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    thresh_display = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    gray_display = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
    
    # Stack displays horizontally and vertically
    try:
        top_row = np.concat([display_roi, edge_display])
        bottom_row = np.concat([thresh_display, gray_display])
        combined = np.concat([top_row, bottom_row])
    except ValueError:
        # Ensure all images are the same size
        h, w = display_roi.shape[:2]
        edge_display = cv2.resize(edge_display, (w, h))
        thresh_display = cv2.resize(thresh_display, (w, h))
        gray_display = cv2.resize(gray_display, (w, h))
        
        top_row = np.concat([display_roi, edge_display])
        bottom_row = np.concat([thresh_display, gray_display])
        combined = np.concat([top_row, bottom_row])
    
    # Resize if too large
    if combined.shape[0] > 800 or combined.shape[1] > 1200:
        scale = min(800/combined.shape[0], 1200/combined.shape[1])
        combined = cv2.resize(combined, (0, 0), fx=scale, fy=scale)
    
    # Display results
    cv2.imshow('Original Image', frame)
    cv2.imshow('Processing Steps', combined)
    
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

    
    cv2.destroyAllWindows()
    return all_corners

# Main function to select and process an image
def main():
    # Create a simple Tkinter root window
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    # Ask the user to select an image file
    file_path = filedialog.askopenfilename(
        title="Select Image File",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
            ("All files", "*.*")
        ]
    )
    
    if file_path:
        corners = process_image(file_path)
        if corners:
            print(f"Found {len(corners)} corners in the image")
    else:
        print("No file selected")

if __name__ == "__main__":
    main()