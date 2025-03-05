import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class ImageProcessingConfig:
    """Configuration parameters for image processing."""
    bilateral_filter_diameter: int = 9
    bilateral_sigma_color: float = 75
    bilateral_sigma_space: float = 75
    adaptive_thresh_method: int = cv2.ADAPTIVE_THRESH_MEAN_C
    adaptive_thresh_type: int = cv2.THRESH_BINARY_INV
    adaptive_block_size: int = 15
    adaptive_const: float = 2
    canny_threshold1: float = 50
    canny_threshold2: float = 150
    min_contour_area: float = 100
    max_corners: int = 25
    corner_quality_level: float = 0.01
    corner_min_distance: float = 10

class ImageLoader:
    """Handles loading and basic image validation."""
    @staticmethod
    def load_image(image_path: str) -> Optional[np.ndarray]:
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not read image {image_path}")
        return frame

class ImagePreprocessor:
    """Applies preprocessing techniques to the image."""
    @staticmethod
    def preprocess_image(image: np.ndarray, config: ImageProcessingConfig) -> Tuple[np.ndarray, np.ndarray]:
        # Select ROI (can be customized)
        frame_height, frame_width = image.shape[:2]
        roi_size = min(frame_height, frame_width) - 20
        roi_start_x = frame_width // 2 - roi_size // 2
        roi_start_y = frame_height // 2 - roi_size // 2
        roi_end_x = roi_start_x + roi_size
        roi_end_y = roi_start_y + roi_size
        roi = image[roi_start_y:roi_end_y, roi_start_x:roi_end_x]

        # Convert to grayscale
        gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter
        filtered = cv2.bilateralFilter(
            gray_frame, 
            config.bilateral_filter_diameter, 
            config.bilateral_sigma_color, 
            config.bilateral_sigma_space
        )
        
        return filtered, roi

class EdgeDetector:
    """Handles edge and contour detection."""
    @staticmethod
    def detect_edges_and_contours(filtered_image: np.ndarray, config: ImageProcessingConfig) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            filtered_image, 255, 
            config.adaptive_thresh_method,
            config.adaptive_thresh_type, 
            config.adaptive_block_size, 
            config.adaptive_const
        )
        
        # Edge detection
        edges = cv2.Canny(filtered_image, config.canny_threshold1, config.canny_threshold2)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find and filter contours
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > config.min_contour_area]
        
        return thresh, edges, valid_contours

class CornerDetector:
    """Detects corners in the image."""
    @staticmethod
    def detect_corners(gray_frame: np.ndarray, contours: List[np.ndarray], config: ImageProcessingConfig) -> List[np.ndarray]:
        all_corners = []
        
        for contour in contours:
            # Polygon approximation
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Create a mask for the current contour
            mask = np.zeros_like(gray_frame)
            cv2.drawContours(mask, [contour], 0, 255, -1)
            
            # Detect good features (corners)
            corners = cv2.goodFeaturesToTrack(
                gray_frame, 
                mask=mask,
                maxCorners=config.max_corners,
                qualityLevel=config.corner_quality_level,
                minDistance=config.corner_min_distance
            )
            
            if corners is not None:
                all_corners.extend(corners)
                
                # Add polygon corners
                for point in approx:
                    x, y = point[0]
                    all_corners.append(np.array([[x, y]], dtype=np.float32))
        
        return all_corners

class Visualizer:
    """Handles visualization of processing steps."""
    @staticmethod
    def create_visualization(roi: np.ndarray, display_roi: np.ndarray, edges: np.ndarray, 
                              thresh: np.ndarray, filtered: np.ndarray) -> np.ndarray:
        # Convert single-channel images to 3-channel for concatenation
        edge_display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        thresh_display = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        gray_display = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
        
        # Ensure all images are the same size
        try:
            top_row = np.hstack([display_roi, edge_display])
            bottom_row = np.hstack([thresh_display, gray_display])
            combined_frames = np.vstack([top_row, bottom_row])
        except ValueError:
            # Resize if size mismatch
            h, w = display_roi.shape[:2]
            edge_display = cv2.resize(edge_display, (w, h))
            thresh_display = cv2.resize(thresh_display, (w, h))
            gray_display = cv2.resize(gray_display, (w, h))
            
            top_row = np.hstack([display_roi, edge_display])
            bottom_row = np.hstack([thresh_display, gray_display])
            combined_frames = np.vstack([top_row, bottom_row])
        
        # Resize if too large
        if combined_frames.shape[0] > 800 or combined_frames.shape[1] > 1200:
            scale = min(800/combined_frames.shape[0], 1200/combined_frames.shape[1])
            combined_frames = cv2.resize(combined_frames, (0, 0), fx=scale, fy=scale)
        
        return combined_frames

class ImageCornerDetectionPipeline:
    """Main pipeline for image corner detection."""
    def __init__(self, config: Optional[ImageProcessingConfig] = None):
        self.config = config or ImageProcessingConfig()
    
    def process_image(self, image_path: str):
        # Load the image
        frame = ImageLoader.load_image(image_path)
        if frame is None:
            return None
        
        # Preprocess the image
        filtered, roi = ImagePreprocessor.preprocess_image(frame, self.config)
        
        # Detect edges and contours
        thresh, edges, valid_contours = EdgeDetector.detect_edges_and_contours(filtered, self.config)
        
        # Create display image with contours
        display_roi = roi.copy()
        cv2.drawContours(display_roi, valid_contours, -1, (0, 255, 0), 1)
        
        # Detect corners
        all_corners = CornerDetector.detect_corners(
            cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), 
            valid_contours, 
            self.config
        )
        
        # Draw corners on display image
        for corner in all_corners:
            x, y = corner.ravel()
            cv2.circle(display_roi, (int(x), int(y)), 3, (0, 0, 255), -1)
        
        # Create visualization
        combined_frames = Visualizer.create_visualization(roi, display_roi, edges, thresh, filtered)
        
        # Display results
        # cv2.imshow('Original Image', frame)
        cv2.imshow('Processing Steps', combined_frames)
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
        
        cv2.destroyAllWindows()
        return all_corners

def main():
    """Main function to select and process an image."""
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
        # Create pipeline with default configuration
        pipeline = ImageCornerDetectionPipeline()
        
        corners = pipeline.process_image(file_path)
        if corners:
            print(f"Found {len(corners)} corners in the image")
    else:
        print("No file selected")

if __name__ == "__main__":
    main()