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
        # Use the entire image as ROI
        roi = image.copy()

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

class ImagePipeline:
    """Main pipeline for image corner detection."""
    def __init__(self, config: Optional[ImageProcessingConfig] = None):
        self.config = config or ImageProcessingConfig()
    
    def compare_with_reference(self, reference_path: str, test_path: str):
        """Compare a test image with a reference image to find anomalies."""
        # Load images
        reference_image = ImageLoader.load_image(reference_path)
        test_image = ImageLoader.load_image(test_path)
        
        if reference_image is None or test_image is None:
            print("Error loading one or both images")
            return None
        
        # Preprocess both images the same way
        filtered_ref, roi_ref = ImagePreprocessor.preprocess_image(reference_image, self.config)
        filtered_test, roi_test = ImagePreprocessor.preprocess_image(test_image, self.config)
        
        # Compare preprocessed regions
        diff_mask, similarity_score = ImageComparator.compare_images(filtered_ref, filtered_test)
        
        # Detect specific anomalies
        anomalies = ImageComparator.detect_anomalies(diff_mask)
        
        # Create visualization
        comparison_viz = ImageComparator.highlight_anomalies(roi_ref, roi_test, diff_mask)
        
        # Display results
        cv2.imshow('Image Comparison', comparison_viz)
        
        # Print analysis
        print(f"Similarity: {100-similarity_score:.2f}% (Difference: {similarity_score:.2f}%)")
        print(f"Found {len(anomalies)} anomaly regions")
        
        for i, ((x, y, w, h), area) in enumerate(anomalies):
            print(f"Anomaly #{i+1}: Position (x={x}, y={y}), Size {w}x{h}, Area {area:.1f} px")
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
        
        cv2.destroyAllWindows()
        return anomalies, similarity_score

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

class ImageComparator:
    """Handles comparison between two images to detect anomalies."""
    
    @staticmethod
    def compare_images(image1: np.ndarray, image2: np.ndarray, 
                      threshold: float = 30, 
                      blur_size: int = 5) -> Tuple[np.ndarray, float]:
        # Convert images to grayscale
        if len(image1.shape) == 3:
            gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = image1
            
        if len(image2.shape) == 3:
            gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = image2
        
        # Ensure images are the same size
        if gray1.shape != gray2.shape:
            gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
        
        # Compute absolute difference between the images
        diff = cv2.absdiff(gray1, gray2)
        
        # Apply threshold to highlight significant differences
        _, thresholded_diff = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        
        # Apply Gaussian blur to reduce noise
        blurred_diff = cv2.GaussianBlur(thresholded_diff, (blur_size, blur_size), 0)
        
        # Calculate similarity score (lower means more similar)
        # Normalized by image size to get percentage
        non_zero = np.count_nonzero(blurred_diff)
        total_pixels = gray1.size
        similarity_score = (non_zero / total_pixels) * 100
        
        return blurred_diff, similarity_score
    
    @staticmethod
    def highlight_anomalies(image1: np.ndarray, image2: np.ndarray, 
                          diff_mask: np.ndarray) -> np.ndarray:
        # Ensure images are the same size and in color
        if image1.shape != image2.shape:
            image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
        
        if len(image1.shape) == 2:
            image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
        
        if len(image2.shape) == 2:
            image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
        
        # Create a red mask for anomalies
        red_mask = np.zeros_like(image1)
        if len(diff_mask.shape) == 2:  # If diff_mask is grayscale
            red_mask[diff_mask > 0] = [0, 0, 255]  # Set red where differences are detected
        
        # Blend original image with red highlights
        highlighted = cv2.addWeighted(image2, 0.7, red_mask, 0.3, 0)
        
        # Create comparison visualization
        top_row = np.hstack([image1, image2])
        bottom_row = np.hstack([cv2.cvtColor(diff_mask, cv2.COLOR_GRAY2BGR), highlighted])
        comparison = np.vstack([top_row, bottom_row])
        
        return comparison
    
    @staticmethod
    def detect_anomalies(diff_mask: np.ndarray, 
                        min_area: int = 50) -> List[Tuple[Tuple[int, int, int, int], float]]:
        # Find contours in the difference mask
        contours, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        anomalies = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(contour)
                anomalies.append(((x, y, w, h), area))
        
        return anomalies
    

def main():
    """Main function to select and process images."""
    # Create a simple Tkinter root window
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Create pipeline with default configuration
    pipeline = ImagePipeline()
    
    # Ask user what they want to do
    print("Select operation:")
    print("1. Detect corners in a single image")
    print("2. Compare two images to find anomalies")
    
    # choice = input("Enter your choice (1 or 2): ")
    choice = '2'
    if choice == '1':
        # Single image processing
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            corners = pipeline.process_image(file_path)
            if corners:
                print(f"Found {len(corners)} corners in the image")
        else:
            print("No file selected")
            
    elif choice == '2':
        # Image comparison mode
        print("\033[93m Select reference image (good/normal image): \033[0m")
        reference_path = filedialog.askopenfilename(
            title="Select Reference Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if not reference_path:
            print("No reference image selected")
            return
            
        print("\033[93m Select test image (image to check for anomalies): \033[0m")
        test_path = filedialog.askopenfilename(
            title="Select Test Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if not test_path:
            print("No test image selected")
            return
            
        pipeline.compare_with_reference(reference_path, test_path)
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()