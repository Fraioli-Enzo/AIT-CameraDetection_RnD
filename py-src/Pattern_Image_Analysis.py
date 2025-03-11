import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from dataclasses import dataclass
from typing import List, Tuple, Optional

##############################################General###################################################### 
@dataclass
class ImageProcessingConfig:
    """Configuration parameters for image processing."""
    bilateral_filter_diameter: int = 6
    bilateral_sigma_color: float = 100
    bilateral_sigma_space: float = 100

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



class ImageComparator:
    """Handles comparison between two images to detect anomalies."""
    @staticmethod
    def compare_images(image1: np.ndarray, image2: np.ndarray, 
                    threshold: float = 30, 
                    blur_size: int = 3) -> Tuple[np.ndarray, float]:
        
        # If images is colorful convert images to grayscale
        if len(image1.shape) == 3:
            gray1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
        else:
            gray1 = image1
            
        if len(image2.shape) == 3:
            gray2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
        else:
            gray2 = image2
        
        # Ensure images are the same size
        if gray1.shape != gray2.shape:
            gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
        
        # Apply bilateral filter to both images BEFORE calculating differences
        if blur_size > 0:
            # For bilateral filter parameters
            d = blur_size       # Diameter of each pixel neighborhood
            sigma_color = 10    # Filter sigma in the color space
            sigma_space = 10    # Filter sigma in the coordinate space
            
            # Apply bilateral filter to both grayscale images
            gray1_filtered = cv2.bilateralFilter(gray1, d, sigma_color, sigma_space)
            gray2_filtered = cv2.bilateralFilter(gray2, d, sigma_color, sigma_space)
        else:
            # No blur applied
            gray1_filtered = gray1
            gray2_filtered = gray2
        
        # Create a 2x3 grid displaying the original and filtered images
        # Convert all images to BGR if they're grayscale for proper concatenation
        gray1_bgr = cv2.cvtColor(gray1, cv2.COLOR_GRAY2BGR)
        gray2_bgr = cv2.cvtColor(gray2, cv2.COLOR_GRAY2BGR)
        gray1_filtered_bgr = cv2.cvtColor(gray1_filtered, cv2.COLOR_GRAY2BGR)
        gray2_filtered_bgr = cv2.cvtColor(gray2_filtered, cv2.COLOR_GRAY2BGR)
        
        # Add labels to each image
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(gray1_bgr, "Gray Reference", (10, 30), font, 0.7, (0, 255, 0), 2)
        cv2.putText(gray2_bgr, "Gray Test", (10, 30), font, 0.7, (0, 255, 0), 2)
        cv2.putText(gray1_filtered_bgr, "Filtered Reference", (10, 30), font, 0.7, (0, 255, 0), 2)
        cv2.putText(gray2_filtered_bgr, "Filtered Test", (10, 30), font, 0.7, (0, 255, 0), 2)
        
        # Create rows and then combine into final display
        middle_row = np.hstack([gray1_bgr, gray2_bgr])
        bottom_row = np.hstack([gray1_filtered_bgr, gray2_filtered_bgr])
        
        # Stack the rows vertically
        comparison_grid = np.vstack([middle_row, bottom_row])
        
        # Display the grid
        cv2.imshow('Image Processing Steps', comparison_grid)

        print("\033[91m Press q to close windows\033[0m")
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()

        # Compute absolute difference between the FILTERED images
        diff = cv2.absdiff(gray1_filtered, gray2_filtered)
        
        # Apply threshold to highlight significant differences
        _, thresholded_diff = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        
        # No need for additional filtering here since we already filtered the input images
        filtered_diff = thresholded_diff
        
        # Calculate similarity score (lower means more similar)
        # Normalized by image size to get percentage
        non_zero = np.count_nonzero(filtered_diff)
        total_pixels = gray1.size
        similarity_score = (non_zero / total_pixels) * 100
        
        return filtered_diff, similarity_score
    
    @staticmethod
    def detect_anomalies(diff_mask: np.ndarray, 
                        min_area: int = 20) -> List[Tuple[Tuple[int, int, int, int], float]]:
        # Find contours in the difference mask
        contours, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        anomalies = []
        # For debugging: print contour details
        print(f"Found {len(contours)} potential anomaly contours")
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(contour)
                anomalies.append(((x, y, w, h), area))
        
        return anomalies
    
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
        thresh_display = cv2.cvtColor(diff_mask, cv2.COLOR_GRAY2BGR)

        # Add labels to images
        cv2.putText(image1, "Reference Image", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(image2, "Test Image", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(thresh_display, "Difference Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(highlighted, "Anomalies Highlighted", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Create comparison visualization
        top_row = np.hstack([image1, image2])
        bottom_row = np.hstack([thresh_display, highlighted])
        comparison = np.vstack([top_row, bottom_row])
        
        return comparison
##############################################Pipeline######################################################

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
        
        tolerance = 4
        blur_effect = 0
        # Compare preprocessed regions
        diff_mask, similarity_score = ImageComparator.compare_images(filtered_ref, filtered_test, tolerance, blur_effect)
        
        # Detect specific anomalies
        anomalies = ImageComparator.detect_anomalies(diff_mask, 11)
        
        # Create visualization
        comparison_viz = ImageComparator.highlight_anomalies(roi_ref, roi_test, diff_mask)
        
        # Display results
        cv2.imshow('Image Comparison', comparison_viz)
        
        # Print analysis
        print(f"Similarity: {100-similarity_score:.2f}% (Difference: {similarity_score:.2f}%)")
        print(f"Found {len(anomalies)} anomaly regions")
        
        for i, ((x, y, w, h), area) in enumerate(anomalies):
            print(f"Anomaly #{i+1}: Position (x={x}, y={y}), Size {w}x{h}, Area {area:.1f} px")
        
        print("\033[91m Press q to close windows / press r to restart program \033[0m")
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r'):
                cv2.destroyAllWindows()
                main()  # Restart the program by calling main() again
                return
            
        cv2.destroyAllWindows()
        return anomalies, similarity_score
    

##############################################Main######################################################
def main():
    """Main function to select and process images."""
    # Create a simple Tkinter root window
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Create pipeline with default configuration
    pipeline = ImagePipeline()

    # Image comparison mode (with all features)
    print("\033[93m Select reference image (good/normal image): \033[0m")
    reference_path = filedialog.askopenfilename(
        title="Select Pattern Image",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
            ("All files", "*.*")
        ]
    )
    
    if not reference_path:
        print("No reference image selected")
        return
        
    print("\033[93m Select Test/Anomali Image: \033[0m")
    test_path = filedialog.askopenfilename(
        title="Select Test/Anomali Image",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
            ("All files", "*.*")
        ]
    )
    
    if not test_path:
        print("No test image selected")
        return
        
    pipeline.compare_with_reference(reference_path, test_path)


if __name__ == "__main__":
    main()