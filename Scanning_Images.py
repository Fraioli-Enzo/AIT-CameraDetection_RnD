import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class ImageProcessingConfig:
    """Configuration parameters for image processing."""
    bilateral_filter_diameter: int = 5
    bilateral_sigma_color: float = 50
    bilateral_sigma_space: float = 50
    adaptive_thresh_method: int = cv2.ADAPTIVE_THRESH_MEAN_C
    adaptive_thresh_type: int = cv2.THRESH_BINARY_INV
    adaptive_block_size: int = 15
    adaptive_const: float = 2
    canny_threshold1: float = 25
    canny_threshold2: float = 250
    min_contour_area: float = 20
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
    
    @staticmethod
    def divide_image_into_four(image: np.ndarray) -> List[np.ndarray]:
        if image is None:
            print("Error: No image provided")
            return []
        
        height, width = image.shape[:2]
        
        # Calculer les points médians
        mid_h = height // 2
        mid_w = width // 2
        
        # Extraire les quatre quadrants
        top_left = image[:mid_h, :mid_w].copy()
        top_right = image[:mid_h, mid_w:].copy()
        bottom_left = image[mid_h:, :mid_w].copy()
        bottom_right = image[mid_h:, mid_w:].copy()
        
        # Créer une visualisation des quadrants pour le débogage
        # (facultatif, vous pouvez supprimer cette partie)
        vis_top = np.hstack([top_left, top_right])
        vis_bottom = np.hstack([bottom_left, bottom_right])
        visualization = np.vstack([vis_top, vis_bottom])
        
        # Ajouter des étiquettes pour l'identification
        font = cv2.FONT_HERSHEY_SIMPLEX
        if len(image.shape) == 3:  # Image en couleur
            cv2.putText(visualization, "TL", (10, 30), font, 0.7, (0, 255, 0), 2)
            cv2.putText(visualization, "TR", (mid_w + 10, 30), font, 0.7, (0, 255, 0), 2)
            cv2.putText(visualization, "BL", (10, mid_h + 30), font, 0.7, (0, 255, 0), 2)
            cv2.putText(visualization, "BR", (mid_w + 10, mid_h + 30), font, 0.7, (0, 255, 0), 2)
        
        # Afficher la visualisation
        cv2.imshow('Image divided into four quadrants', visualization)
        print("\033[91m Press q to close windows \033[0m")
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
            
        cv2.destroyAllWindows()
        
        return [top_left, top_right, bottom_left, bottom_right]
    
    def _extract_main_object(self, image: np.ndarray) -> np.ndarray:
        """Extract the main object from the image, returning a binary mask with eroded edges to avoid edge noise."""
        # Ensure image is grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply adaptive thresholding to separate foreground from background
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            self.config.adaptive_thresh_method,
            self.config.adaptive_thresh_type,
            self.config.adaptive_block_size,
            self.config.adaptive_const
        )
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # If no contours found, return the original threshold
        if not contours:
            return thresh
        
        # Create an empty mask
        mask = np.zeros_like(gray)
        
        # Find the largest contour (assume it's the main object)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Draw the largest contour
        cv2.drawContours(mask, [largest_contour], 0, 255, -1)
        
        # Optional: Fill holes in the mask
        mask_floodfill = mask.copy()
        h, w = mask.shape[:2]
        mask_zero = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(mask_floodfill, mask_zero, (0, 0), 255)
        mask_floodfill_inv = cv2.bitwise_not(mask_floodfill)
        mask = mask | mask_floodfill_inv
        
        # Create an eroded version to avoid edge noise
        erosion_kernel = np.ones((3, 3), np.uint8)
        eroded_mask = cv2.erode(mask, erosion_kernel, iterations=2)
        
        # Create a dilated version for visualization
        dilated_mask = cv2.dilate(mask, erosion_kernel, iterations=1)
        
        # Show the original and eroded masks for debugging
        cv2.imshow('Original Mask', mask)
        cv2.imshow('Eroded Mask (Used for Comparison)', eroded_mask)
        cv2.imshow('Dilated Mask (For Visualization)', dilated_mask)
        
        return eroded_mask

class BasicImageProcess:
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

    """Handles visualization of processing steps."""
    @staticmethod
    def create_visualization(display_roi: np.ndarray, edges: np.ndarray, 
                            thresh: np.ndarray, filtered: np.ndarray) -> np.ndarray:
    
        # Convert single-channel images to 3-channel for concatenation
        edge_display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        thresh_display = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        gray_display = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
        
        # Create colormap versions
        jet_display = cv2.applyColorMap(filtered, cv2.COLORMAP_JET)
        plasma_display = cv2.applyColorMap(filtered, cv2.COLORMAP_PLASMA)
        inferno_display = cv2.applyColorMap(filtered, cv2.COLORMAP_INFERNO)
        magma_display = cv2.applyColorMap(filtered, cv2.COLORMAP_MAGMA)
        hot_display = cv2.applyColorMap(filtered, cv2.COLORMAP_HOT)

        # Add labels to each image
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(edge_display, "EDGE", (10, 30), font, 0.8, (0, 0, 255), 1)
        cv2.putText(thresh_display, "THRESH", (10, 30), font, 0.8, (0, 0, 255), 2)
        cv2.putText(gray_display, "GRAY", (10, 30), font, 0.8, (255, 255, 255), 1)
        cv2.putText(jet_display, "JET", (10, 30), font, 0.8, (0, 0, 0), 1)
        cv2.putText(plasma_display, "PLASMA", (10, 30), font, 0.8, (255, 255, 255), 1)
        cv2.putText(inferno_display, "INFERNO", (10, 30), font, 0.8, (255, 255, 255), 1)
        cv2.putText(magma_display, "MAGMA", (10, 30), font, 0.8, (255, 255, 255), 1)
        cv2.putText(hot_display, "HOT", (10, 30), font, 0.8, (255, 255, 255), 1)

        # Ensure all images are the same size
        try:
            # Make sure all images have the same shape before stacking
            h, w = display_roi.shape[:2]
            edge_display = cv2.resize(edge_display, (w, h))
            thresh_display = cv2.resize(thresh_display, (w, h))
            gray_display = cv2.resize(gray_display, (w, h))
            jet_display = cv2.resize(jet_display, (w, h))
            plasma_display = cv2.resize(plasma_display, (w, h))
            inferno_display = cv2.resize(inferno_display, (w, h))
            magma_display = cv2.resize(magma_display, (w, h))
            hot_display = cv2.resize(hot_display, (w, h))
            
            # Stack horizontally and vertically using np.hstack and np.vstack
            top_row = np.hstack([display_roi, edge_display, jet_display])
            middle_row = np.hstack([thresh_display, gray_display, plasma_display])
            bottom_row = np.hstack([inferno_display, magma_display, hot_display])
            combined_frames = np.vstack([top_row, middle_row, bottom_row])
            
        except ValueError as e:
            print(f"Error stacking images: {e}")
            # Fallback to simpler visualization
            top_row = np.hstack([display_roi, edge_display])
            bottom_row = np.hstack([thresh_display, gray_display])
            combined_frames = np.vstack([top_row, bottom_row])
        
        # Resize if too large
        if combined_frames.shape[0] > 800 or combined_frames.shape[1] > 1200:
            scale = min(800/combined_frames.shape[0], 1200/combined_frames.shape[1])
            combined_frames = cv2.resize(combined_frames, (0, 0), fx=scale, fy=scale)
        
        return combined_frames

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

class ImagePipeline:
    """Main pipeline for image corner detection."""
    def __init__(self, config: Optional[ImageProcessingConfig] = None):
        self.config = config or ImageProcessingConfig()
    
    # Key 1 
    def basic_simple_image_process(self, image_path: str):
        # Load the image
        frame = ImageLoader.load_image(image_path)
        if frame is None:
            return None
        
        # Preprocess the image
        filtered, roi = ImagePreprocessor.preprocess_image(frame, self.config)
        
        # Detect edges and contours
        thresh, edges, valid_contours = BasicImageProcess.detect_edges_and_contours(filtered, self.config)
        
        # Create display image with contours
        display_roi = roi.copy()
        cv2.drawContours(display_roi, valid_contours, -1, (0, 255, 0), 1)
        
        # Detect corners
        all_corners = BasicImageProcess.detect_corners(
            cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), 
            valid_contours, 
            self.config
        )
        
        # Draw corners on display image
        for corner in all_corners:
            x, y = corner.ravel()
            cv2.circle(display_roi, (int(x), int(y)), 3, (0, 0, 255), -1)
        
        # Create visualization
        combined_frames = BasicImageProcess.create_visualization(display_roi, edges, thresh, filtered)
        
        # Display results
        # cv2.imshow('Original Image', frame)
        cv2.imshow('Processing Steps', combined_frames)
        
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
        return all_corners

    # Key 2
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
    
    # Key 3
    def remove_background(self, reference_path: str, test_path: str):
        """Compare a test image with a reference image to find anomalies, focusing ONLY on the main object."""
        # Load images
        reference_image = ImageLoader.load_image(reference_path)
        test_image = ImageLoader.load_image(test_path)
        
        if reference_image is None or test_image is None:
            print("Error loading one or both images")
            return None
        
        # Preprocess both images the same way
        filtered_ref, roi_ref = ImagePreprocessor.preprocess_image(reference_image, self.config)
        filtered_test, roi_test = ImagePreprocessor.preprocess_image(test_image, self.config)
        
        # 1. Extract the main object in both images
        ref_mask = ImagePreprocessor._extract_main_object(filtered_ref)
        test_mask = ImagePreprocessor._extract_main_object(filtered_test)
        
        # 2. Apply masks to original images
        ref_object = cv2.bitwise_and(roi_ref, roi_ref, mask=ref_mask)
        test_object = cv2.bitwise_and(roi_test, roi_test, mask=test_mask)
        
        # 3. Compare only the masked regions
        diff_mask, similarity_score = ImageComparator.compare_images(ref_object, test_object)
        
        # 4. Apply additional filtering to remove small anomalies
        kernel = np.ones((3, 3), np.uint8)
        filtered_diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, kernel)
        
        # 5. Apply strong erosion to focus only on significant center anomalies
        erosion_kernel = np.ones((5, 5), np.uint8)
        center_diff_mask = cv2.erode(filtered_diff_mask, erosion_kernel, iterations=3)
        
        # Detect specific anomalies with higher threshold
        anomalies = ImageComparator.detect_anomalies(center_diff_mask, min_area=10)
        
        # Create visualization
        comparison_viz = ImageComparator.highlight_anomalies(ref_object, test_object, center_diff_mask)
        
        # Display results
        cv2.imshow('Objects Only Comparison', comparison_viz)
        
        # Print analysis
        print(f"Similarity: {100-similarity_score:.2f}% (Difference: {similarity_score:.2f}%)")
        print(f"Found {len(anomalies)} significant anomaly regions in the center of the object")
        
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

    # Key 4
    def group_images(self, image_path: str):
        image = ImageLoader.load_image(image_path)
        if image is None:
            print("Error loading image")
            return None
            
        # Diviser l'image en quatre quadrants
        quadrants = ImagePreprocessor.divide_image_into_four(image)
        if len(quadrants) != 4:
            print("Failed to divide image into quadrants")
            return None
            
        print("\n=== Analysing quadrant similarities ===")
        
        # Comparer chaque quadrant avec les autres
        similarities = []
        for i in range(len(quadrants)):
            for j in range(i+1, len(quadrants)):
                # Prétraiter les deux quadrants
                filtered_i, _ = ImagePreprocessor.preprocess_image(quadrants[i], self.config)
                filtered_j, _ = ImagePreprocessor.preprocess_image(quadrants[j], self.config)
                
                # Calculer la similarité
                tolerance = 4  # Valeur de seuil pour les différences
                blur_size = 0  # Taille du flou pour réduire le bruit
                _, similarity_score = ImageComparator.compare_images(filtered_i, filtered_j, tolerance, blur_size)
                
                # Enregistrer les résultats
                quad_names = ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"]
                similarities.append({
                    'pair': (i, j),
                    'names': (quad_names[i], quad_names[j]),
                    'score': similarity_score
                })
                print(f"Similarity between {quad_names[i]} and {quad_names[j]}: {100-similarity_score:.2f}% (Diff: {similarity_score:.2f}%)")
        
        # Déterminer les groupes en fonction des scores de similarité
        similarity_threshold = 0.5  # Seuil pour considérer deux images comme similaires
        groups = []
        outliers = []
        processed_indices = set()
        
        # Trier les similarités
        similarities.sort(key=lambda x: x['score'])
        
        # Créer des groupes basés sur la similarité
        for sim in similarities:
            i, j = sim['pair']
            score = sim['score']
            
            # Si la différence est faible, les deux quadrants sont similaires
            if score < similarity_threshold:
                # Chercher si l'un des quadrants est déjà dans un groupe
                found_group = False
                for group in groups:
                    if i in group or j in group:
                        # Ajouter l'autre quadrant au groupe
                        group.add(i)
                        group.add(j)
                        processed_indices.add(i)
                        processed_indices.add(j)
                        found_group = True
                        break
                
                # Si aucun des deux n'est dans un groupe, créer un nouveau groupe
                if not found_group:
                    groups.append(set([i, j]))
                    processed_indices.add(i)
                    processed_indices.add(j)
            else:
                # Pour les dissimilarités significatives, noter pour analyse
                if i not in processed_indices:
                    outliers.append(i)
                    processed_indices.add(i)
                if j not in processed_indices:
                    outliers.append(j)
                    processed_indices.add(j)
        
        # Vérifier s'il y a des indices non traités (0,1,2,3)
        for i in range(4):
            if i not in processed_indices:
                outliers.append(i)
        
        # Afficher les résultats
        quad_names = ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"]
        
        print("\n=== Grouping Results ===")
        print(f"Found {len(groups)} groups of similar quadrants:")
        for i, group in enumerate(groups):
            group_indices = sorted(list(group))
            group_names = [quad_names[idx] for idx in group_indices]
            print(f"  Group {i+1}: {', '.join(group_names)}")
        
        print(f"\nFound {len(outliers)} outlier quadrants:")
        for idx in outliers:
            print(f"  Outlier: {quad_names[idx]}")
        
        # Visualiser les groupes
        # Créer une image de sortie montrant les groupes avec des couleurs différentes
        output_img = image.copy()
        height, width = image.shape[:2]
        mid_h = height // 2
        mid_w = width // 2
        
        # Couleurs pour les groupes (BGR)
        group_colors = [
            (0, 255, 0),    # Vert
            (0, 255, 255),  # Jaune
            (255, 0, 0),    # Bleu
            (255, 0, 255)   # Magenta
        ]
        
        # Couleur pour les outliers
        outlier_color = (0, 0, 255)  # Rouge
        
        # Dessiner des rectangles colorés pour les groupes
        positions = [
            (0, 0, mid_w, mid_h),         # Top-Left
            (mid_w, 0, width, mid_h),     # Top-Right
            (0, mid_h, mid_w, height),    # Bottom-Left
            (mid_w, mid_h, width, height) # Bottom-Right
        ]
        
        # Dessiner d'abord les groupes
        for i, group in enumerate(groups):
            color = group_colors[i % len(group_colors)]
            for idx in group:
                x, y, x2, y2 = positions[idx]
                cv2.rectangle(output_img, (x+5, y+5), (x2-5, y2-5), color, 3)
                cv2.putText(output_img, f"Group {i+1}", (x+15, y+30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Dessiner les outliers
        for idx in outliers:
            x, y, x2, y2 = positions[idx]
            cv2.rectangle(output_img, (x+5, y+5), (x2-5, y2-5), outlier_color, 3)
            cv2.putText(output_img, "Outlier", (x+15, y+30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, outlier_color, 1)
        
        # Afficher le résultat
        cv2.imshow("Quadrant Groups Analysis", output_img)
        
        print("\n\033[91m Press 'q' to close windows / 'r' to restart program \033[0m")
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r'):
                cv2.destroyAllWindows()
                main()  # Redémarrer le programme
                return None
        
        cv2.destroyAllWindows()
        
        # Retourner les résultats pour une utilisation ultérieure
        return {
            'quadrants': quadrants,
            'groups': [list(group) for group in groups],
            'outliers': outliers,
            'similarities': similarities
        }



####################################################################################################
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
    print("3. Compare two images focusing only on the main object")
    print("4. Analyze image by dividing it into four quadrants")

    choice = input("Enter your choice (1, 2, 3 or 4): ")
    
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
            corners = pipeline.basic_simple_image_process(file_path)
            if corners:
                print(f"Found {len(corners)} corners in the image")
        else:
            print("No file selected")
            
    elif choice == '2':
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
        
    elif choice == '3':
        # Object-focused comparison mode
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
            
        print("\033[93m Select Test/Anomali Image \033[0m")
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
            
        pipeline.remove_background(reference_path, test_path)
    
    elif choice == '4':
        # Quadrant analysis mode
        file_path = filedialog.askopenfilename(
            title="Select Image File for Quadrant Analysis",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            results = pipeline.group_images(file_path)
            if not results:
                print("Analysis failed or no results generated")
        else:
            print("No file selected")
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()