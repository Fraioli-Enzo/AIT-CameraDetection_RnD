import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import correlate2d
from skimage.feature import local_binary_pattern

class FabricDefectDetector:
    """
    Fabric defect detection using Completed Local Quartet Patterns (CLQP)
    """
    
    def __init__(self, P=8, R=1):
        """
        Initialize the detector with parameters for CLQP
        
        Parameters:
        -----------
        P : int
            Number of neighboring pixels
        R : int
            Radius of the neighborhood
        """
        self.P = P
        self.R = R
        self.threshold = None
        self.pattern_size = None
        self.global_features = None
    
    def calculate_clqp(self, image):
        """
        Calculate Completed Local Quartet Patterns (CLQP)
        
        Parameters:
        -----------
        image : 2D numpy array
            Input grayscale image
            
        Returns:
        --------
        clqp_features : numpy array
            CLQP features extracted from the image
        """
        # Step 1: Calculate sign component (similar to traditional LBP)
        lbp = local_binary_pattern(image, self.P, self.R, method='uniform')
        
        # Step 2: Calculate magnitude component
        h, w = image.shape
        magnitude = np.zeros_like(image, dtype=np.float64)
        
        for i in range(self.R, h-self.R):
            for j in range(self.R, w-self.R):
                center = image[i, j]
                # Calculate magnitude differences with neighbors
                diff_sum = 0
                for p in range(self.P):
                    # Calculate neighbor coordinates
                    x = i + int(self.R * np.cos(2 * np.pi * p / self.P))
                    y = j + int(self.R * np.sin(2 * np.pi * p / self.P))
                    # Add absolute difference
                    diff_sum += abs(image[x, y] - center)
                magnitude[i, j] = diff_sum / self.P
        
        # Step 3: Calculate central pixel intensity
        center_intensity = np.zeros_like(image, dtype=np.float64)
        for i in range(self.R, h-self.R):
            for j in range(self.R, w-self.R):
                avg_intensity = np.mean(image[max(0,i-self.R):min(h,i+self.R+1), 
                                        max(0,j-self.R):min(w,j+self.R+1)])
                center_intensity[i, j] = 1 if image[i, j] >= avg_intensity else 0
        
        # Compute histograms for each component and concatenate
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=self.P+2, range=(0, self.P+2))
        magnitude_hist, _ = np.histogram(magnitude.ravel(), bins=self.P+2)
        center_hist, _ = np.histogram(center_intensity.ravel(), bins=2)
        
        # Normalize histograms
        lbp_hist = lbp_hist.astype(float) / np.sum(lbp_hist)
        magnitude_hist = magnitude_hist.astype(float) / np.sum(magnitude_hist)
        center_hist = center_hist.astype(float) / np.sum(center_hist)
        
        # Combine all features to form CLQP
        clqp_features = np.concatenate((lbp_hist, magnitude_hist, center_hist))
        
        return clqp_features
    
    def determine_pattern_size(self, image):
        """
        Determine the size of the repetitive pattern using autocorrelation
        
        Parameters:
        -----------
        image : 2D numpy array
            Input grayscale image
            
        Returns:
        --------
        pattern_size : tuple
            (height, width) of the repetitive pattern
        """
        # Apply autocorrelation
        auto_corr = correlate2d(image, image, mode='same')
        
        # Find peaks in autocorrelation
        h, w = auto_corr.shape
        center_y, center_x = h // 2, w // 2
        
        # Ignore the central peak
        auto_corr[center_y-5:center_y+6, center_x-5:center_x+6] = 0
        
        # Find the closest peak to center
        y_peaks, x_peaks = np.where(auto_corr > 0.6 * np.max(auto_corr))
        
        if len(y_peaks) == 0 or len(x_peaks) == 0:
            # Default to small windows if no clear pattern
            return (32, 32)
        
        # Calculate distances from center
        distances = np.sqrt((y_peaks - center_y)**2 + (x_peaks - center_x)**2)
        closest_idx = np.argmin(distances)
        
        # Pattern size is twice the distance to the closest peak
        pattern_height = 2 * abs(y_peaks[closest_idx] - center_y)
        pattern_width = 2 * abs(x_peaks[closest_idx] - center_x)
        
        # Ensure minimum size
        pattern_height = max(16, pattern_height)
        pattern_width = max(16, pattern_width)
        
        return (pattern_height, pattern_width)
    
    def train(self, train_image):
        """
        Training phase for defect detection using a defect-free image
        
        Parameters:
        -----------
        train_image : 2D numpy array
            Defect-free training image
        """
        # Ensure grayscale
        if len(train_image.shape) > 2:
            train_image = cv2.cvtColor(train_image, cv2.COLOR_BGR2GRAY)
        
        # Determine pattern size
        self.pattern_size = self.determine_pattern_size(train_image)
        print(f"Determined pattern size: {self.pattern_size}")
        
        # Extract global features from the entire image
        self.global_features = self.calculate_clqp(train_image)
        
        # Divide image into non-overlapping windows
        h, w = train_image.shape
        max_dissimilarity = 0
        
        for y in range(0, h - self.pattern_size[0] + 1, self.pattern_size[0]):
            for x in range(0, w - self.pattern_size[1] + 1, self.pattern_size[1]):
                # Extract window
                window = train_image[y:y+self.pattern_size[0], x:x+self.pattern_size[1]]
                
                # Extract features from window
                window_features = self.calculate_clqp(window)
                
                # Calculate non-similarity (using chi-square distance)
                dissimilarity = 0.5 * np.sum(
                    (window_features - self.global_features)**2 / 
                    (window_features + self.global_features + 1e-10)
                )
                
                max_dissimilarity = max(max_dissimilarity, dissimilarity)
        
        # Set threshold with a safety margin (20%)
        self.threshold = max_dissimilarity * 1.2
        print(f"Training complete. Threshold: {self.threshold:.4f}")
    
    def detect(self, test_image, overlap=0.5):
        """
        Detect defects in test image using the trained model
        
        Parameters:
        -----------
        test_image : 2D numpy array
            Test image to check for defects
        overlap : float
            Overlap ratio for sliding windows (for improved localization)
            
        Returns:
        --------
        defect_map : 2D numpy array
            Binary map showing defective regions
        """
        if self.threshold is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Ensure grayscale
        if len(test_image.shape) > 2:
            test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        
        h, w = test_image.shape
        step_y = int(self.pattern_size[0] * (1 - overlap))
        step_x = int(self.pattern_size[1] * (1 - overlap))
        
        # Initialize vote map for majority voting
        vote_map = np.zeros((h, w), dtype=np.int32)
        count_map = np.zeros((h, w), dtype=np.int32)
        
        for y in range(0, h - self.pattern_size[0] + 1, step_y):
            for x in range(0, w - self.pattern_size[1] + 1, step_x):
                # Extract window
                window = test_image[y:y+self.pattern_size[0], x:x+self.pattern_size[1]]
                
                # Extract features from window
                window_features = self.calculate_clqp(window)
                
                # Calculate non-similarity
                dissimilarity = 0.5 * np.sum(
                    (window_features - self.global_features)**2 / 
                    (window_features + self.global_features + 1e-10)
                )
                
                # Determine if window is defective
                is_defective = dissimilarity > self.threshold
                
                # Update vote map
                vote_map[y:y+self.pattern_size[0], x:x+self.pattern_size[1]] += int(is_defective)
                count_map[y:y+self.pattern_size[0], x:x+self.pattern_size[1]] += 1
        
        # Apply majority voting (MD algorithm)
        with np.errstate(divide='ignore', invalid='ignore'):
            defect_prob = np.divide(vote_map, count_map)
            defect_prob = np.nan_to_num(defect_prob)
        
        # Create final defect map
        defect_map = (defect_prob > 0.5).astype(np.uint8) * 255
        
        return defect_map
    
    def visualize_results(self, original_image, defect_map):
        """
        Visualize detected defects
        
        Parameters:
        -----------
        original_image : numpy array
            Original test image
        defect_map : 2D numpy array
            Binary map showing defective regions
        """
        # Create a color copy of the original image
        if len(original_image.shape) == 2:
            original_color = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        else:
            original_color = original_image.copy()
        
        # Create colored defect overlay
        defect_overlay = np.zeros_like(original_color)
        defect_overlay[defect_map > 0] = [0, 0, 255]  # Red color for defects
        
        # Create alpha blend
        alpha = 0.6
        blended = cv2.addWeighted(original_color, 1-alpha, defect_overlay, alpha, 0)
        
        # Display results
        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        if len(original_image.shape) == 3:
            plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(original_image, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(132)
        plt.imshow(defect_map, cmap='gray')
        plt.title('Defect Map')
        plt.axis('off')
        
        plt.subplot(133)
        plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
        plt.title('Defects Highlighted')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()


def main():
    """
    Main function demonstrating the fabric defect detection pipeline
    """
    detector = FabricDefectDetector(P=8, R=1)
    
    # Replace these paths with your own images
    train_image_path = "Images/pattern_19.png"
    test_image_path = "Images/pattern_19.png"
    
    # Load images
    try:
        train_image = cv2.imread(train_image_path, cv2.IMREAD_GRAYSCALE)
        test_image = cv2.imread(test_image_path)
        
        if train_image is None or test_image is None:
            print("Error: Could not read image files!")
            return
            
        # Train the detector
        detector.train(train_image)
        
        # Detect defects
        defect_map = detector.detect(test_image)
        
        # Visualize results
        detector.visualize_results(test_image, defect_map)
        
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()