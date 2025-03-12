import cv2
import numpy as np
import tkinter as tk
from typing import Optional
from tkinter import filedialog
from dataclasses import dataclass

@dataclass
class ImageProcessingConfig:
    """Configuration parameters for pattern image analysis."""
    #----------Preprocessing parameters----------#
    # CLACHE parameters
    tileGridSize = (4, 4)
    clipLimit = 3
    #Gaussian Blur parameters
    ksize = (5, 5)
    sigmaX = 0
    #Fourier Transform parameters
    inner_radius = 30
    outer_radius = 90


##############################################General###################################################### 
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
    def preprocess_image(image: np.ndarray, config: ImageProcessingConfig) -> np.ndarray:
        # Convert to grayscale if the image is not already in grayscale
        if len(image.shape) == 3:
            gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = image.copy()
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for contrast enhancement
        clahe = cv2.createCLAHE(config.clipLimit, config.tileGridSize)
        enhanced_frame = clahe.apply(gray_frame)
        blurred = cv2.GaussianBlur(enhanced_frame, config.ksize, config.sigmaX)
        
        # Extraction of fabric periodic pattern
        # Apply Fourier Transform to find periodic patterns
        dft = cv2.dft(np.float32(blurred), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        # Create a mask to keep only the high-frequency components (fabric pattern)
        rows, cols = blurred.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.zeros((rows, cols, 2), np.uint8)

        # Keep only the frequency components in a specific band
        # This targets the periodic patterns typically found in fabrics
        for i in range(rows):
            for j in range(cols):
                dist = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
                if config.inner_radius < dist < config.outer_radius:
                    mask[i, j] = 1

        # Apply the mask and perform inverse DFT
        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        filtered = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        # Normalize the result for better visualization
        cv2.normalize(filtered, filtered, 0, 255, cv2.NORM_MINMAX)
        filtered = filtered.astype(np.uint8)

        return filtered

class ImageTextureFiltering:
    """Handles texture-based filtering and segmentation of images."""
    
    @staticmethod
    def apply_gabor_filter(image: np.ndarray, ksize: int = 31, sigma: float = 4.0, 
                           theta: float = 0, lambd: float = 10.0, 
                           gamma: float = 0.5, psi: float = 0) -> np.ndarray:
        """
        Apply Gabor filter for texture analysis.
        
        Parameters:
            image: Input grayscale image
            ksize: Size of the Gabor kernel
            sigma: Standard deviation of the Gaussian envelope
            theta: Orientation of the Gabor filter
            lambd: Wavelength of the sinusoidal factor
            gamma: Spatial aspect ratio
            psi: Phase offset
        """
        # Ensure image is grayscale
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image.copy()
            
        # Apply Gabor filter
        gabor_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
        filtered_image = cv2.filter2D(gray_image, cv2.CV_8UC3, gabor_kernel)
        
        return filtered_image
    
    @staticmethod
    def multi_orientation_gabor(image: np.ndarray, ksize: int = 31, sigma: float = 4.0,
                               lambd: float = 10.0, gamma: float = 0.5) -> np.ndarray:
        """Apply Gabor filters at multiple orientations and combine the results."""
        # Define orientations (0°, 45°, 90°, 135°)
        orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        # Apply Gabor filter at each orientation
        gabor_responses = []
        for theta in orientations:
            filtered = ImageTextureFiltering.apply_gabor_filter(image, ksize, sigma, theta, lambd, gamma)
            gabor_responses.append(filtered)
        
        # Combine responses (maximum response at each pixel)
        combined = np.max(gabor_responses, axis=0)
        
        return combined
    
    @staticmethod
    def compute_lbp(image: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
        """
        Compute Local Binary Pattern for texture analysis.
        
        A simplified LBP implementation without using specialized libraries.
        """
        # Ensure image is grayscale
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image.copy()
        
        rows, cols = gray_image.shape
        result = np.zeros((rows, cols), dtype=np.uint8)
        
        for i in range(radius, rows - radius):
            for j in range(radius, cols - radius):
                center = gray_image[i, j]
                binary = 0
                
                # Compute LBP code
                if gray_image[i - radius, j - radius] >= center:
                    binary |= 1 << 0
                if gray_image[i - radius, j] >= center:
                    binary |= 1 << 1
                if gray_image[i - radius, j + radius] >= center:
                    binary |= 1 << 2
                if gray_image[i, j + radius] >= center:
                    binary |= 1 << 3
                if gray_image[i + radius, j + radius] >= center:
                    binary |= 1 << 4
                if gray_image[i + radius, j] >= center:
                    binary |= 1 << 5
                if gray_image[i + radius, j - radius] >= center:
                    binary |= 1 << 6
                if gray_image[i, j - radius] >= center:
                    binary |= 1 << 7
                    
                result[i, j] = binary
        
        return result
    
    @staticmethod
    def segment_by_texture(image: np.ndarray, n_segments: int = 3) -> np.ndarray:
        """
        Segment image based on texture features using K-means clustering.
        
        Parameters:
            image: Input image
            n_segments: Number of texture segments to create
        """
        # Compute texture features (LBP)
        lbp = ImageTextureFiltering.compute_lbp(image)
        
        # Reshape for K-means
        features = lbp.reshape((-1, 1)).astype(np.float32)
        
        # Apply K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(features, n_segments, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Reshape labels back to image dimensions
        segmented = labels.reshape(lbp.shape).astype(np.uint8)
        
        # Scale to 0-255 for visualization
        segmented = (segmented * (255 // (n_segments-1))).astype(np.uint8)
        
        return segmented
    
    @staticmethod
    def remove_noise(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Remove noise using morphological operations."""
        # Apply morphological opening to remove small noise
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        
        return opening


##############################################Pipeline######################################################
class ImagePipeline:
    def defect_detection(self, reference_path: str):
        # Load images
        reference_image = ImageLoader.load_image(reference_path)
        
        if reference_image is None:
            print("Error loading the image")
            return None
        
        # Preprocess image
        filtered_ref = ImagePreprocessor.preprocess_image(reference_image, ImageProcessingConfig)
        
        # Apply texture filtering
        # Get Gabor filter response
        gabor_result = ImageTextureFiltering.multi_orientation_gabor(filtered_ref)
        
        # Segment image based on texture
        segmented = ImageTextureFiltering.segment_by_texture(filtered_ref, n_segments=4)
        
        # Remove noise
        denoised = ImageTextureFiltering.remove_noise(segmented)
        
        # Display results
        cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
        cv2.imshow('Original', reference_image)
        
        cv2.namedWindow('Filtered Reference', cv2.WINDOW_NORMAL)
        cv2.imshow('Filtered Reference', filtered_ref)
        
        cv2.namedWindow('Gabor Filter Response', cv2.WINDOW_NORMAL)
        cv2.imshow('Gabor Filter Response', gabor_result)
        
        cv2.namedWindow('Texture Segmentation', cv2.WINDOW_NORMAL)
        cv2.imshow('Texture Segmentation', segmented)
        
        cv2.namedWindow('Denoised Segmentation', cv2.WINDOW_NORMAL)
        cv2.imshow('Denoised Segmentation', denoised)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return filtered_ref, gabor_result, segmented, denoised
        

    

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

        
    pipeline.defect_detection(reference_path)


if __name__ == "__main__":
    main()