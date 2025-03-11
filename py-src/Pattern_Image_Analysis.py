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



##############################################Pipeline######################################################
class ImagePipeline:
    """Main pipeline for image corner detection."""
    def compare_with_reference(self, reference_path: str):
        """Compare a test image with a reference image to find anomalies."""
        # Load images
        reference_image = ImageLoader.load_image(reference_path)
        
        if reference_image is None:
            print("Error loading one or both images")
            return None
        
        # Preprocess both images the same way
        filtered_ref = ImagePreprocessor.preprocess_image(reference_image, ImageProcessingConfig)

        # Display the filtered reference image
        cv2.namedWindow('Filtered Reference', cv2.WINDOW_NORMAL)
        cv2.imshow('Filtered Reference', filtered_ref)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Return the filtered reference image for further use if needed
        return filtered_ref
        

    

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

        
    pipeline.compare_with_reference(reference_path)


if __name__ == "__main__":
    main()