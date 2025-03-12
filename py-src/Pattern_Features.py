import cv2
import numpy as np
import tkinter as tk
from typing import Optional
from tkinter import filedialog
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class ImageProcessingConfig:
    """Configuration parameters for pattern image analysis."""


##############################################General###################################################### 
class ImageLoader:
    """Handles loading and basic image validation."""
    @staticmethod
    def load_image(image_path: str) -> Optional[np.ndarray]:
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not read image {image_path}")
        return frame
    
    @staticmethod
    def choose_image() -> Optional[np.ndarray]:
        image_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("All files", "*.*")
            ]
        )
        if not image_path:
            print("No image selected")
            return None
        return image_path

class ImagePreprocessor: 
    @staticmethod
    def preprocess_image(image: np.ndarray, config: ImageProcessingConfig) -> np.ndarray:
        """
        Preprocesses an image and generates intensity profile visualizations.
        
        Args:
            image: Input image as numpy array
            config: Configuration parameters for image processing
            
        Returns:
            Preprocessed grayscale image
        """
        # Convert to grayscale if the image is in color
        grayscale_image = ImagePreprocessor._convert_to_grayscale(image)
        
        # Apply noise reduction
        filtered_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)
        
        # Generate and save intensity profile visualizations
        ImagePreprocessor._generate_intensity_profiles(filtered_image)
        
        return filtered_image
    
    @staticmethod
    def _convert_to_grayscale(image: np.ndarray):
        """Convert an image to grayscale if not already."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image.copy()
    
    @staticmethod
    def _generate_intensity_profiles(grayscale_image: np.ndarray):
        """Generate and visualize horizontal and vertical intensity profiles."""
        height, width = grayscale_image.shape
        print(f"Image dimensions: {height}x{width}")
        
        # Calculate average intensities
        horizontal_profile = np.mean(grayscale_image, axis=0)  # Average by column (left to right)
        vertical_profile = np.mean(grayscale_image, axis=1)    # Average by row (top to bottom)
        
        # Save and display horizontal profile
        horizontal_profile_path = 'horizontal_intensity_profile.png'
        ImagePreprocessor._plot_intensity_profile(
            horizontal_profile, 
            range(width),
            'Horizontal Intensity Profile (Left to Right)',
            'X Position (pixels)',
            horizontal_profile_path
        )
        
        # Save and display vertical profile
        vertical_profile_path = 'vertical_intensity_profile.png'
        ImagePreprocessor._plot_intensity_profile(
            vertical_profile, 
            range(height),
            'Vertical Intensity Profile (Top to Bottom)',
            'Y Position (pixels)',
            vertical_profile_path
        )
        
        # Display the profile images
        ImagePreprocessor._display_profile_images(horizontal_profile_path, vertical_profile_path)
    
    @staticmethod
    def _plot_intensity_profile(intensity_values, positions, title, xlabel, save_path):
        """Create and save an intensity profile plot."""
        plt.figure(figsize=(10, 6))
        plt.plot(positions, intensity_values)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel('Average Intensity')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    @staticmethod
    def _display_profile_images(horizontal_path, vertical_path):
        """Display the saved profile images."""
        horizontal_img = cv2.imread(horizontal_path)
        vertical_img = cv2.imread(vertical_path)
        
        if horizontal_img is not None and vertical_img is not None:
            cv2.imshow('Horizontal Intensity Profile', horizontal_img)
            cv2.imshow('Vertical Intensity Profile', vertical_img)

##############################################Pipeline######################################################
class ImagePipeline:
    def defect_detection(self, reference_path: str):
        # Load images
        reference_image = ImageLoader.load_image(reference_path)
        
        if reference_image is None:
            print("Error loading the image")
            return None
        
        # Preprocess image
        image_patern = ImagePreprocessor.preprocess_image(reference_image, ImageProcessingConfig)
        
        print("\033[91m Image preprocessing completed, press 'q' to exit the window. \033[0m")
        cv2.imshow('Patern after preprocessing', image_patern)
        cv2.waitKey(0)

        return image_patern
        

    

##############################################Main######################################################
def main():
    """Main function to select and process images."""
    # Create a simple Tkinter root window
    root = tk.Tk()
    root.withdraw()  

    pipeline = ImagePipeline()

    # Image comparison mode (with all features)
    print("\033[93m Select reference image (good/normal image): \033[0m")
    reference_path = ImageLoader.choose_image()
    if reference_path is None:
        return
    
    pipeline.defect_detection(reference_path)


if __name__ == "__main__":
    main()