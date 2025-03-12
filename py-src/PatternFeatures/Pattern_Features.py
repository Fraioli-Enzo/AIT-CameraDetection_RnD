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

        # Maintain a copy of the original colored image for RGB analysis
        original_image = image.copy()

        # Generate RGB channel profiles
        if len(original_image.shape) == 3:
            # Split the image into its BGR channels (OpenCV uses BGR)
            b, g, r = cv2.split(original_image)

            # Generate and save RGB intensity profiles
            ImagePreprocessor._generate_rgb_intensity_profiles(r, g, b)
        
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
        folder_path = 'py-src/PatternFeatures'
        # Save and display horizontal profile
        horizontal_profile_path = f'{folder_path}/horizontal_intensity_profile.png'
        ImagePreprocessor._plot_intensity_profile(
            horizontal_profile, 
            range(width),
            'Horizontal Intensity Profile (Left to Right)',
            'X Position (pixels)',
            horizontal_profile_path
        )
        
        # Save and display vertical profile
        vertical_profile_path = f'{folder_path}/vertical_intensity_profile.png'
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
    def _generate_rgb_intensity_profiles(r: np.ndarray, g: np.ndarray, b: np.ndarray):
        """Generate and save RGB channel intensity profiles."""
        height, width = r.shape
        folder_path = 'py-src/PatternFeatures'

        # Generate horizontal intensity profiles for each channel
        r_horizontal = np.mean(r, axis=0)
        g_horizontal = np.mean(g, axis=0)
        b_horizontal = np.mean(b, axis=0)

        # Generate vertical intensity profiles for each channel
        r_vertical = np.mean(r, axis=1)
        g_vertical = np.mean(g, axis=1)
        b_vertical = np.mean(b, axis=1)

        # Create and save horizontal RGB profile
        plt.figure(figsize=(10, 6))
        plt.plot(range(width), r_horizontal, 'r-', label='Red')
        plt.plot(range(width), g_horizontal, 'g-', label='Green')
        plt.plot(range(width), b_horizontal, 'b-', label='Blue')
        plt.title('Horizontal RGB Intensity Profiles')
        plt.xlabel('X Position (pixels)')
        plt.ylabel('Average Intensity')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{folder_path}/horizontal_rgb_profile.png')
        plt.close()

        # Create and save vertical RGB profile
        plt.figure(figsize=(10, 6))
        plt.plot(range(height), r_vertical, 'r-', label='Red')
        plt.plot(range(height), g_vertical, 'g-', label='Green')
        plt.plot(range(height), b_vertical, 'b-', label='Blue')
        plt.title('Vertical RGB Intensity Profiles')
        plt.xlabel('Y Position (pixels)')
        plt.ylabel('Average Intensity')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{folder_path}/vertical_rgb_profile.png')
        plt.close()

        cv2.imshow('Horizontal RGB Intensity Profiles', cv2.imread(f'{folder_path}/horizontal_rgb_profile.png'))
        cv2.imshow('Vertical RGB Intensity Profiles', cv2.imread(f'{folder_path}/vertical_rgb_profile.png'))
        cv2.waitKey(0)


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


class ImagePattern:
    @staticmethod
    def separate_pattern(image: np.ndarray) -> np.ndarray:
        """
        Detects peaks in horizontal and vertical intensity profiles and draws lines on the image.
        
        Args:
            image: Preprocessed grayscale image
            
        Returns:
            Image with pattern separation lines
        """
        height, width = image.shape
        result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Calculate intensity profiles
        horizontal_profile = np.mean(image, axis=0)  # Average by column
        vertical_profile = np.mean(image, axis=1)    # Average by row
        
        # Load saved profiles for analysis (if needed)
        folder_path = 'py-src/PatternFeatures'
        
        # Find peaks in horizontal profile (columns)
        horizontal_peaks = []
        window_size = 30
        threshold = np.mean(horizontal_profile) 
        
        for i in range(window_size, len(horizontal_profile) - window_size):
            window = horizontal_profile[i-window_size:i+window_size]
            if horizontal_profile[i] == max(window) and horizontal_profile[i] > threshold:
                horizontal_peaks.append(i)
        
        # Find peaks in vertical profile (rows)
        vertical_peaks = []
        threshold = np.mean(vertical_profile)
        
        for i in range(window_size, len(vertical_profile) - window_size):
            window = vertical_profile[i-window_size:i+window_size]
            if vertical_profile[i] == max(window) and vertical_profile[i] > threshold:
                vertical_peaks.append(i)
        
        # Draw vertical lines (at horizontal peaks)
        for x in horizontal_peaks:
            cv2.line(result_image, (x, 0), (x, height-1), (0, 0, 255), 1)
            
        # Draw horizontal lines (at vertical peaks)
        for y in vertical_peaks:
            cv2.line(result_image, (0, y), (width-1, y), (0, 255, 0), 1)
            
        print(f"Found {len(horizontal_peaks)} vertical lines and {len(vertical_peaks)} horizontal lines")
        return result_image

##############################################Pipeline######################################################
class ImagePipeline:
    def defect_detection(self, reference_path: str):
        # Load images
        reference_image = ImageLoader.load_image(reference_path)
        
        if reference_image is None:
            print("Error loading the image")
            return None
        
        # Preprocess image
        image_gray = ImagePreprocessor.preprocess_image(reference_image, ImageProcessingConfig)

        print("\033[91m Image preprocessing completed, press 'q' to exit the window. \033[0m")
        cv2.imshow('Patern after preprocessing', image_gray)
        cv2.waitKey(0)

        # Separate pattern
        pattern_image = ImagePattern.separate_pattern(image_gray)
        
        print("\033[91m Image preprocessing completed, press 'q' to exit the window. \033[0m")
        cv2.imshow('Patern after preprocessing', pattern_image)
        cv2.waitKey(0)

        return image_gray
        

    

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