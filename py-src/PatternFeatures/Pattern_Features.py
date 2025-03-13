import cv2
import numpy as np
import tkinter as tk
from typing import Optional
from tkinter import filedialog
import matplotlib.pyplot as plt
from dataclasses import dataclass
import json
import os

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
        
        # Generate and save intensity profile visualizations
        ImagePreprocessor._generate_intensity_profiles(grayscale_image)

        # Maintain a copy of the original colored image for RGB analysis
        original_image = image.copy()

        # Generate RGB channel profiles
        if len(original_image.shape) == 3:
            # Split the image into its BGR channels (OpenCV uses BGR)
            b, g, r = cv2.split(original_image)

            # Generate and save RGB intensity profiles
            ImagePreprocessor._generate_rgb_intensity_profiles(r, g, b)
        
        return grayscale_image
    
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

        # Create and save combined RGB intensity plots
        plt.figure(figsize=(10, 6))

        # Combined RGB horizontal profile (sum of channels)
        combined_horizontal = r_horizontal + g_horizontal + b_horizontal
        plt.plot(range(width), combined_horizontal, 'k-', label='Combined RGB')
        plt.title('Combined Horizontal RGB Intensity Profile')
        plt.xlabel('X Position (pixels)')
        plt.ylabel('Combined Intensity')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{folder_path}/combined_horizontal_rgb_profile.png')
        plt.close()

        # Combined RGB vertical profile (sum of channels)
        plt.figure(figsize=(10, 6))
        combined_vertical = r_vertical + g_vertical + b_vertical
        plt.plot(range(height), combined_vertical, 'k-', label='Combined RGB')
        plt.title('Combined Vertical RGB Intensity Profile')
        plt.xlabel('Y Position (pixels)')
        plt.ylabel('Combined Intensity')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{folder_path}/combined_vertical_rgb_profile.png')
        plt.close()

        cv2.imshow('Horizontal RGB Intensity Profiles', cv2.imread(f'{folder_path}/horizontal_rgb_profile.png'))
        cv2.imshow('Vertical RGB Intensity Profiles', cv2.imread(f'{folder_path}/vertical_rgb_profile.png'))
        cv2.imshow('Vertical new', cv2.imread(f'{folder_path}/combined_vertical_rgb_profile.png'))
        # Save all profile data to JSON for use by other functions

        # Create a dictionary to store all the profile data
        profile_data = {
            "horizontal_rgb": {
                "red": r_horizontal.tolist(),
                "green": g_horizontal.tolist(),
                "blue": b_horizontal.tolist()
            },
            "vertical_rgb": {
                "red": r_vertical.tolist(),
                "green": g_vertical.tolist(),
                "blue": b_vertical.tolist()
            },
            "image_dimensions": {
                "height": height,
                "width": width
            }
        }

        # Create directory if it doesn't exist
        os.makedirs(folder_path, exist_ok=True)

        # Write the data to a JSON file
        with open(f'{folder_path}/profile_data.json', 'w') as json_file:
            json.dump(profile_data, json_file, indent=4)

        print(f"Profile data saved to {folder_path}/profile_data.json")
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
        Separate pattern features based on intensity profiles.
        Identifies peaks in RGB profiles and marks them on the grayscale image.
        Uses a 10-pixel window to reduce density of lines.
        
        Args:
            image: Preprocessed grayscale image
            
        Returns:
            Image with pattern features marked
        """
        folder_path = 'py-src/PatternFeatures'
        profile_path = f'{folder_path}/profile_data.json'
        
        # Check if profile data exists
        if not os.path.exists(profile_path):
            print(f"Profile data not found at {profile_path}")
            return image
        
        # Load profile data from JSON
        try:
            with open(profile_path, 'r') as file:
                profile_data = json.load(file)
        except Exception as e:
            print(f"Error loading profile data: {e}")
            return image
        
        # Create a color version of the grayscale image to draw colored lines
        result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        height, width = image.shape
        
        # Find peaks in horizontal and vertical profiles
        for channel, color in [('red', (0, 0, 255)), ('green', (0, 255, 0)), ('blue', (255, 0, 0))]:
            horizontal_data = np.array(profile_data["horizontal_rgb"][channel])
            vertical_data = np.array(profile_data["vertical_rgb"][channel])
            
            # Find significant peaks in horizontal profile (left to right)
            threshold = np.mean(horizontal_data) + np.std(horizontal_data)
            peak_indices_h = np.where(horizontal_data > threshold)[0]
            
            # Find significant peaks in vertical profile (top to bottom)
            threshold = np.mean(vertical_data) + np.std(vertical_data)
            peak_indices_v = np.where(vertical_data > threshold)[0]
            
            # Group peaks by 10-pixel windows for horizontal lines
            window_size = 10
            for i in range(0, len(peak_indices_h), window_size):
                window_peaks = peak_indices_h[i:i+window_size]
                if len(window_peaks) > 0:
                    # Use the position with the highest intensity within the window
                    window_intensities = [horizontal_data[x] for x in window_peaks]
                    best_peak_idx = window_peaks[np.argmax(window_intensities)]
                    cv2.line(result_image, (best_peak_idx, 0), (best_peak_idx, height-1), color, 1)
            
            # Group peaks by 10-pixel windows for vertical lines
            for i in range(0, len(peak_indices_v), window_size):
                window_peaks = peak_indices_v[i:i+window_size]
                if len(window_peaks) > 0:
                    # Use the position with the highest intensity within the window
                    window_intensities = [vertical_data[y] for y in window_peaks]
                    best_peak_idx = window_peaks[np.argmax(window_intensities)]
                    cv2.line(result_image, (0, best_peak_idx), (width-1, best_peak_idx), color, 1)
        
        # Save the result for reference
        cv2.imwrite(f'{folder_path}/pattern_peaks.png', result_image)
        
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