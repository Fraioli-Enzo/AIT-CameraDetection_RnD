import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from typing import List, Tuple, Optional, Dict, Any


class ImagePreprocessing:
    @staticmethod
    def clear_image(image_path: str, kernel_size: int = 5) -> Optional[np.ndarray]:
        # Load the image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not read image {image_path}")
            return None
        
        # Apply median blur to reduce noise
        # Kernel size must be odd and greater than 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        denoised_frame = cv2.medianBlur(frame, kernel_size)
        return denoised_frame
    
    @staticmethod
    def show_histogram(frame: np.ndarray) -> None:
        if frame is None:
            print("Error: No frame provided for histogram calculation")
            return
        
        # Convert BGR to RGB (OpenCV loads as BGR by default)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create a figure with 3 subplots
        plt.figure(figsize=(15, 5))
        colors = ('r', 'g', 'b')
        channel_names = ('Red', 'Green', 'Blue')
        
        # Calculate and plot histogram for each channel
        for i, color in enumerate(colors):
            # Calculate histogram
            hist = cv2.calcHist([rgb_frame], [i], None, [256], [0, 256])
            
            # Create subplot
            plt.subplot(1, 3, i+1)
            plt.plot(hist, color=color)
            plt.title(f'{channel_names[i]} Histogram')
            plt.xlim([0, 256])
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Calculate statistics for each channel
        stats = []
        for i, name in enumerate(channel_names):
            channel = rgb_frame[:,:,i].flatten()
            mean_val = np.mean(channel)
            median_val = np.median(channel)
            std_val = np.std(channel)
            min_val = np.min(channel)
            max_val = np.max(channel)
            
            print(f"{name} Channel Statistics:")
            print(f"  Mean: {mean_val:.2f}")
            print(f"  Median: {median_val:.2f}")
            print(f"  Std Dev: {std_val:.2f}")
            print(f"  Range: {min_val} to {max_val}")
            print()
            
            stats.append({
                'channel': name,
                'mean': mean_val,
                'median': median_val,
                'std_dev': std_val,
                'min': min_val,
                'max': max_val
            })
        
        return stats
        
    
class ImageAnalysis:
    @staticmethod
    def load_image(image_path: str) -> Optional[np.ndarray]:
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not read image {image_path}")
        return frame
    
    @staticmethod
    def extract_colors(frame: np.ndarray, num_colors: int = 3) -> List[Tuple[int, int, int]]:
        if frame is None:
            return []
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pixels = frame.reshape(-1, frame.shape[-1])
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        # Check for pixels uniqueness to avoid errors if image has fewer unique colors than requested
        unique_pixels = np.unique(pixels, axis=0)
        if len(unique_pixels) < num_colors:
            num_colors = len(unique_pixels)
            if num_colors == 0:  # Handle completely blank images
                return []
        _, labels, centers = cv2.kmeans(pixels.astype(np.float32), num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        return [tuple(map(int, center)) for center in centers]
    
    @staticmethod
    def print_colors(colors: List[Tuple[int, int, int]]):
        for i, color in enumerate(colors):
            print(f"Color {i + 1}: {color}")

    @staticmethod
    def show_colors(colors: List[Tuple[int, int, int]], frame: Optional[np.ndarray] = None):
        # Create a display panel
        color_patch_width = 100
        panel_height = 100 * len(colors) if len(colors) > 0 else 100
        
        if frame is not None:
            # Calculate resize factor to match the height of color patches
            img_height, img_width = frame.shape[:2]
            resize_factor = panel_height / img_height
            resized_width = int(img_width * resize_factor)
            resized_img = cv2.resize(frame, (resized_width, panel_height))
            
            # Create panel with image + color patches
            panel = np.zeros((panel_height, resized_width + (color_patch_width * len(colors)), 3), dtype=np.uint8)
            panel[:, :resized_width] = resized_img
            
            # Add colors to panel
            for i, color in enumerate(colors):
                x_offset = resized_width + (i * color_patch_width)
                panel[:, x_offset:x_offset+color_patch_width] = color
                
            # Display the panel
            cv2.imshow("Image with Extracted Colors", panel)
        else:
            # Just show individual color patches if no image
            for i, color in enumerate(colors):
                color_patch = np.zeros((100, 100, 3), dtype=np.uint8)
                color_patch[:, :] = color
                cv2.imshow(f"Color {i + 1}", color_patch)
                
        cv2.waitKey(0)
        cv2.destroyAllWindows()

class ImageManagement:
    @staticmethod
    def contrast_increase_image(colors: List[Tuple[int, int, int]], contrast_factor: float = 1.5, 
                               frame: Optional[np.ndarray] = None) -> Tuple[List[Tuple[int, int, int]], Optional[np.ndarray]]:
        enhanced_colors = []
        enhanced_frame = None
        
        # Enhance colors
        for color in colors:
            # Create a small image with this color
            color_patch = np.zeros((1, 1, 3), dtype=np.uint8)
            color_patch[0, 0] = color
            
            # Apply contrast adjustment
            enhanced_patch = cv2.convertScaleAbs(color_patch, alpha=contrast_factor, beta=0)
            
            # Extract the enhanced color
            enhanced_color = tuple(map(int, enhanced_patch[0, 0]))
            enhanced_colors.append(enhanced_color)
        
        # Enhance the frame if provided
        if frame is not None:
            # Create a mask for each color and apply contrast only to matching regions
            enhanced_frame = frame.copy()
            
            for color in colors:
                # Create a mask for pixels close to this color (with some tolerance)
                lower_bound = np.array([max(0, c - 15) for c in color], dtype=np.uint8)
                upper_bound = np.array([min(255, c + 15) for c in color], dtype=np.uint8)
                mask = cv2.inRange(frame, lower_bound, upper_bound)
                
                # Apply contrast enhancement only to the masked areas
                color_enhanced = cv2.convertScaleAbs(frame, alpha=contrast_factor, beta=0)
                enhanced_frame = np.where(mask[:,:,np.newaxis] > 0, color_enhanced, enhanced_frame)
            
        return enhanced_colors, enhanced_frame

class ImagePipeline:
    @staticmethod
    def process_image(image_path: str) -> List[Tuple[int, int, int]]:
        frame = ImageAnalysis.load_image(image_path)
        frame = ImagePreprocessing.clear_image(image_path)

        ImagePreprocessing.show_histogram(frame)

        colors = ImageAnalysis.extract_colors(frame, 5)
        ImageAnalysis.show_colors(colors, frame)
        ImageAnalysis.print_colors(colors)
        enhanced_colors, new_frame = ImageManagement.contrast_increase_image(colors, 2, frame)
        return ImageAnalysis.show_colors(enhanced_colors, new_frame)
        
    
    @staticmethod
    def process_two_histogram(image_path1: str, image_path2: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        # Load and process both images
        frame1 = ImagePreprocessing.clear_image(image_path1)
        frame2 = ImagePreprocessing.clear_image(image_path2)
        
        # Check if images were loaded properly
        if frame1 is None or frame2 is None:
            print("Error loading one or both images")
            return [], []
        
        # Convert BGR to RGB for both frames
        rgb_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        rgb_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        
        # Create a figure with 6 subplots (3 for each image)
        plt.figure(figsize=(20, 10))
        colors = ('r', 'g', 'b')
        channel_names = ('Red', 'Green', 'Blue')
        
        # Process both images' histograms
        stats1 = []
        stats2 = []
        
        # Plot histograms for both images
        for i, color in enumerate(colors):
            # Image 1 histogram
            plt.subplot(2, 3, i+1)
            hist1 = cv2.calcHist([rgb_frame1], [i], None, [256], [0, 256])
            plt.plot(hist1, color=color)
            plt.title(f'Image 1: {channel_names[i]} Histogram')
            plt.xlim([0, 256])
            plt.grid(True, alpha=0.3)
            
            # Image 2 histogram
            plt.subplot(2, 3, i+4)
            hist2 = cv2.calcHist([rgb_frame2], [i], None, [256], [0, 256])
            plt.plot(hist2, color=color)
            plt.title(f'Image 2: {channel_names[i]} Histogram')
            plt.xlim([0, 256])
            plt.grid(True, alpha=0.3)
            
            # Calculate statistics for each channel
            channel1 = rgb_frame1[:,:,i].flatten()
            channel2 = rgb_frame2[:,:,i].flatten()
            
            stats1.append({
                'channel': channel_names[i],
                'mean': np.mean(channel1),
                'median': np.median(channel1),
                'std_dev': np.std(channel1),
                'min': np.min(channel1),
                'max': np.max(channel1)
            })
            
            stats2.append({
                'channel': channel_names[i],
                'mean': np.mean(channel2),
                'median': np.median(channel2),
                'std_dev': np.std(channel2),
                'min': np.min(channel2),
                'max': np.max(channel2)
            })
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics for both images
        print("--- Image 1 Statistics ---")
        for stat in stats1:
            print(f"{stat['channel']} Channel:")
            print(f"  Mean: {stat['mean']:.2f}")
            print(f"  Median: {stat['median']:.2f}")
            print(f"  Std Dev: {stat['std_dev']:.2f}")
            print(f"  Range: {stat['min']} to {stat['max']}")
            print()
            
        print("--- Image 2 Statistics ---")
        for stat in stats2:
            print(f"{stat['channel']} Channel:")
            print(f"  Mean: {stat['mean']:.2f}")
            print(f"  Median: {stat['median']:.2f}")
            print(f"  Std Dev: {stat['std_dev']:.2f}")
            print(f"  Range: {stat['min']} to {stat['max']}")
            print()
        
        return stats1, stats2

def main():
    root = tk.Tk()
    root.withdraw()
    image_path = filedialog.askopenfilename(title="Select an image file")
    image_path2 = filedialog.askopenfilename(title="Select an image file")
    ImagePipeline.process_two_histogram(image_path, image_path2)


if __name__ == "__main__":
    main()
 